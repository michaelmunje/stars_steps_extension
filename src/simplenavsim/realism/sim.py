"""Closed-loop simulators for ETH/UCY windows: CV, goal-directed straight line, SFM, ORCA and STARS.

Every simulator maps padded inputs ``obs [B, N, OBS, 2]``, ``obs_valid [B, N, OBS]``, ``goal [B, N, 2]``
(``agents [B, N]`` marks real cohort members) to positions ``[B, N, steps, 2]`` at 0.4 s, all agents
jointly; only ``Window.inputs()`` is used, so the ground-truth future cannot leak in. SFM and ORCA
integrate at 0.1 s; STARS re-forecasts every 0.4 s from its own simulated history and executes the
forecast's first step (closed loop). The ``goal`` adapter steers STARS' executed steps toward the
oracle goal at the forecast's own speed (an inference-time adapter). A goal-conditioned checkpoint
(``decoder_recipe["goal"]``, ``stars_noethucy_goal``) instead reads every agent's oracle goal as an input.
``Stars(mode=True)`` uses the CVAE prior mean as the latent (deterministic "mode" rollout, no sampling).
STARS diffusion checkpoints run the same loop: every re-forecast is one DDIM chain whose initial noise is
temporally correlated across re-forecasts, as the CVAE latent is.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from simplenavsim.core.state import PedestrianState
from simplenavsim.pedestrians.orca import OrcaParams, orca_velocities
from simplenavsim.pedestrians.social_force import SocialForceParams, integrate, social_forces

from .data import DT, OBS

REPO = Path(__file__).resolve().parents[3]
CHECKPOINTS = {"stars_steps": REPO / "experiments/stars_ma/runs/cvae_kl01_finetune/checkpoints/best.pt",
               "stars_pooled": REPO / "experiments/stars_ma/runs/cvae_all_lr3e-4_seed42/checkpoints/best.pt",
               # Trained without ETH or UCY (configs/stars_ma/all_no_ethucy.yaml): zero-shot on every scene.
               "stars_noethucy": REPO / "experiments/stars_ma/runs/cvae_noethucy_nogoal/checkpoints/best.pt",
               "stars_noethucy_goal": REPO / "experiments/stars_ma/runs/cvae_noethucy_goal/checkpoints/best.pt",
               # results/stars_improve_2026-11-15: all 7 datasets, seed 2 (best validation); used with oversample=4.
               "stars_improved": REPO / "experiments/stars_ma/runs/si_final_base_s2/checkpoints/best.pt",
               # Collision-aware variant: refinement round + decoder-only collision loss, KL 0.15.
               "stars_improved_social": REPO / "experiments/stars_ma/runs/si_final_social_s42/checkpoints/best.pt",
               # STEPS-only diffusion decoders (per agent / joint over the graph), zero-shot on ETH/UCY.
               "stars_steps_diffusion": REPO / "experiments/stars_ma/runs/diffusion_finetune/checkpoints/best.pt",
               "stars_steps_joint_diffusion": REPO / "experiments/stars_ma/runs/joint_diffusion_finetune/checkpoints/best.pt",
               # results/stars_rollout_ft_2026-11-18: stars_improved(_social) fine-tuned on their own closed-loop
               # 12-step rollouts (train_stars_ma --init-checkpoint ... --rft-steps 12).
               "stars_rft": REPO / "experiments/stars_ma/runs/rft_base_noatc/checkpoints/best.pt",
               "stars_rft_social": REPO / "experiments/stars_ma/runs/rft_social_coll_progress_noatc/checkpoints/best.pt"}
# ETH/UCY protocol (branch ethucy-protocol, results/ethucy_protocol_*): A = pretrained on STEPS, PTDPES, MADRAS,
# Grand Central only (zero-shot); B = A fine-tuned leave-one-scene-out, one checkpoint per held-out scene
# (PER_DATASET: each window uses the checkpoint that never saw its scene).
_EP = REPO / "experiments/forecast_bench/loso"
CHECKPOINTS["stars_ep_zs"] = REPO / "experiments/stars_ma/runs/ep_pre4_s{seed}/checkpoints/best.pt"
CHECKPOINTS["stars_ep_zs_rft"] = REPO / "experiments/stars_ma/runs/ep_rft_zs_s{seed}/checkpoints/best.pt"
_FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
PER_DATASET = {"stars_ep_loso": {f: _EP / "ep_loso_s{seed}" / f / "best.pt" for f in _FOLDS},
               "stars_ep_loso_rft": {f: _EP / "ep_loso_rft_s{seed}" / f / "best.pt" for f in _FOLDS}}
EP_SEED = __import__("os").environ.get("EP_SEED", "42")   # pretraining seed used in the closed loop (env EP_SEED)
OVERSAMPLE = {"stars_improved": 4, "stars_ep_zs": 4, "stars_ep_loso": 4, "stars_ep_zs_rft": 4, "stars_ep_loso_rft": 4}   # closed-loop rollouts per window; the least-colliding one is kept
RADIUS = 0.2              # pedestrian body radius (m) used by SFM / ORCA
ARRIVAL = 0.3             # goal-directed models stop within this distance of the goal (m)
DEFAULT_SPEED = 1.0       # desired speed of an agent with a single observed position (context agents only)
SUBSTEPS = 4              # SFM / ORCA integration at 0.1 s


class PerDataset(dict):
    """``{dataset: Stars}``: routes each window group to the checkpoint of its dataset (``dataset=`` keyword)."""

    def __call__(self, x, steps, seed: int = 0, dataset: str = "", **kw):
        return self[dataset](x, steps, seed=seed, **kw)

    def with_oversample(self, m: int) -> "PerDataset":
        return PerDataset({k: v.with_oversample(m) for k, v in self.items()})

    def with_mode(self) -> "PerDataset":
        return PerDataset({k: v.with_mode() for k, v in self.items()})


def pad(windows) -> dict:
    """Stack ``Window.inputs()`` of several windows into padded tensors (float64 numpy)."""
    n = max(len(w.agent_ids) for w in windows)
    B = len(windows)
    obs, valid = np.zeros((B, n, OBS, 2)), np.zeros((B, n, OBS), bool)
    goal, agents = np.zeros((B, n, 2)), np.zeros((B, n), bool)
    for b, w in enumerate(windows):
        x = w.inputs()
        k = len(w.agent_ids)
        last = x["obs"][:, -1]
        obs[b, :k] = np.where(x["obs_valid"][..., None], x["obs"], last[:, None])   # missing -> current position
        valid[b, :k], goal[b, :k], agents[b, :k] = x["obs_valid"], x["goal"], True
    return {"obs": obs, "obs_valid": valid, "goal": goal, "agents": agents}


def observed_motion(obs, valid):
    """Last observed velocity ``[B, N, 2]`` and mean observed speed ``[B, N]``."""
    step_ok = valid[..., 1:] & valid[..., :-1]
    disp = np.diff(obs, axis=2)
    vel = np.where(step_ok[..., -1:], disp[:, :, -1], 0.0) / DT
    speeds = np.linalg.norm(disp, axis=-1) / DT
    count = step_ok.sum(-1)
    mean = np.where(count > 0, (speeds * step_ok).sum(-1) / np.maximum(count, 1), DEFAULT_SPEED)
    return vel, mean


def _toward(pos, goal, speed, dt):
    """Velocity toward ``goal`` at ``speed``, slowing so as not to overshoot and stopping on arrival."""
    rel = goal - pos
    dist = np.linalg.norm(rel, axis=-1, keepdims=True)
    unit = rel / np.maximum(dist, 1e-9)
    return np.where(dist > ARRIVAL, unit * np.minimum(speed[..., None], dist / dt), 0.0)


def cv(x, steps, **_):
    vel, _ = observed_motion(x["obs"], x["obs_valid"])
    k = np.arange(1, steps + 1)[None, None, :, None]
    return x["obs"][:, :, -1:] + vel[:, :, None] * k * DT


def goal_straight(x, steps, **_):
    _, speed = observed_motion(x["obs"], x["obs_valid"])
    pos, out = x["obs"][:, :, -1].copy(), []
    for _ in range(steps):
        pos = pos + _toward(pos, x["goal"], speed, DT) * DT
        out.append(pos)
    return np.stack(out, 2)


def _batch(agents, radius):
    B, N = agents.shape
    return SimpleNamespace(ped_radius=torch.full((B, N), radius, dtype=torch.float64),
                           circles=torch.zeros(B, 0, 3, dtype=torch.float64), circle_mask=torch.zeros(B, 0, dtype=torch.bool),
                           segments=torch.zeros(B, 0, 4, dtype=torch.float64), segment_mask=torch.zeros(B, 0, dtype=torch.bool))


def sfm(x, steps, params: SocialForceParams | None = None, **_):
    """Helbing-style social force (``simplenavsim.pedestrians.social_force``), desired velocity toward the goal."""
    p = params or SocialForceParams()
    vel0, speed = observed_motion(x["obs"], x["obs_valid"])
    agents = torch.as_tensor(x["agents"])
    batch = _batch(agents, RADIUS)
    B = agents.shape[0]
    robot = SimpleNamespace(pos=torch.full((B, 2), 1e6, dtype=torch.float64))
    absent = torch.zeros(B, dtype=torch.bool)
    state = PedestrianState(torch.as_tensor(x["obs"][:, :, -1].copy()), torch.as_tensor(vel0), agents, ~agents)
    speed_t = torch.as_tensor(speed)
    max_speed = (p.max_speed_factor * speed_t).clamp(min=0.5)
    dt, out = DT / SUBSTEPS, []
    for _ in range(steps):
        for _ in range(SUBSTEPS):
            desired = torch.as_tensor(_toward(state.pos.numpy(), x["goal"], speed, DT))
            acc = social_forces(state, desired, batch, robot, 0.0, absent, p)
            pos, vel = integrate(state, acc, max_speed, dt)
            state = PedestrianState(pos, vel, state.active, state.done)
        out.append(state.pos.numpy().copy())
    return np.stack(out, 2)


ORCA_PUBLISHED = OrcaParams(neighbor_dist=10.0, max_neighbors=10, time_horizon=5.0, mode="goal", obstacles=False)


def orca(x, steps, params: OrcaParams | None = None, radius: float = 0.3, **_):
    """ORCA (``simplenavsim.pedestrians.orca``, RVO2 port) with preferred velocity toward the goal.

    Defaults (``ORCA_PUBLISHED``, ``radius`` 0.3 m) are the CrowdNav pedestrian settings (neighbour
    distance 10 m, 10 neighbours, 5 s horizon, 0.3 m radius, preferred speed = observed speed).
    """
    p = params or ORCA_PUBLISHED
    vel0, speed = observed_motion(x["obs"], x["obs_valid"])
    agents = torch.as_tensor(x["agents"])
    B, N = agents.shape
    pos, vel = torch.as_tensor(x["obs"][:, :, -1].copy()), torch.as_tensor(vel0)
    rad = torch.full((B, N), radius, dtype=torch.float64)
    max_speed = (p.max_speed_factor * torch.as_tensor(speed)).clamp(min=0.5)
    dt, out = DT / SUBSTEPS, []
    for _ in range(steps):
        for _ in range(SUBSTEPS):
            pref = torch.as_tensor(_toward(pos.numpy(), x["goal"], speed, DT))
            vel = orca_velocities(pos, vel, pref, agents, rad, max_speed, dt, p)
            pos = pos + vel * dt
        out.append(pos.numpy().copy())
    return np.stack(out, 2)


@dataclass
class Stars:
    """Closed-loop STARS crowd: re-forecast every 0.4 s, execute the first step (as ``stars_crowd``)."""
    checkpoint: Path
    device: str = "cpu"
    intent_correlation: float = 0.8
    goal_weight: float = 0.0          # > 0: inference-time goal-steering adapter
    oversample: int = 1               # > 1: M rollouts per window, keep the one with the fewest close pairs
    mode: bool = False                # CVAE: latent = prior mean (deterministic rollout)

    def __post_init__(self):
        from baselines.train_stars_ma import load_forecaster
        self.model, payload = load_forecaster(Path(self.checkpoint), torch.device(self.device))
        self.model.eval()
        self.knn, self.radius = payload["graph"]["knn"], payload["graph"]["radius"]
        assert int(payload["window"]["observation_steps"]) == OBS and float(payload["window"]["dt"]) == DT

    def with_goal(self, weight: float) -> "Stars":
        other = object.__new__(Stars)
        other.__dict__.update(self.__dict__)
        other.goal_weight = weight
        return other

    def with_oversample(self, m: int) -> "Stars":
        other = object.__new__(Stars)
        other.__dict__.update(self.__dict__)
        other.oversample = m
        return other

    def with_mode(self) -> "Stars":
        """Deterministic copy: prior-mean latent, a single rollout."""
        other = self.with_oversample(1)
        other.mode = True
        return other

    def __call__(self, x, steps, seed: int = 0, hook=None, **_):
        """Closed-loop rollout [B, N, steps, 2]. With ``oversample`` M > 1, M rollouts (seeds seed*M .. seed*M+M-1)
        are drawn per window and the one with the fewest close pairs (< 0.4 m at any step) is kept; pairs that look
        like a walking group in the observation (within 1 m, velocities within 0.3 m/s) are not counted. Only the
        model's own rollouts and the observation are used (no future information)."""
        if self.oversample <= 1:
            return self._rollout(x, steps, seed, hook)
        runs = np.stack([self._rollout(x, steps, seed * self.oversample + j, hook) for j in range(self.oversample)])
        agents, obs = np.asarray(x["agents"], bool), np.asarray(x["obs"], float)
        vel = (obs[:, :, -1] - obs[:, :, -2]) / DT
        group = (np.linalg.norm(obs[:, :, None, -1] - obs[:, None, :, -1], axis=-1) < 1.0) & \
                (np.linalg.norm(vel[:, :, None] - vel[:, None], axis=-1) < 0.3)
        n = agents.shape[1]
        pair = agents[:, :, None] & agents[:, None, :] & ~group & np.triu(np.ones((n, n), bool), 1)
        dist = np.linalg.norm(runs[:, :, :, None] - runs[:, :, None, :], axis=-1).min(-1)   # [M, B, N, N]
        close = ((dist < 0.4) & pair).sum((-1, -2))                                        # [M, B]
        best = close.argmin(0)
        return runs[best, np.arange(runs.shape[1])]

    @torch.no_grad()
    def _rollout(self, x, steps, seed: int = 0, hook=None):
        """``hook(k, positions) -> positions`` may edit each executed step (tests: closed-loop check)."""
        from baselines.stars_ma_batch import graph_batch_from_padded
        from baselines.stars_ma_graph import to_scene
        dev = torch.device(self.device)
        xy = torch.as_tensor(x["obs"], dtype=torch.float32, device=dev)
        valid = torch.as_tensor(x["obs_valid"], device=dev) & torch.as_tensor(x["agents"], device=dev)[..., None]
        goal = torch.as_tensor(x["goal"], dtype=torch.float32, device=dev)
        B, N = valid.shape[:2]
        gen = torch.Generator().manual_seed(seed)
        latent = 2 * self.model.prediction_steps if self.model.diffusion else max(1, self.model.latent_dim)
        eps = torch.randn((B, N, latent), generator=gen)
        rho, out = self.intent_correlation, []
        for k in range(steps):
            if k:
                eps = rho * eps + (1 - rho * rho) ** 0.5 * torch.randn((B, N, latent), generator=gen)
            node_mask = valid[..., -1] & (valid.sum(-1) >= 2)
            order = torch.argsort((~node_mask).to(torch.int8), dim=1, stable=True)
            take = lambda v: torch.gather(v, 1, order.view(B, N, *([1] * (v.dim() - 2))).expand_as(v))
            cur = xy[:, :, -1]
            nxt = cur.clone()
            if bool(node_mask.any()):
                packed = torch.gather(node_mask, 1, order)
                goals = None
                if self.model.decoder_recipe.get("goal"):   # goal-conditioned checkpoint: oracle goal as input
                    goals = take(torch.cat([goal, torch.ones_like(goal[..., :1])], -1))
                graph = graph_batch_from_padded(take(xy), take(valid), packed, dt=DT, knn=self.knn, radius=self.radius,
                                                prediction_steps=self.model.prediction_steps, goal=goals)
                encoded = self.model.encode(graph)
                e = take(eps.to(dev))[graph.node_mask]
                if self.model.diffusion:    # one DDIM chain from the correlated initial noise
                    hidden = self.model.decoder.hidden(encoded["context"], graph.edge_index, encoded["edge_emb"])
                    local = self.model.decoder.sample(hidden, graph.node_attr[:, -1, 2:4], self.model.dt, 1, graph.edge_index,
                                                      encoded["edge_emb"], noise=e)[0].float()
                else:
                    z = None
                    if self.model.latent_dim:
                        mu, logvar = self.model.decoder.prior_params(encoded["context"])
                        z = (mu if self.mode else mu + torch.exp(0.5 * logvar) * e)[None]
                    local = self.model.decode(graph, encoded, z)[0].float()
                first = graph.to_padded(to_scene(local, graph.origin, graph.heading)[:, 0])   # [B, N, 2] packed
                unpacked = torch.zeros_like(first).scatter_(1, order[..., None].expand_as(first), first)
                nxt = torch.where(node_mask[..., None], unpacked, cur)
            if self.goal_weight:
                step = nxt - cur
                size = torch.linalg.norm(step, dim=-1, keepdim=True)
                rel = goal - cur
                dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
                steer = rel / dist.clamp(min=1e-9) * torch.minimum(size, dist)
                nxt = torch.where(dist > ARRIVAL, cur + (1 - self.goal_weight) * step + self.goal_weight * steer, cur)
            if hook is not None:
                nxt = hook(k, nxt)
            out.append(nxt)
            xy = torch.cat([xy[:, :, 1:], nxt[:, :, None]], 2)
            valid = torch.cat([valid[:, :, 1:], valid[:, :, -1:]], 2)
        return torch.stack(out, 2).double().cpu().numpy()


def calibrated(params: dict | None, kind: str):
    """Calibrated parameter objects from a ``calibration.json`` entry."""
    if kind == "sfm":
        return replace(SocialForceParams(), **(params or {}))
    if kind == "orca":
        params = dict(params or {})
        radius = params.pop("radius", 0.3)
        return replace(ORCA_PUBLISHED, **params), radius
    raise KeyError(kind)
