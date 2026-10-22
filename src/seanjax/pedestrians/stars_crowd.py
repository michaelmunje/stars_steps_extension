"""Model E: pedestrians driven by the multi-agent STARS forecaster (data-driven crowd simulation).

Each pedestrian follows its reference trajectory for its first ``history`` x ``stride`` simulation
steps (3.2 s at the STEPS rate), building the observed history the forecaster needs. From then on
its motion comes from the forecaster:

* every ``replan_every`` simulation steps (0.4 s by default, the model's rate) all model-driven
  pedestrians of every environment are forecast jointly from their histories;
* the CVAE intent is temporally correlated per pedestrian: standard-normal ``eps`` drawn at reset and
  updated as an AR(1) process at every re-forecast (``intent_correlation``, default 0.8 per 0.4 s,
  ~2 s correlation time) gives ``z = mu + sigma * eps`` under the current prior. Draws are seeded and
  made on a fixed schedule, so the robot-absent counterfactual uses the same ones. (A fully held
  draw, read in each pedestrian's re-centred local frame, made pedestrians spiral.)
* between replanning steps pedestrians move along the forecast's first step (its displacement
  spread evenly over the simulation steps, applied incrementally so hand-overs are smooth);
* the robot joins the graph as one more person, so pedestrians react to it as they react to each
  other (STEPS contains no robot), and pedestrians can never interpenetrate it
  (``resolve_robot_contact``);
* the forecaster has no notion of walls, so a model-driven move that would cross one slides along
  it (``block_wall_crossings``); recorded scenes' arena box does not block pedestrians;
* a pedestrian leaves when its reference ends (recorded exits are kept).

This makes the trajectory forecaster the crowd simulator, for recorded or scripted scenes.
"""
from __future__ import annotations

import torch

from baselines.stars_ma_batch import graph_batch_from_padded
from baselines.stars_ma_graph import to_scene
from baselines.train_stars_ma import load_forecaster
from seanjax.core.geometry import block_wall_crossings
from seanjax.core.state import PedestrianState
from seanjax.pedestrians.base import PedestrianModel, reference_velocity
from seanjax.pedestrians.social_force import resolve_robot_contact


class StarsCrowdModel(PedestrianModel):
    name = "stars_crowd"

    def __init__(self, checkpoint, *, device: str = "cpu", replan_every: int = 4, include_robot: bool = True,
                 seed: int = 0, amp: bool = True, intent_correlation: float = 0.8) -> None:
        self.device = torch.device(device)
        self.model, payload = load_forecaster(checkpoint, self.device)
        self.model.eval()
        self.knn, self.radius = payload["graph"]["knn"], payload["graph"]["radius"]
        self.history = int(payload["window"]["observation_steps"])
        self.dt_model = float(payload["window"]["dt"])
        self.replan_every, self.include_robot, self.seed, self.amp = int(replan_every), include_robot, seed, amp
        self.intent_correlation = float(intent_correlation)

    def reset(self, batch):
        state = super().reset(batch)
        B, P = batch.ped_mask.shape
        stride = int(round(self.dt_model / batch.dt))
        if stride != self.replan_every:
            self.replan_every = stride
        self.stride = stride
        self.buffer = [(state.pos.clone(), state.active.clone())]
        self.robot_buffer = []
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        latent = max(1, self.model.latent_dim)
        self.eps = torch.randn((B, P + 1, latent), generator=self.generator).to(batch.ped_ref.device)
        self.plan = None                                     # [B, P, steps, 2] positions for the coming steps
        self.plan_start = 0
        self.age = torch.zeros(B, P, dtype=torch.long, device=batch.ped_ref.device)  # steps since spawn
        self.last_influence = None
        return state

    def _history(self, extra_pos=None, extra_active=None):
        """``[B, N, T, 2]`` / ``[B, N, T]`` at the model rate, newest last (optionally + the robot)."""
        frames = [self.buffer[max(0, len(self.buffer) - 1 - self.stride * (self.history - 1 - i))]
                  for i in range(self.history)]
        available = [len(self.buffer) - 1 - self.stride * (self.history - 1 - i) >= 0 for i in range(self.history)]
        xy = torch.stack([f[0] for f in frames], dim=2)
        valid = torch.stack([f[1] & a for f, a in zip(frames, available)], dim=2)
        if extra_pos is not None:
            xy = torch.cat([xy, extra_pos[:, None]], dim=1)
            valid = torch.cat([valid, extra_active[:, None]], dim=1)
        return xy, valid

    def _forecast(self, batch, robot, robot_present) -> torch.Tensor:
        """Next-step positions, and (``self.last_influence``) how much the robot changed each forecast."""
        nxt, full_with = self._forecast_once(batch, robot, robot_present, self.include_robot)
        self.last_influence = torch.zeros_like(batch.ped_mask, dtype=nxt.dtype)
        if self.include_robot and bool(robot_present.any()):
            _, full_without = self._forecast_once(batch, robot, robot_present, False)
            change = torch.linalg.norm(full_with - full_without, dim=-1).mean(-1)   # [B, P] over 4.8 s
            self.last_influence = torch.where(robot_present[:, None], change, torch.zeros_like(change)).clamp(min=1e-9) \
                * self._model_driven
        return nxt

    def _forecast_once(self, batch, robot, robot_present, include_robot: bool):
        """Next-step positions ``[B, P, 2]`` (scene frame) after one model step for model-driven pedestrians."""
        B, P = batch.ped_mask.shape
        extra_pos = extra_valid = None
        if include_robot:
            frames = [self.robot_buffer[max(0, len(self.robot_buffer) - 1 - self.stride * (self.history - 1 - i))]
                      for i in range(self.history)] if self.robot_buffer else [robot.pos] * self.history
            extra_pos = torch.stack(frames, dim=1)
            extra_valid = robot_present[:, None].expand(B, self.history).clone()
        xy, valid = self._history(extra_pos, extra_valid)
        node_mask = valid[..., -1] & (valid.sum(-1) >= 2)
        N = node_mask.shape[1]
        order = torch.argsort((~node_mask).to(torch.int8), dim=1, stable=True)
        gather = lambda value: torch.gather(value, 1, order.view(B, N, *([1] * (value.dim() - 2))).expand_as(value))
        packed_mask = torch.gather(node_mask, 1, order)
        out = xy[:, :, -1].clone()
        full = out[:, :, None].expand(B, N, self.model.prediction_steps, 2).clone()
        if not bool(packed_mask.any()):
            return out[:, :P], full[:, :P]
        graph = graph_batch_from_padded(gather(xy.to(self.device)), gather(valid.to(self.device)), packed_mask.to(self.device),
                                        dt=self.dt_model, knn=self.knn, radius=self.radius,
                                        prediction_steps=self.model.prediction_steps)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                             enabled=self.amp and self.device.type == "cuda"):
            encoded = self.model.encode(graph)
            z = None
            if self.model.latent_dim:
                mu, logvar = self.model.decoder.prior_params(encoded["context"])
                eps = torch.gather(self.eps.to(self.device)[:, :N], 1,
                                   order.to(self.device)[..., None].expand(B, N, mu.shape[-1]))
                eps = eps[graph.node_mask]
                z = (mu + torch.exp(0.5 * logvar) * eps.to(mu.dtype))[None]
            local = self.model.decode(graph, encoded, z)[0].float()
        scene = to_scene(local, graph.origin, graph.heading)                       # [Nt, 12, 2]
        packed = graph.to_padded(scene)                                             # [B, N, 12, 2]
        unpacked = torch.zeros_like(packed)
        unpacked.scatter_(1, order.to(self.device)[..., None, None].expand_as(packed), packed)
        have = node_mask.to(self.device)
        full = torch.where(have[..., None, None], unpacked, full.to(self.device))
        out = torch.where(have[..., None], unpacked[:, :, 0], out.to(self.device))
        device = batch.ped_ref.device
        return out[:, :P].to(device), full[:, :P].to(device)

    def step(self, state, batch, k, robot, robot_radius, robot_present) -> PedestrianState:
        K = batch.steps
        kk = min(k, K - 1)
        spawn = (self.first == k) & batch.ped_mask & ~state.active & ~state.done
        leave = state.active & (k > self.last)
        active = (state.active & ~leave) | spawn
        self.age = torch.where(spawn, torch.zeros_like(self.age), self.age + state.active.long())
        warm = self.age < self.history * self.stride                               # still on its reference
        self._model_driven = (state.active & ~warm).to(state.pos.dtype)
        self.last_influence = None
        if (k - 1) % self.replan_every == 0 or self.plan is None:
            if self.plan is not None and self.intent_correlation < 1.0:
                # Intents drift (AR(1), correlation time ~ replan / (1 - rho)): a fixed draw is read
                # in each pedestrian's re-centred local frame every re-forecast and makes them spiral.
                rho = self.intent_correlation
                fresh = torch.randn(self.eps.shape, generator=self.generator).to(self.eps.device)
                self.eps = rho * self.eps + (1.0 - rho * rho) ** 0.5 * fresh
            target = self._forecast(batch, robot, robot_present)
            # Per-step displacement toward the forecast's next model-rate position: applied to the
            # current position each step, so a pedestrian handed over mid-plan does not jump.
            self.plan = (target - state.pos) / self.replan_every
            self.plan_start = k
        model_pos = state.pos + self.plan
        pos = torch.where((warm | spawn)[..., None], batch.ped_ref[:, :, kk], model_pos)
        # The forecaster knows nothing of walls: model-driven moves that would cross one slide along it.
        blocked = block_wall_crossings(state.pos, pos, batch.segments,
                                       batch.segment_mask & batch.ped_walls[:, None])
        pos = torch.where((~(warm | spawn))[..., None], blocked, pos)
        pos = torch.where(active[..., None], pos, state.pos)
        vel = (pos - state.pos) / batch.dt
        vel = torch.where(spawn[..., None], reference_velocity(batch, kk), vel)
        pos, vel = resolve_robot_contact(pos, vel, batch, robot, robot_radius, robot_present)
        vel = torch.where(active[..., None], vel, 0.0)
        self.buffer.append((pos.clone(), active.clone()))
        self.robot_buffer.append(robot.pos.clone())
        keep = (self.history - 1) * self.stride + 1
        self.buffer, self.robot_buffer = self.buffer[-keep:], self.robot_buffer[-keep:]
        return PedestrianState(pos, vel, active, state.done | leave)
