"""Model E: pedestrians driven by the multi-agent STARS forecaster (data-driven crowd simulation).

Each pedestrian follows its reference trajectory for its first ``history`` x ``stride`` simulation
steps (3.2 s at the STEPS rate), building the observed history the forecaster needs. From then on
its motion comes from the forecaster:

* every ``replan_every`` simulation steps (0.4 s by default, the model's rate) all model-driven
  pedestrians of every environment are forecast jointly from their histories;
* the CVAE intent is held per pedestrian: a standard-normal draw ``eps`` made at reset (seeded, so
  the robot-absent counterfactual uses the same draws) gives ``z = mu + sigma * eps`` under the
  current prior at every replanning step;
* between replanning steps pedestrians move along the forecast's first step (its displacement
  spread evenly over the simulation steps, applied incrementally so hand-overs are smooth);
* the robot joins the graph as one more person, so pedestrians react to it as they react to each
  other (STEPS contains no robot), and pedestrians can never interpenetrate it
  (``resolve_robot_contact``);
* a pedestrian leaves when its reference ends (recorded exits are kept).

This makes the trajectory forecaster the crowd simulator, for recorded or scripted scenes.
"""
from __future__ import annotations

import torch

from baselines.stars_ma_batch import graph_batch_from_padded
from baselines.stars_ma_graph import to_scene
from baselines.train_stars_ma import load_forecaster
from seanjax.core.state import PedestrianState
from seanjax.pedestrians.base import PedestrianModel, reference_velocity
from seanjax.pedestrians.social_force import resolve_robot_contact


class StarsCrowdModel(PedestrianModel):
    name = "stars_crowd"

    def __init__(self, checkpoint, *, device: str = "cpu", replan_every: int = 4, include_robot: bool = True,
                 seed: int = 0, amp: bool = True) -> None:
        self.device = torch.device(device)
        self.model, payload = load_forecaster(checkpoint, self.device)
        self.model.eval()
        self.knn, self.radius = payload["graph"]["knn"], payload["graph"]["radius"]
        self.history = int(payload["window"]["observation_steps"])
        self.dt_model = float(payload["window"]["dt"])
        self.replan_every, self.include_robot, self.seed, self.amp = int(replan_every), include_robot, seed, amp

    def reset(self, batch):
        state = super().reset(batch)
        B, P = batch.ped_mask.shape
        stride = int(round(self.dt_model / batch.dt))
        if stride != self.replan_every:
            self.replan_every = stride
        self.stride = stride
        self.buffer = [(state.pos.clone(), state.active.clone())]
        self.robot_buffer = []
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        latent = max(1, self.model.latent_dim)
        self.eps = torch.randn((B, P + 1, latent), generator=generator).to(batch.ped_ref.device)
        self.plan = None                                     # [B, P, steps, 2] positions for the coming steps
        self.plan_start = 0
        self.age = torch.zeros(B, P, dtype=torch.long, device=batch.ped_ref.device)  # steps since spawn
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
        """Next-step positions ``[B, P, 2]`` (scene frame) after one model step for model-driven pedestrians."""
        B, P = batch.ped_mask.shape
        extra_pos = extra_valid = None
        if self.include_robot:
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
        if not bool(packed_mask.any()):
            return out[:, :P]
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
        first = to_scene(local[:, :1], graph.origin, graph.heading)[:, 0]          # [Nt, 2]
        packed = graph.to_padded(first)                                             # [B, N, 2]
        nxt = torch.zeros_like(packed)
        nxt.scatter_(1, order.to(self.device)[..., None].expand_as(packed), packed)
        have = node_mask.to(self.device)
        out = torch.where(have[..., None], nxt, out.to(self.device))
        return out[:, :P].to(batch.ped_ref.device)

    def step(self, state, batch, k, robot, robot_radius, robot_present) -> PedestrianState:
        K = batch.steps
        kk = min(k, K - 1)
        spawn = (self.first == k) & batch.ped_mask & ~state.active & ~state.done
        leave = state.active & (k > self.last)
        active = (state.active & ~leave) | spawn
        self.age = torch.where(spawn, torch.zeros_like(self.age), self.age + state.active.long())
        warm = self.age < self.history * self.stride                               # still on its reference
        if (k - 1) % self.replan_every == 0 or self.plan is None:
            target = self._forecast(batch, robot, robot_present)
            # Per-step displacement toward the forecast's next model-rate position: applied to the
            # current position each step, so a pedestrian handed over mid-plan does not jump.
            self.plan = (target - state.pos) / self.replan_every
            self.plan_start = k
        model_pos = state.pos + self.plan
        pos = torch.where((warm | spawn)[..., None], batch.ped_ref[:, :, kk], model_pos)
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
