"""Learned subgoals for the tuned controller, trained with PPO (cf. Brito et al., RA-L 2021).

Every ``hold`` control steps a network picks a subgoal: an offset of up to ``max_offset`` metres
from the real goal, in the robot's frame. The social controller (its arcs, costs, hard safety
constraint and tuned parameters, unchanged) then drives toward that subgoal. The offset head
starts at zero, so the untrained policy *is* the controller; learning can only change where it
steers. Observation: 64 egocentric numbers (goal, own motion, the 6 nearest pedestrians with the
controller's 1 s and 2 s forecasts, nearest obstacle).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch import nn

from simplenavsim.core.geometry import obstacle_clearance
from simplenavsim.core.state import Observation
from simplenavsim.policies.base import Policy

K_PEDS = 6
OBS_DIM = 3 + 4 + K_PEDS * 9 + 3


def _rotate(vec, cos, sin):
    """World-frame vectors ``[..., 2]`` into the robot frame (the inverse with ``-sin``)."""
    return torch.stack([cos * vec[..., 0] + sin * vec[..., 1], -sin * vec[..., 0] + cos * vec[..., 1]], -1)


def features(obs: Observation, future: torch.Tensor | None, plan_dt: float, robot_radius: float = 0.3,
             max_angular_speed: float = 1.5) -> torch.Tensor:
    """``[B, OBS_DIM]`` observation; ``future [B, P, H, 2]`` is the controller's pedestrian forecast."""
    B, device = obs.robot_pose.shape[0], obs.robot_pose.device
    xy, cos, sin = obs.robot_pose[:, :2], torch.cos(obs.robot_pose[:, 2]), torch.sin(obs.robot_pose[:, 2])
    to_goal = _rotate(obs.goal - xy, cos, sin)
    dist = torch.linalg.norm(to_goal, dim=-1).clamp(min=1e-6)
    goal = torch.stack([dist.clamp(max=10.0) / 10.0, to_goal[:, 0] / dist, to_goal[:, 1] / dist], -1)
    own = torch.stack([obs.robot_v, obs.robot_w / max_angular_speed, obs.last_command[:, 0],
                       obs.last_command[:, 1] / max_angular_speed], -1)
    peds = torch.zeros(B, K_PEDS, 9, device=device)
    if future is not None and obs.ped_pos.shape[1]:
        rel = obs.ped_pos - xy[:, None]
        gap = torch.linalg.norm(rel, dim=-1) - obs.ped_radius - robot_radius
        gap = torch.where(obs.ped_visible, gap, torch.full_like(gap, 1e6))
        k = min(K_PEDS, gap.shape[1])
        order = gap.topk(k, dim=1, largest=False).indices
        pick = lambda t: torch.gather(t, 1, order.view(B, k, *([1] * (t.dim() - 2))).expand(B, k, *t.shape[2:]))
        at = lambda s: pick(future[:, :, min(future.shape[2], round(s / plan_dt)) - 1]) - xy[:, None]
        present = (pick(gap) < 1e5).float()[..., None]
        rot = lambda v: _rotate(v, cos[:, None], sin[:, None]) / 5.0
        robot_vel = torch.stack([obs.robot_v * cos, obs.robot_v * sin], -1)[:, None]
        block = torch.cat([rot(pick(rel)), rot(pick(obs.ped_vel) - robot_vel) * 2.5, rot(at(1.0)), rot(at(2.0)), present], -1)
        peds[:, :k] = block * present
    clearance, away = obstacle_clearance(xy[:, None], obs.circles, obs.circle_mask, obs.segments, obs.segment_mask)
    obstacle = torch.cat([(clearance[:, :1] - robot_radius).clamp(0.0, 3.0) / 3.0, _rotate(away[:, 0], cos, sin)], -1)
    return torch.nan_to_num(torch.cat([goal, own, peds.reshape(B, -1), obstacle], -1))


class SubgoalNet(nn.Module):
    """Gaussian over a 2-D offset (pre-``tanh``) and a value; the mean head starts at exactly zero."""

    def __init__(self, hidden: int = 256, init_std: float = 0.3) -> None:
        super().__init__()
        self.hidden = hidden
        self.body = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mean, self.value = nn.Linear(hidden, 2), nn.Linear(hidden, 1)
        nn.init.zeros_(self.mean.weight), nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((2,), float(torch.log(torch.tensor(init_std)))))

    def forward(self, x):
        h = self.body(x)
        return torch.distributions.Normal(self.mean(h), self.log_std.exp()), self.value(h)[:, 0]


class SubgoalPolicy(Policy):
    """The controller, steered by learned subgoals. ``sample=True`` explores and records decisions."""

    name = "subgoal"
    parameters = ()

    def __init__(self, controller, controller_values: dict, net: SubgoalNet, *, hold: int = 10,
                 max_offset: float = 3.0) -> None:
        self.controller, self.controller_values, self.net = controller, controller_values, net
        self.hold, self.max_offset, self.sample = hold, max_offset, False
        self.last_info: dict = {}

    def reset(self, batch_size: int) -> None:
        self.controller.reset(batch_size)
        self.k, self.subgoal, self.decisions = 0, None, []

    def act(self, obs: Observation, params: torch.Tensor) -> torch.Tensor:
        c = self.controller.config
        if self.k % self.hold == 0:
            x = features(obs, self.controller.last_info.get("predicted") if self.k else None, c.plan_dt,
                         c.robot_radius, c.max_angular_speed)
            with torch.no_grad():
                dist, value = self.net(x)
            raw = dist.sample() if self.sample else dist.mean
            offset = torch.tanh(raw) * self.max_offset                               # robot frame
            cos, sin = torch.cos(obs.robot_pose[:, 2]), torch.sin(obs.robot_pose[:, 2])
            self.subgoal = obs.goal + _rotate(offset, cos, -sin)
            if self.sample:
                self.decisions.append({"k": self.k, "x": x, "raw": raw, "logp": dist.log_prob(raw).sum(-1),
                                       "value": value})
        self.k += 1
        values = self.controller.parameter_tensor(self.controller_values, obs.robot_pose.shape[0], obs.robot_pose.device)
        command = self.controller.act(replace(obs, goal=self.subgoal), values)
        self.last_info = {**self.controller.last_info, "subgoal": self.subgoal}
        return command

    def save(self, directory: Path, stem: str = "best") -> Path:
        torch.save({"state_dict": self.net.state_dict(), "hidden": self.net.hidden}, directory / f"{stem}.pt")
        path = directory / f"{stem}_policy.yaml"
        path.write_text(yaml.safe_dump({"policy": "subgoal", "checkpoint": f"{stem}.pt", "hold": self.hold,
                                        "max_offset": self.max_offset,
                                        "controller_parameters": {k: float(v) for k, v in self.controller_values.items()}},
                                       sort_keys=False))
        return path

    @classmethod
    def load(cls, path, controller, device="cpu") -> "SubgoalPolicy":
        path = Path(path)
        spec = yaml.safe_load(path.read_text())
        payload = torch.load(path.parent / spec["checkpoint"], map_location=device)
        net = SubgoalNet(payload["hidden"]).to(device)
        net.load_state_dict(payload["state_dict"])
        return cls(controller, spec["controller_parameters"], net.eval(), hold=spec["hold"], max_offset=spec["max_offset"])
