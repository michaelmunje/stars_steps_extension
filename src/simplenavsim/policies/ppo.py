"""PPO navigation policy: a network chooses one of the social controller's arcs (decision 98).

The action space is the controller's grid of constant ``(v, w)`` arcs. Arcs that fail the
controller's hard safety check (a pedestrian within ``safety_margin`` inside ``safety_horizon``, or
a wall) are masked out, so the learned policy keeps exactly the tuned controller's safety
constraint; with no arc left the robot stops. The network sees 64 egocentric numbers: the goal, the
robot's own motion, the 6 nearest visible pedestrians (relative position and velocity, and the
predictor's forecast 1 s and 2 s ahead) and the nearest obstacle. It outputs arc logits and a value.
"""
from __future__ import annotations

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
    """World-frame vectors ``[..., 2]`` into the robot frame."""
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


class ActorCritic(nn.Module):
    def __init__(self, arcs: int, hidden: int = 256) -> None:
        super().__init__()
        self.arcs, self.hidden = arcs, hidden
        self.body = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.logits, self.value = nn.Linear(hidden, arcs), nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        h = self.body(x)
        logits = torch.where(valid, self.logits(h), torch.full_like(valid, -1e9, dtype=h.dtype))
        return logits, self.value(h)[:, 0]


class PPOPolicy(Policy):
    """Greedy (or sampled) arc choice by ``net``; the wrapped controller supplies arcs, masks and forecasts."""

    name = "ppo"
    parameters = ()

    def __init__(self, controller, controller_values: dict, net: ActorCritic) -> None:
        self.controller, self.controller_values, self.net = controller, controller_values, net
        self.last_info: dict = {}

    def reset(self, batch_size: int) -> None:
        self.controller.reset(batch_size)

    def observe(self, obs: Observation):
        """Features ``[B, OBS_DIM]``, arc validity ``[B, M]`` and the controller's own choice ``[B]``."""
        params = self.controller.parameter_tensor(self.controller_values, obs.robot_pose.shape[0], obs.robot_pose.device)
        self.controller.act(obs, params)
        info, c = self.controller.last_info, self.controller.config
        x = features(obs, info["predicted"], c.plan_dt, c.robot_radius, c.max_angular_speed)
        return x, info["valid"], info["choice"]

    def command(self, choice: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        command = self.controller.candidates(valid.device)[choice]
        return torch.where(valid.any(1)[:, None], command, torch.zeros_like(command))

    def act(self, obs: Observation, params: torch.Tensor) -> torch.Tensor:
        x, valid, _ = self.observe(obs)
        with torch.no_grad():
            logits, _ = self.net(x, valid)
        info = self.controller.last_info
        self.last_info = {"emergency_stop": ~valid.any(1), "plan": info["plan"], "plan_steps": info["plan_steps"],
                          "predicted": info["predicted"]}
        return self.command(logits.argmax(1), valid)

    def save(self, directory: Path, stem: str = "best") -> Path:
        torch.save({"state_dict": self.net.state_dict(), "arcs": self.net.arcs, "hidden": self.net.hidden},
                   directory / f"{stem}.pt")
        path = directory / f"{stem}_policy.yaml"
        path.write_text(yaml.safe_dump({"policy": "ppo", "checkpoint": f"{stem}.pt",
                                        "controller_parameters": {k: float(v) for k, v in self.controller_values.items()}},
                                       sort_keys=False))
        return path

    @classmethod
    def load(cls, path, controller, device="cpu") -> "PPOPolicy":
        path = Path(path)
        spec = yaml.safe_load(path.read_text())
        payload = torch.load(path.parent / spec["checkpoint"], map_location=device)
        net = ActorCritic(payload["arcs"], payload["hidden"]).to(device)
        net.load_state_dict(payload["state_dict"])
        return cls(controller, spec["controller_parameters"], net.eval())
