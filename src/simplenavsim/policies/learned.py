"""A learned navigation policy: a small MLP from a fixed egocentric observation to ``(v, w)``.

Observation (all in the robot frame, metres and m/s, 64 numbers):

* goal: distance (clipped at 10 m, /10) and bearing (cos, sin);
* the robot's ``v``, ``w`` and its last command;
* the ``K`` (6) nearest visible pedestrians, nearest first: relative position, relative velocity,
  and the predictor's forecast of their relative position 1 s and 2 s ahead, plus a presence flag;
* the nearest static obstacle: clearance (clipped at 3 m, /3) and the unit direction away from it.

Weights are one flat vector per environment (``params [B, D]``), so a whole population of networks
runs as one simulation batch (evolution strategies, ``simplenavsim.optimization.es``). A safety shield
mirrors the social controller's hard constraint: if the commanded motion, held for
``shield_horizon``, comes within ``safety_margin`` of a predicted pedestrian or a wall, the robot
stops. Decision 91.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from simplenavsim.core.geometry import obstacle_clearance
from simplenavsim.core.state import Observation
from simplenavsim.policies.base import ConstantVelocityPredictor, PedestrianPredictor, Policy

K_PEDS = 6
OBS_DIM = 3 + 4 + K_PEDS * 9 + 3


@dataclass
class LearnedConfig:
    hidden: int = 64
    max_speed: float = 1.0
    min_speed: float = -0.3
    max_angular_speed: float = 1.5
    max_acceleration: float = 1.0
    max_angular_acceleration: float = 3.0
    robot_radius: float = 0.3
    plan_dt: float = 0.2
    shield: bool = True
    shield_horizon: float = 1.0
    safety_margin: float = 0.05
    shield_fallback: bool = False  # unsafe command -> nearest safe command on a (v, w) grid, not a stop (decision 91)
    fallback_speeds: int = 7
    fallback_turns: int = 11


def _rotate(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """World-frame vectors ``[..., 2]`` into the robot frame (``cos``/``sin`` of the robot heading)."""
    return torch.stack([cos * vec[..., 0] + sin * vec[..., 1], -sin * vec[..., 0] + cos * vec[..., 1]], -1)


class LearnedPolicy(Policy):
    name = "learned"
    parameters = ()

    def __init__(self, config: LearnedConfig | None = None, predictor: PedestrianPredictor | None = None) -> None:
        self.config = config or LearnedConfig()
        self.predictor = predictor or ConstantVelocityPredictor()
        h = self.config.hidden
        self.shapes = [(OBS_DIM, h), (h,), (h, h), (h,), (h, 2), (2,)]
        self.dim = sum(int(np.prod(s)) for s in self.shapes)
        self.weights: np.ndarray | None = None       # default weights (from a saved policy)
        self.last_info: dict[str, torch.Tensor] = {}

    def reset(self, batch_size: int) -> None:
        if hasattr(self.predictor, "reset"):
            self.predictor.reset()

    # --------------------------------------------------------------- parameters
    def default_parameters(self) -> dict[str, float]:
        return {}

    def parameter_tensor(self, values, batch: int, device="cpu") -> torch.Tensor:
        """``[batch, D]`` weights: one vector (broadcast), a list of vectors (one per row), or ``{}``
        (the loaded default weights)."""
        if isinstance(values, dict):
            if self.weights is None:
                raise ValueError("LearnedPolicy has no weights loaded")
            values = self.weights
        if isinstance(values, list):
            rows = torch.as_tensor(np.stack([np.asarray(v, np.float32) for v in values]), device=device)
            if rows.shape[0] != batch:
                raise ValueError(f"Expected {batch} weight vectors, got {rows.shape[0]}")
            return rows
        vector = torch.as_tensor(np.asarray(values, np.float32), device=device)
        return vector[None].expand(batch, -1)

    def _unflatten(self, params: torch.Tensor) -> list[torch.Tensor]:
        out, start = [], 0
        for shape in self.shapes:
            n = int(np.prod(shape))
            out.append(params[:, start:start + n].reshape(params.shape[0], *shape))
            start += n
        return out

    def forward(self, features: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Raw outputs ``[B, 2]`` in ``[-1, 1]`` for features ``[B, OBS_DIM]``."""
        w1, b1, w2, b2, w3, b3 = self._unflatten(params)
        x = torch.tanh(torch.einsum("bi,bij->bj", features, w1) + b1)
        x = torch.tanh(torch.einsum("bi,bij->bj", x, w2) + b2)
        return torch.tanh(torch.einsum("bi,bij->bj", x, w3) + b3)

    def to_command(self, raw: torch.Tensor) -> torch.Tensor:
        c = self.config
        v = c.min_speed + (raw[:, 0] + 1) * 0.5 * (c.max_speed - c.min_speed)
        return torch.stack([v, raw[:, 1] * c.max_angular_speed], -1)

    def from_command(self, command: torch.Tensor) -> torch.Tensor:
        """Inverse of ``to_command`` (behaviour-cloning targets in ``[-1, 1]``)."""
        c = self.config
        v = (command[:, 0] - c.min_speed) / (c.max_speed - c.min_speed) * 2 - 1
        return torch.stack([v, command[:, 1] / c.max_angular_speed], -1).clamp(-0.999, 0.999)

    # --------------------------------------------------------------- observation
    def features(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``[B, OBS_DIM]`` features and the pedestrian forecast ``[B, P, H, 2]`` (for the shield)."""
        c = self.config
        B, device = obs.robot_pose.shape[0], obs.robot_pose.device
        xy, th = obs.robot_pose[:, :2], obs.robot_pose[:, 2]
        cos, sin = torch.cos(th), torch.sin(th)
        to_goal = _rotate(obs.goal - xy, cos, sin)
        dist = torch.linalg.norm(to_goal, dim=-1)
        goal = torch.stack([dist.clamp(max=10.0) / 10.0, to_goal[:, 0] / dist.clamp(min=1e-6),
                            to_goal[:, 1] / dist.clamp(min=1e-6)], -1)
        own = torch.stack([obs.robot_v, obs.robot_w / c.max_angular_speed, obs.last_command[:, 0],
                           obs.last_command[:, 1] / c.max_angular_speed], -1)
        peds = torch.zeros(B, K_PEDS, 9, device=device)
        future = None
        if obs.ped_pos.shape[1]:
            steps = int(round(2.0 / c.plan_dt))
            future = self.predictor.predict(obs, steps, c.plan_dt)                 # [B, P, steps, 2]
            rel = obs.ped_pos - xy[:, None]
            gap = torch.linalg.norm(rel, dim=-1) - obs.ped_radius - c.robot_radius
            gap = torch.where(obs.ped_visible, gap, torch.full_like(gap, 1e6))
            k = min(K_PEDS, gap.shape[1])
            order = gap.topk(k, dim=1, largest=False).indices                     # [B, k]
            pick = lambda t: torch.gather(t, 1, order.view(B, k, *([1] * (t.dim() - 2))).expand(B, k, *t.shape[2:]))
            rel_k, vel_k = pick(rel), pick(obs.ped_vel)
            f1 = pick(future[:, :, int(round(1.0 / c.plan_dt)) - 1]) - xy[:, None]
            f2 = pick(future[:, :, steps - 1]) - xy[:, None]
            present = (pick(gap) < 1e5).float()
            rot = lambda v: _rotate(v, cos[:, None], sin[:, None]) / 5.0
            robot_vel = torch.stack([obs.robot_v * cos, obs.robot_v * sin], -1)[:, None]
            block = torch.cat([rot(rel_k), rot(vel_k - robot_vel) * 2.5, rot(f1), rot(f2), present[..., None]], -1)
            peds[:, :k] = block * present[..., None]
        clearance, away = obstacle_clearance(xy[:, None], obs.circles, obs.circle_mask, obs.segments, obs.segment_mask)
        obstacle = torch.cat([(clearance[:, :1] - c.robot_radius).clamp(0.0, 3.0) / 3.0,
                              _rotate(away[:, 0], cos, sin)], -1)
        features = torch.cat([goal, own, peds.reshape(B, -1), obstacle], -1)
        return torch.nan_to_num(features), future

    # --------------------------------------------------------------- acting
    def _shield(self, obs: Observation, command: torch.Tensor, future: torch.Tensor | None) -> torch.Tensor:
        """``[B]`` bool: holding ``command`` (``[B, 2]``, or ``[B, M, 2]`` for M candidates -> ``[B, M]``)
        for ``shield_horizon`` comes too close to someone or a wall."""
        c = self.config
        single = command.dim() == 2
        command = command[:, None] if single else command
        B, M = command.shape[:2]
        steps = max(1, int(round(c.shield_horizon / c.plan_dt)))
        x = obs.robot_pose[:, None, 0].expand(B, M).clone()
        y = obs.robot_pose[:, None, 1].expand(B, M).clone()
        th = obs.robot_pose[:, None, 2].expand(B, M).clone()
        v, w = obs.robot_v[:, None].expand(B, M).clone(), obs.robot_w[:, None].expand(B, M).clone()
        unsafe = torch.zeros(B, M, dtype=torch.bool, device=command.device)
        for step in range(steps):
            v = v + (command[..., 0] - v).clamp(-c.max_acceleration * c.plan_dt, c.max_acceleration * c.plan_dt)
            w = w + (command[..., 1] - w).clamp(-c.max_angular_acceleration * c.plan_dt, c.max_angular_acceleration * c.plan_dt)
            mid = th + 0.5 * w * c.plan_dt
            x, y, th = x + v * torch.cos(mid) * c.plan_dt, y + v * torch.sin(mid) * c.plan_dt, th + w * c.plan_dt
            point = torch.stack([x, y], -1)                                         # [B, M, 2]
            if future is not None:
                peds = future[:, :, min(step, future.shape[2] - 1)]                  # [B, P, 2]
                gap = torch.linalg.norm(peds[:, None] - point[:, :, None], dim=-1) - obs.ped_radius[:, None] - c.robot_radius
                unsafe |= ((gap < c.safety_margin) & obs.ped_visible[:, None]).any(-1)
            clearance, _ = obstacle_clearance(point, obs.circles, obs.circle_mask, obs.segments, obs.segment_mask)
            unsafe |= clearance - c.robot_radius < c.safety_margin
        return unsafe[:, 0] if single else unsafe

    def _fallback(self, obs: Observation, command: torch.Tensor, future: torch.Tensor | None):
        """Nearest safe command on a ``(v, w)`` grid to the unsafe ``command``; ``(0, 0)`` if none is safe."""
        c = self.config
        v = torch.linspace(c.min_speed, c.max_speed, c.fallback_speeds, device=command.device)
        w = torch.linspace(-c.max_angular_speed, c.max_angular_speed, c.fallback_turns, device=command.device)
        grid = torch.cartesian_prod(v, w)                                            # [M, 2]
        candidates = grid[None].expand(command.shape[0], -1, -1)
        unsafe = self._shield(obs, candidates, future)                               # [B, M]
        scale = torch.tensor([c.max_speed - c.min_speed, 2 * c.max_angular_speed], device=command.device)
        distance = (((candidates - command[:, None]) / scale) ** 2).sum(-1)
        distance = torch.where(unsafe, torch.full_like(distance, float("inf")), distance)
        best = distance.argmin(-1)
        chosen = grid[best]
        none_safe = unsafe.all(-1)
        return torch.where(none_safe[:, None], torch.zeros_like(chosen), chosen), none_safe

    def act(self, obs: Observation, params: torch.Tensor) -> torch.Tensor:
        features, future = self.features(obs)
        command = self.to_command(self.forward(features, params.to(features.dtype)))
        emergency = torch.zeros(command.shape[0], dtype=torch.bool, device=command.device)
        if self.config.shield:
            unsafe = self._shield(obs, command, future)
            if self.config.shield_fallback and bool(unsafe.any()):
                safe, emergency = self._fallback(obs, command, future)
                command = torch.where(unsafe[:, None], safe, command)
                emergency = emergency & unsafe
            else:
                emergency = unsafe
                command = torch.where(emergency[:, None], torch.zeros_like(command), command)
        self.last_info = {"emergency_stop": emergency, "features": features}
        return command

    # --------------------------------------------------------------- io
    def init_weights(self, seed: int = 0, scale: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        parts = []
        for shape in self.shapes:
            if len(shape) == 2:
                parts.append(rng.normal(0, scale / math.sqrt(shape[0]), shape).ravel())
            else:
                parts.append(np.zeros(shape))
        return np.concatenate(parts).astype(np.float32)
