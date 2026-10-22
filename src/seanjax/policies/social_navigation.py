"""Socially aware local trajectory-scoring controller (batched, backend independent).

At every control step, for every environment:

1. Candidate commands: a grid of ``num_speeds x num_turn_rates`` constant ``(v, w)`` pairs.
2. Each candidate is rolled out over ``prediction_horizon`` seconds (``plan_dt`` steps) with the
   robot's acceleration limits; pedestrians are forecast by a ``PedestrianPredictor`` (constant
   velocity by default; the STARS forecaster can be plugged in).
3. Costs: goal progress, soft predicted collisions, pedestrian proximity (personal space),
   disruption (entering the space in front of moving pedestrians), static-obstacle clearance,
   speed tracking, turning and command smoothness, each weighted by a tunable parameter.
4. **Hard safety constraint**: candidates whose predicted clearance to a pedestrian falls below
   ``safety_margin`` within ``safety_horizon`` seconds, or that touch a static obstacle within
   the horizon, are discarded. If none survive, the robot performs an emergency stop.
5. The best command is applied and the controller replans at the next step.

``GoalSeekingPolicy`` (``goal_seeking.py``) is the same controller blind to pedestrians: it only
avoids static obstacles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from seanjax.core.geometry import obstacle_clearance, wrap_angle
from seanjax.core.state import Observation
from seanjax.policies.base import ConstantVelocityPredictor, ParameterSpec, PedestrianPredictor, Policy

PARAMETERS = (
    ParameterSpec("goal_weight", 2.0, 0.5, 50.0, log=True, description="Weight of progress toward the goal"),
    ParameterSpec("collision_weight", 5.0, 0.1, 200.0, log=True, description="Penalty for predicted collisions beyond the safety horizon"),
    ParameterSpec("pedestrian_weight", 1.0, 0.01, 200.0, log=True, description="Penalty for entering pedestrians' personal space"),
    ParameterSpec("disruption_weight", 1.0, 0.01, 200.0, log=True, description="Penalty for blocking the path ahead of moving pedestrians"),
    ParameterSpec("obstacle_weight", 1.0, 0.01, 200.0, log=True, description="Penalty for small static-obstacle clearance"),
    ParameterSpec("preferred_speed", 0.8, 0.3, 1.0, description="Preferred forward speed (m/s)"),
    ParameterSpec("personal_space_radius", 0.8, 0.3, 1.5, description="Desired clearance around pedestrians (m)"),
    ParameterSpec("prediction_horizon", 2.0, 1.0, 4.0, description="Local trajectory prediction duration (s)"),
    ParameterSpec("turn_penalty", 0.1, 0.0, 2.0, description="Penalty for turning rate"),
    ParameterSpec("smoothness_weight", 0.2, 0.0, 2.0, description="Penalty for command changes"),
)


@dataclass
class ControllerConfig:
    num_speeds: int = 6
    reverse_speed: float = 0.3     # also consider backing off at this speed (0 disables)
    num_turn_rates: int = 11
    plan_dt: float = 0.2
    max_horizon: float = 4.0
    safety_horizon: float = 1.0
    safety_margin: float = 0.05
    obstacle_clearance: float = 0.6
    speed_cost: float = 0.5
    retreat_cost: float = 2.0      # penalty on moving away from the goal (no orbiting instead of waiting)
    disruption_lookahead: float = 1.5
    max_speed: float = 1.0
    max_angular_speed: float = 1.5
    max_acceleration: float = 1.0
    max_angular_acceleration: float = 3.0
    robot_radius: float = 0.3


class SocialNavigationPolicy(Policy):
    name = "social"
    parameters = PARAMETERS
    consider_pedestrians = True

    def __init__(self, config: ControllerConfig | None = None, predictor: PedestrianPredictor | None = None) -> None:
        self.config = config or ControllerConfig()
        self.predictor = predictor or ConstantVelocityPredictor()
        self.last_info: dict[str, torch.Tensor] = {}

    def reset(self, batch_size: int) -> None:
        if hasattr(self.predictor, "reset"):
            self.predictor.reset()

    def _p(self, params: torch.Tensor, name: str) -> torch.Tensor:
        return params[:, self.parameter_names.index(name)]

    def candidates(self, device) -> torch.Tensor:
        c = self.config
        v = torch.linspace(0.0, c.max_speed, c.num_speeds, device=device)
        if c.reverse_speed > 0:
            v = torch.cat([v.new_tensor([-c.reverse_speed]), v])
        w = torch.linspace(-c.max_angular_speed, c.max_angular_speed, c.num_turn_rates, device=device)
        return torch.cartesian_prod(v, w)                                          # [M, 2]

    def rollout_robot(self, obs: Observation, commands: torch.Tensor, steps: int) -> torch.Tensor:
        """Predicted robot positions ``[B, M, H, 2]`` and headings ``[B, M, H]`` for commands ``[M, 2]``."""
        c, dt = self.config, self.config.plan_dt
        B, M = obs.robot_pose.shape[0], commands.shape[0]
        x = obs.robot_pose[:, None, 0].expand(B, M).clone()
        y = obs.robot_pose[:, None, 1].expand(B, M).clone()
        theta = obs.robot_pose[:, None, 2].expand(B, M).clone()
        v, w = obs.robot_v[:, None].expand(B, M).clone(), obs.robot_w[:, None].expand(B, M).clone()
        positions, headings = [], []
        for _ in range(steps):
            v = v + (commands[None, :, 0] - v).clamp(-c.max_acceleration * dt, c.max_acceleration * dt)
            w = w + (commands[None, :, 1] - w).clamp(-c.max_angular_acceleration * dt, c.max_angular_acceleration * dt)
            mid = theta + 0.5 * w * dt
            x, y, theta = x + v * torch.cos(mid) * dt, y + v * torch.sin(mid) * dt, theta + w * dt
            positions.append(torch.stack([x, y], -1))
            headings.append(theta)
        return torch.stack(positions, 2), torch.stack(headings, 2)

    def act(self, obs: Observation, params: torch.Tensor) -> torch.Tensor:
        c = self.config
        device = obs.robot_pose.device
        commands = self.candidates(device)
        M = commands.shape[0]
        H = int(math.ceil(c.max_horizon / c.plan_dt))
        horizon = self._p(params, "prediction_horizon")
        t = torch.arange(1, H + 1, device=device) * c.plan_dt                      # [H]
        in_horizon = (t[None] <= horizon[:, None] + 1e-6).float()                 # [B, H]
        steps_used = in_horizon.sum(-1).clamp(min=1)
        robot_xy, robot_theta = self.rollout_robot(obs, commands, H)               # [B, M, H, 2]
        B = robot_xy.shape[0]
        goal = obs.goal[:, None, None]
        distance0 = torch.linalg.norm(obs.goal - obs.robot_pose[:, :2], dim=-1)    # [B]
        distance = torch.linalg.norm(robot_xy - goal, dim=-1)                      # [B, M, H]
        last = (steps_used.long() - 1)[:, None, None].expand(B, M, 1)
        final_distance = torch.gather(distance, 2, last)[..., 0]
        progress = (distance0[:, None] - final_distance) / (c.max_speed * horizon[:, None])
        final_theta = torch.gather(robot_theta, 2, last)[..., 0]
        final_xy = torch.gather(robot_xy, 2, last[..., None].expand(B, M, 1, 2))[:, :, 0]
        bearing = torch.atan2(obs.goal[:, None, 1] - final_xy[..., 1], obs.goal[:, None, 0] - final_xy[..., 0])
        alignment = torch.cos(wrap_angle(bearing - final_theta))
        # Arriving: no reward for overshooting; progress saturates at the goal.
        cost = -self._p(params, "goal_weight")[:, None] * (progress + 0.1 * alignment)
        # Retreating from the goal is only worth it to avoid danger: penalise it outright, so the
        # controller waits rather than circling when pedestrians block the way.
        cost = cost + c.retreat_cost * torch.relu(-progress)
        crowding = torch.zeros(B, device=device)
        valid = torch.ones(B, M, dtype=torch.bool, device=device)

        # Static obstacles (hard within the whole horizon, soft clearance cost).
        clearance_o, _ = obstacle_clearance(robot_xy.reshape(B, M * H, 2), obs.circles, obs.circle_mask,
                                            obs.segments, obs.segment_mask)
        clearance_o = clearance_o.view(B, M, H) - c.robot_radius
        valid &= ~((clearance_o < c.safety_margin) & (in_horizon[:, None] > 0)).any(-1)
        obstacle = (torch.relu(1.0 - clearance_o / c.obstacle_clearance) ** 2 * in_horizon[:, None]).sum(-1) / steps_used[:, None]
        cost = cost + self._p(params, "obstacle_weight")[:, None] * obstacle

        if self.consider_pedestrians and obs.ped_pos.shape[1]:
            future = self.predictor.predict(obs, H, c.plan_dt)                     # [B, P, H, 2]
            visible = obs.ped_visible[:, None, None, :]                            # [B, 1, 1, P]
            rel = robot_xy[:, :, :, None] - future.permute(0, 2, 1, 3)[:, None]    # [B, M, H, P, 2]
            gap = torch.linalg.norm(rel, dim=-1) - obs.ped_radius[:, None, None] - c.robot_radius
            gap = torch.where(visible, gap, torch.full_like(gap, 1e3))
            safety = (t <= c.safety_horizon + 1e-6)[None, None, :, None]
            valid &= ~((gap < c.safety_margin) & safety).any(dim=(-1, -2))
            # Current overlap check too (a pedestrian already too close: only moving away stays valid).
            weight = in_horizon[:, None, :, None]
            collision = ((gap < 0).float() * weight).amax(-1).sum(-1) / steps_used[:, None]
            personal = self._p(params, "personal_space_radius")[:, None, None, None]
            proximity = (torch.relu(1.0 - gap / personal) ** 2 * weight).sum(dim=(-1, -2)) / steps_used[:, None]
            speed = torch.linalg.norm(obs.ped_vel, dim=-1)                          # [B, P]
            heading = obs.ped_vel / speed.clamp(min=1e-6)[..., None]
            ahead = (rel * heading[:, None, None]).sum(-1) / torch.linalg.norm(rel, dim=-1).clamp(min=1e-6)
            zone = personal + speed[:, None, None] * c.disruption_lookahead
            moving = (speed > 0.2).float()[:, None, None]
            disruption = (0.5 * (1 + ahead) * torch.relu(1.0 - gap / zone) * moving * (speed[:, None, None] / 1.3)
                          * weight * visible).sum(dim=(-1, -2)) / steps_used[:, None]
            # How close the nearest visible pedestrian is now, relative to personal space (0 = far).
            now_gap = torch.where(obs.ped_visible, torch.linalg.norm(obs.ped_pos - obs.robot_pose[:, None, :2], dim=-1)
                                  - obs.ped_radius - c.robot_radius, torch.full_like(obs.ped_radius, 1e3))
            crowding = torch.relu(1.0 - now_gap.min(dim=-1).values / (2.0 * personal[:, 0, 0, 0])).clamp(max=1.0)
            cost = (cost + self._p(params, "collision_weight")[:, None] * collision
                    + self._p(params, "pedestrian_weight")[:, None] * proximity
                    + self._p(params, "disruption_weight")[:, None] * disruption)

        v, w = commands[None, :, 0], commands[None, :, 1]
        # Speed tracking fades out when a pedestrian is close: slowing down / waiting is then fine.
        cost = cost + c.speed_cost * (1.0 - crowding)[:, None] * ((v - self._p(params, "preferred_speed")[:, None]) / c.max_speed) ** 2
        cost = cost + self._p(params, "turn_penalty")[:, None] * w.abs() / c.max_angular_speed
        change = ((v - obs.last_command[:, None, 0]).abs() / c.max_speed
                  + (w - obs.last_command[:, None, 1]).abs() / c.max_angular_speed)
        cost = cost + self._p(params, "smoothness_weight")[:, None] * change
        # Slow down near the goal instead of orbiting it.
        near = (distance0 < 1.0)[:, None]
        cost = cost + near * (v - distance0[:, None].clamp(max=1.0) * c.max_speed).abs()

        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        best = cost.argmin(dim=1)
        command = commands[best]
        emergency = ~valid.any(dim=1)
        command = torch.where(emergency[:, None], torch.zeros_like(command), command)
        self.last_info = {"emergency_stop": emergency, "valid_fraction": valid.float().mean(1)}
        return command
