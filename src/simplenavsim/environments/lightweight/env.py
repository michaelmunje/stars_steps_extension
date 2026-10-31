"""Lightweight 2-D backend: batched differential-drive robot among pedestrians and static obstacles.

The robot is a disc with unicycle (differential-drive) kinematics and acceleration limits:
commands ``(v, w)`` are tracked within ``max_acceleration`` / ``max_angular_acceleration`` per
second, then the pose is integrated exactly along the arc. Pedestrians are discs advanced by a
``PedestrianModel``. A robot-pedestrian or robot-obstacle overlap is a collision and ends the
robot's episode; reaching within ``goal_tolerance`` of the goal is success; ``duration`` is the
time limit. Once a robot is done it stops; a robot that reached its goal leaves the scene
(pedestrians no longer react to it), while one that collided or timed out stays as a stopped
obstacle. Pedestrians keep moving to the end of the horizon either way, so pedestrian outcomes
(and disruption) always cover the same time span.

Everything is a torch tensor with one row per environment, on CPU or GPU, with no other
dependencies; ``Observation`` is what every policy consumes, independent of the backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from simplenavsim.core.geometry import obstacle_clearance, wrap_angle
from simplenavsim.core.state import Observation, PedestrianState, RobotState, ScenarioBatch
from simplenavsim.pedestrians.base import PedestrianModel


@dataclass
class RobotConfig:
    """Defaults approximate a Clearpath Jackal-class differential-drive base."""

    radius: float = 0.3
    max_speed: float = 1.0
    min_speed: float = -0.3          # slow reverse (backing off in narrow spaces)
    max_angular_speed: float = 1.5
    max_acceleration: float = 1.0
    max_angular_acceleration: float = 3.0


@dataclass
class SimConfig:
    goal_tolerance: float = 1.0   # m; SEAN 2.0 used 1.2 m. At 0.3 m the robot orbited the goal
    sensing_radius: float = 8.0
    history_steps: int = 8       # past pedestrian positions in observations (for learned predictors)
    history_stride: int = 4      # in simulation steps (0.4 s at dt = 0.1, the STEPS rate)


@dataclass
class EnvState:
    k: int
    robot: RobotState
    peds: PedestrianState
    command: torch.Tensor            # [B, 2] last applied command
    robot_present: torch.Tensor      # [B] pedestrians react to the robot
    done: torch.Tensor               # [B]
    success: torch.Tensor
    ped_collision: torch.Tensor
    obstacle_collision: torch.Tensor
    end_step: torch.Tensor           # [B] step at which the episode ended (K - 1 if still running)
    history: list                    # past (positions [B, P, 2], active [B, P]), newest last


class LightweightEnv:
    name = "lightweight"

    def __init__(self, pedestrians: PedestrianModel, robot: RobotConfig | None = None, sim: SimConfig | None = None):
        self.pedestrians, self.robot, self.sim = pedestrians, robot or RobotConfig(), sim or SimConfig()

    def reset(self, batch: ScenarioBatch, *, robot_present: bool = True) -> EnvState:
        B, device = batch.size, batch.robot_start.device
        peds = self.pedestrians.reset(batch)
        robot = RobotState(batch.robot_start.clone(), torch.zeros(B, device=device), torch.zeros(B, device=device))
        false = torch.zeros(B, dtype=torch.bool, device=device)
        present = torch.full((B,), bool(robot_present), dtype=torch.bool, device=device)
        state = EnvState(0, robot, peds, torch.zeros(B, 2, device=device), present, false.clone(), false.clone(),
                         false.clone(), false.clone(), torch.full((B,), batch.steps - 1, device=device),
                         [(peds.pos.clone(), peds.active.clone())])
        if robot_present:
            self._check(state, batch)
        return state

    def step(self, state: EnvState, batch: ScenarioBatch, command: torch.Tensor | None) -> EnvState:
        """Apply ``command [B, 2]`` (ignored for finished or absent robots) and advance one step."""
        r, dt = self.robot, batch.dt
        robot = state.robot
        active = state.robot_present & ~state.done
        visible = state.robot_present & ~state.success  # stopped robots remain obstacles; arrived ones leave
        if command is None:
            command = torch.zeros_like(state.command)
        v_cmd = command[:, 0].clamp(r.min_speed, r.max_speed)
        w_cmd = command[:, 1].clamp(-r.max_angular_speed, r.max_angular_speed)
        v = robot.v + (v_cmd - robot.v).clamp(-r.max_acceleration * dt, r.max_acceleration * dt)
        w = robot.w + (w_cmd - robot.w).clamp(-r.max_angular_acceleration * dt, r.max_angular_acceleration * dt)
        v, w = torch.where(active, v, 0.0), torch.where(active, w, 0.0)
        pose = integrate_unicycle(robot.pose, v, w, dt)
        new_robot = RobotState(torch.where(active[:, None], pose, robot.pose), v, w)
        k = state.k + 1
        # Pedestrians react to the robot's state at the start of the step (no look-ahead).
        peds = self.pedestrians.step(state.peds, batch, k, robot, r.radius, visible)
        keep = max(1, (self.sim.history_steps - 1) * self.sim.history_stride + 1)
        history = (state.history + [(peds.pos.clone(), peds.active.clone())])[-keep:]
        new = EnvState(k, new_robot, peds, torch.stack([v_cmd, w_cmd], -1), state.robot_present, state.done.clone(),
                       state.success.clone(), state.ped_collision.clone(), state.obstacle_collision.clone(),
                       state.end_step.clone(), history)
        if bool(state.robot_present.any()):
            self._check(new, batch)
        timeout = (k * dt >= batch.duration - 1e-6) | (k >= batch.steps - 1)
        newly = timeout & ~new.done
        new.end_step = torch.where(newly, torch.full_like(new.end_step, k), new.end_step)
        new.done = new.done | timeout
        return new

    def _check(self, state: EnvState, batch: ScenarioBatch) -> None:
        r = self.robot
        live = state.robot_present & ~state.done
        pos = state.robot.pos
        gap = torch.linalg.norm(state.peds.pos - pos[:, None], dim=-1) - batch.ped_radius - r.radius
        ped_hit = ((gap < 0) & state.peds.active).any(dim=-1) & live
        clearance, _ = obstacle_clearance(pos[:, None], batch.circles, batch.circle_mask, batch.segments,
                                          batch.segment_mask)
        obstacle_hit = (clearance[:, 0] < r.radius) & live
        reached = (torch.linalg.norm(batch.robot_goal - pos, dim=-1) < self.sim.goal_tolerance) & live & ~ped_hit & ~obstacle_hit
        ended = ped_hit | obstacle_hit | reached
        state.ped_collision |= ped_hit
        state.obstacle_collision |= obstacle_hit
        state.success |= reached
        state.end_step = torch.where(ended, torch.full_like(state.end_step, state.k), state.end_step)
        state.done = state.done | ended

    def observe(self, state: EnvState, batch: ScenarioBatch) -> Observation:
        robot, peds = state.robot, state.peds
        visible = peds.active & (torch.linalg.norm(peds.pos - robot.pos[:, None], dim=-1) < self.sim.sensing_radius)
        history = valid = None
        if self.sim.history_steps:
            stride, n = self.sim.history_stride, self.sim.history_steps
            index = [len(state.history) - 1 - stride * (n - 1 - i) for i in range(n)]
            frames = [state.history[max(0, j)] for j in index]
            history = torch.stack([f[0] for f in frames], dim=2)                     # [B, P, n, 2]
            valid = torch.stack([f[1] & (j >= 0) for f, j in zip(frames, index)], dim=2)
            valid &= visible[..., None]
        return Observation(robot.pose, robot.v, robot.w, batch.robot_goal, peds.pos, peds.vel, visible,
                           batch.ped_radius, batch.circles, batch.circle_mask, batch.segments, batch.segment_mask,
                           state.command, time=state.k * batch.dt, ped_history=history, history_valid=valid)


def integrate_unicycle(pose: torch.Tensor, v: torch.Tensor, w: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact constant-(v, w) arc over ``dt`` (straight line when w ~ 0)."""
    x, y, theta = pose.unbind(-1)
    small = w.abs() < 1e-6
    w_safe = torch.where(small, torch.ones_like(w), w)
    new_theta = theta + w * dt
    dx = torch.where(small, v * torch.cos(theta) * dt, v / w_safe * (torch.sin(new_theta) - torch.sin(theta)))
    dy = torch.where(small, v * torch.sin(theta) * dt, -v / w_safe * (torch.cos(new_theta) - torch.cos(theta)))
    return torch.stack([x + dx, y + dy, wrap_angle(new_theta)], dim=-1)
