"""Batched simulation state shared by every backend, pedestrian model and policy.

Everything is a torch tensor with a leading batch dimension ``B`` (one independent environment
per row: a scenario, or a (policy candidate, scenario) pair during optimisation). Pedestrians are
padded to ``P`` per environment with ``ped_mask``; obstacles likewise. Units are metres, seconds
and radians in the scenario's world frame.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch


def _map(obj, fn):
    return replace(obj, **{f.name: fn(getattr(obj, f.name)) for f in fields(obj)
                           if torch.is_tensor(getattr(obj, f.name))})


@dataclass
class ScenarioBatch:
    """``B`` scenarios on one time grid (``dt``, ``K`` steps)."""

    robot_start: torch.Tensor   # [B, 3] x, y, heading
    robot_goal: torch.Tensor    # [B, 2]
    duration: torch.Tensor      # [B] time limit (s)
    ped_ref: torch.Tensor       # [B, P, K, 2] reference (recorded or scripted) positions
    ped_valid: torch.Tensor     # [B, P, K] bool: the reference pedestrian is present
    ped_radius: torch.Tensor    # [B, P]
    ped_speed: torch.Tensor     # [B, P] preferred walking speed
    ped_mask: torch.Tensor      # [B, P] bool: real (not padding) pedestrian
    circles: torch.Tensor       # [B, C, 3] x, y, r
    circle_mask: torch.Tensor   # [B, C]
    segments: torch.Tensor      # [B, S, 4] x0, y0, x1, y1 (walls, polygon edges)
    segment_mask: torch.Tensor  # [B, S]
    ped_walls: torch.Tensor     # [B] bool: walls also block pedestrians (False: robot arena only)
    dt: float
    names: list[str]

    @property
    def size(self) -> int:
        return int(self.robot_start.shape[0])

    @property
    def steps(self) -> int:
        return int(self.ped_ref.shape[2])

    def to(self, device) -> "ScenarioBatch":
        return _map(self, lambda t: t.to(device))

    def repeat(self, times: int) -> "ScenarioBatch":
        """Every scenario ``times`` times, candidate-major: row ``c * B + b`` is scenario ``b``."""
        out = _map(self, lambda t: t.repeat(times, *([1] * (t.dim() - 1))))
        return replace(out, names=self.names * times)


@dataclass
class PedestrianState:
    pos: torch.Tensor     # [B, P, 2]
    vel: torch.Tensor     # [B, P, 2]
    active: torch.Tensor  # [B, P] bool: in the scene this step
    done: torch.Tensor    # [B, P] bool: has left the scene for good


@dataclass
class RobotState:
    pose: torch.Tensor    # [B, 3] x, y, heading
    v: torch.Tensor       # [B] linear velocity (m/s)
    w: torch.Tensor       # [B] angular velocity (rad/s)

    @property
    def pos(self) -> torch.Tensor:
        return self.pose[:, :2]

    @property
    def velocity(self) -> torch.Tensor:
        return torch.stack([self.v * torch.cos(self.pose[:, 2]), self.v * torch.sin(self.pose[:, 2])], dim=-1)


@dataclass
class Observation:
    """What a navigation policy sees; backend independent (also built from ROS 2 messages)."""

    robot_pose: torch.Tensor    # [B, 3]
    robot_v: torch.Tensor       # [B]
    robot_w: torch.Tensor       # [B]
    goal: torch.Tensor          # [B, 2]
    ped_pos: torch.Tensor       # [B, P, 2]
    ped_vel: torch.Tensor       # [B, P, 2]
    ped_visible: torch.Tensor   # [B, P] bool
    ped_radius: torch.Tensor    # [B, P]
    circles: torch.Tensor       # [B, C, 3]
    circle_mask: torch.Tensor   # [B, C]
    segments: torch.Tensor      # [B, S, 4]
    segment_mask: torch.Tensor  # [B, S]
    last_command: torch.Tensor  # [B, 2] previous (v, w) command
    time: float = 0.0
    ped_history: torch.Tensor | None = None  # [B, P, H, 2] past positions (oldest first), for learned predictors
    history_valid: torch.Tensor | None = None  # [B, P, H] bool
