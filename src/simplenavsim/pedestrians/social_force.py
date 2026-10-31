"""Model B: Social Force Model (Helbing & Molnar 1995), batched over environments and pedestrians.

Each pedestrian relaxes toward a desired velocity (time constant ``tau``) and is pushed by
exponential repulsions from other pedestrians, the robot and static obstacles:

    a_i = (v_des_i - v_i) / tau
          + sum_j A exp((r_i + r_j - d_ij) / B) n_ij w(phi_ij)       (pedestrians, anisotropic)
          + A_robot exp((r_i + r_robot - d_iR) / B_robot) n_iR w(phi_iR)
          + A_obstacle exp((r_i - c_i) / B_obstacle) n_iO             (c_i: clearance to the nearest wall)

with ``w(phi) = lambda + (1 - lambda)(1 + cos phi) / 2`` weighting interactions in front of the
walker. As in SEAN 2.0's agents, repulsions from pedestrians and the robot also get a sideways
component (``sidestep`` times the repulsion, to the walker's right) so head-on encounters resolve by
passing instead of stalling face to face. Accelerations and speeds are bounded (``max_acceleration``, ``max_speed_factor`` times the
preferred speed), so pedestrians never jump.

``SocialForceModel`` walks each pedestrian to the end of its reference trajectory (goal mode);
``reactive_replay.ReactiveReplayModel`` reuses ``social_forces`` with reference tracking instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from simplenavsim.core.geometry import obstacle_clearance
from simplenavsim.core.state import PedestrianState
from simplenavsim.pedestrians.base import PedestrianModel, clip_norm


@dataclass
class SocialForceParams:
    tau: float = 0.5
    ped_strength: float = 2.1        # A (m/s^2)
    ped_range: float = 0.3           # B (m)
    robot_strength: float = 5.0
    robot_range: float = 0.5
    obstacle_strength: float = 10.0
    obstacle_range: float = 0.2
    anisotropy: float = 0.35         # lambda
    sidestep: float = 0.5            # sideways share of pedestrian/robot repulsion (to the walker's right)
    interaction_radius: float = 5.0  # ignore pedestrians farther than this
    max_acceleration: float = 3.0
    max_speed_factor: float = 1.3
    arrival_distance: float = 0.3


def _anisotropy(direction: torch.Tensor, away: torch.Tensor, lam: float) -> torch.Tensor:
    """``w(phi)`` with phi the angle between a walker's heading and the direction to the source."""
    heading = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-9)
    cos_phi = -(heading * away).sum(-1)
    return lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi)


def social_forces(state: PedestrianState, desired: torch.Tensor, batch, robot, robot_radius: float,
                  robot_present: torch.Tensor, p: SocialForceParams) -> torch.Tensor:
    """Total acceleration ``[B, P, 2]`` for every pedestrian (zero for inactive ones)."""
    pos, vel, active = state.pos, state.vel, state.active
    heading = torch.where(torch.linalg.norm(desired, dim=-1, keepdim=True) > 1e-6, desired, vel)
    force = (desired - vel) / p.tau

    rel = pos[:, :, None] - pos[:, None]                                    # [B, P, P, 2] i <- j
    dist = torch.linalg.norm(rel, dim=-1)
    away = rel / dist.clamp(min=1e-9)[..., None]
    reach = batch.ped_radius[:, :, None] + batch.ped_radius[:, None]
    P = pos.shape[1]
    pair = active[:, :, None] & active[:, None] & ~torch.eye(P, dtype=torch.bool, device=pos.device)
    pair &= dist < p.interaction_radius
    magnitude = p.ped_strength * torch.exp((reach - dist) / p.ped_range)
    magnitude = magnitude * _anisotropy(heading[:, :, None], away, p.anisotropy) * pair
    unit = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp(min=1e-9)
    # Side-stepping is something walkers do; standing pedestrians would otherwise orbit in place.
    walking = (torch.linalg.norm(desired, dim=-1, keepdim=True) / 0.5).clamp(max=1.0)
    right = torch.stack([unit[..., 1], -unit[..., 0]], dim=-1) * walking         # walker's right-hand side
    force = force + (magnitude[..., None] * away).sum(dim=2) + p.sidestep * magnitude.sum(dim=2)[..., None] * right

    rel_r = pos - robot.pos[:, None]
    dist_r = torch.linalg.norm(rel_r, dim=-1)
    away_r = rel_r / dist_r.clamp(min=1e-9)[..., None]
    magnitude_r = p.robot_strength * torch.exp((batch.ped_radius + robot_radius - dist_r) / p.robot_range)
    magnitude_r = magnitude_r * _anisotropy(heading, away_r, p.anisotropy) * robot_present[:, None]
    force = force + magnitude_r[..., None] * (away_r + p.sidestep * right)

    clearance, away_o = obstacle_clearance(pos, batch.circles, batch.circle_mask, batch.segments, batch.segment_mask)
    magnitude_o = p.obstacle_strength * torch.exp((batch.ped_radius - clearance) / p.obstacle_range)
    force = force + magnitude_o[..., None] * away_o
    return torch.where(active[..., None], clip_norm(force, p.max_acceleration), 0.0)


def integrate(state: PedestrianState, acceleration: torch.Tensor, max_speed: torch.Tensor, dt: float):
    vel = clip_norm(state.vel + acceleration * dt, max_speed)
    vel = torch.where(state.active[..., None], vel, 0.0)
    return state.pos + vel * dt, vel


def resolve_robot_contact(pos, vel, batch, robot, robot_radius: float, robot_present: torch.Tensor):
    """Reactive pedestrians cannot walk through the robot: move any overlap back to touching distance
    and drop the velocity component into the robot (a collision then comes from the robot's motion)."""
    rel = pos - robot.pos[:, None]
    dist = torch.linalg.norm(rel, dim=-1)
    reach = batch.ped_radius + robot_radius
    inside = (dist < reach) & robot_present[:, None]
    away = rel / dist.clamp(min=1e-9)[..., None]
    pos = torch.where(inside[..., None], robot.pos[:, None] + away * reach[..., None], pos)
    inward = (vel * away).sum(-1).clamp(max=0.0)
    vel = torch.where(inside[..., None], vel - inward[..., None] * away, vel)
    return pos, vel


class SocialForceModel(PedestrianModel):
    """Pedestrians appear where their reference starts and walk to where it ends."""

    name = "social_force"

    def __init__(self, params: SocialForceParams | None = None) -> None:
        self.params = params or SocialForceParams()

    def step(self, state, batch, k, robot, robot_radius, robot_present) -> PedestrianState:
        p = self.params
        spawn = (self.first == k) & batch.ped_mask & ~state.active & ~state.done
        goal = torch.gather(batch.ped_ref, 2, self.last.clamp(min=0)[:, :, None, None].expand(-1, -1, 1, 2))[:, :, 0]
        to_goal = goal - state.pos
        distance = torch.linalg.norm(to_goal, dim=-1, keepdim=True)
        desired = to_goal / distance.clamp(min=1e-9) * torch.minimum(batch.ped_speed[..., None], distance / batch.dt)
        acceleration = social_forces(state, desired, batch, robot, robot_radius, robot_present, p)
        pos, vel = integrate(state, acceleration, batch.ped_speed * p.max_speed_factor, batch.dt)
        pos, vel = resolve_robot_contact(pos, vel, batch, robot, robot_radius, robot_present)
        start = torch.gather(batch.ped_ref, 2, self.first.clamp(max=batch.steps - 1)[:, :, None, None]
                             .expand(-1, -1, 1, 2))[:, :, 0]
        stationary = torch.linalg.norm(goal - start, dim=-1) < p.arrival_distance  # e.g. a standing group
        arrived = state.active & (torch.linalg.norm(goal - pos, dim=-1) < p.arrival_distance) & ~stationary
        pos = torch.where(spawn[..., None], batch.ped_ref[:, :, k], pos)
        vel = torch.where(spawn[..., None], 0.0, vel)
        active = (state.active & ~arrived) | spawn
        return PedestrianState(pos, torch.where(active[..., None], vel, 0.0), active, state.done | arrived)
