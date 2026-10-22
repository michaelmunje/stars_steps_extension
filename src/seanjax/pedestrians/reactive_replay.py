"""Model D: reactive trajectory replay.

Recorded (or scripted) trajectories give each pedestrian's *preferred* motion; social forces
(``social_force.social_forces``) let them yield to the robot, each other and walls; a tracking
term pulls them back onto the reference afterwards:

    v_des = v_ref(t) + tracking_gain * (x_ref(t) - x)       (clipped to the speed limit)

``tracking_gain`` (1/s) is the reference-tracking strength: 0 keeps only the reference velocity
(the pedestrian drifts after a detour), large values snap back quickly; relaxation time ``tau``
and the acceleration limit keep the return smooth. After its reference ends a pedestrian walks
to the reference's final point and leaves when it arrives (or ``max_overrun`` seconds later),
so robot-induced delays are measurable. With the robot absent and no other pedestrians close
by, a pedestrian reproduces its reference (up to the force model's own interactions).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from seanjax.core.state import PedestrianState
from seanjax.pedestrians.base import PedestrianModel, clip_norm, reference_velocity
from seanjax.pedestrians.social_force import SocialForceParams, integrate, resolve_robot_contact, social_forces


@dataclass
class ReactiveReplayParams:
    tracking_gain: float = 1.0
    max_overrun: float = 10.0
    forces: SocialForceParams = field(default_factory=SocialForceParams)


class ReactiveReplayModel(PedestrianModel):
    name = "reactive_replay"

    def __init__(self, params: ReactiveReplayParams | None = None) -> None:
        self.params = params or ReactiveReplayParams()

    def step(self, state, batch, k, robot, robot_radius, robot_present) -> PedestrianState:
        p, f = self.params, self.params.forces
        K = batch.steps
        kk = min(k, K - 1)
        spawn = (self.first == k) & batch.ped_mask & ~state.active & ~state.done
        tracking = state.active & (k <= self.last)
        # Feed-forward from the step's start: the reference velocity over [k-1, k] plus a correction
        # toward where the reference was at k-1 (so an undisturbed walker reproduces it exactly).
        previous = max(0, kk - 1)
        reference = batch.ped_ref[:, :, kk]
        ref_vel = reference_velocity(batch, previous)
        correction = batch.ped_ref[:, :, previous] - state.pos
        final = torch.gather(batch.ped_ref, 2, self.last.clamp(min=0)[:, :, None, None].expand(-1, -1, 1, 2))[:, :, 0]
        limit = f.max_speed_factor * torch.maximum(batch.ped_speed, torch.linalg.norm(ref_vel, dim=-1))
        desired_track = clip_norm(ref_vel + p.tracking_gain * correction, limit)
        to_final = final - state.pos
        distance = torch.linalg.norm(to_final, dim=-1, keepdim=True)
        desired_finish = to_final / distance.clamp(min=1e-9) * torch.minimum(batch.ped_speed[..., None], distance / batch.dt)
        desired = torch.where(tracking[..., None], desired_track, desired_finish)
        acceleration = social_forces(state, desired, batch, robot, robot_radius, robot_present, f)
        pos, vel = integrate(state, acceleration, limit, batch.dt)
        pos, vel = resolve_robot_contact(pos, vel, batch, robot, robot_radius, robot_present)
        overrun = k - self.last
        finished = state.active & (overrun > 0) & (
            (torch.linalg.norm(final - pos, dim=-1) < f.arrival_distance) | (overrun * batch.dt > p.max_overrun))
        pos = torch.where(spawn[..., None], reference, pos)
        vel = torch.where(spawn[..., None], ref_vel, vel)
        active = (state.active & ~finished) | spawn
        return PedestrianState(pos, torch.where(active[..., None], vel, 0.0), active, state.done | finished)
