"""Model A: pedestrians follow their reference (recorded) trajectories exactly, ignoring the robot."""
from __future__ import annotations

import torch

from seanjax.core.state import PedestrianState
from seanjax.pedestrians.base import PedestrianModel, reference_velocity


class ReplayModel(PedestrianModel):
    name = "replay"

    def step(self, state, batch, k, robot, robot_radius, robot_present) -> PedestrianState:
        active = batch.ped_valid[:, :, k] & batch.ped_mask
        vel = torch.where(active[..., None], reference_velocity(batch, k - 1), 0.0)
        done = state.done | (state.active & ~active)
        return PedestrianState(batch.ped_ref[:, :, k].clone(), vel, active, done)
