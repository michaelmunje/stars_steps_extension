"""Pedestrian model interface and the reference-trajectory bookkeeping every model shares."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from seanjax.core.state import PedestrianState, RobotState, ScenarioBatch


class PedestrianModel(ABC):
    """Advances all pedestrians of a batch by one step.

    ``robot_present [B]`` switches the robot on or off per environment, which is how the
    robot-absent counterfactual is simulated with otherwise identical inputs. The robot never
    yields to pedestrians inside these models: it is a non-reciprocal moving obstacle.
    """

    name = "base"

    def reset(self, batch: ScenarioBatch) -> PedestrianState:
        self.first, self.last = reference_span(batch)
        active = batch.ped_valid[:, :, 0] & batch.ped_mask
        pos = batch.ped_ref[:, :, 0].clone()
        vel = reference_velocity(batch, 0)
        return PedestrianState(pos, torch.where(active[..., None], vel, 0.0), active, torch.zeros_like(active))

    @abstractmethod
    def step(self, state: PedestrianState, batch: ScenarioBatch, k: int, robot: RobotState, robot_radius: float,
             robot_present: torch.Tensor) -> PedestrianState:
        """State at time index ``k`` from the state at ``k - 1``."""


def reference_span(batch: ScenarioBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """First and last valid reference index per pedestrian (``K`` / ``-1`` when never present)."""
    K = batch.steps
    index = torch.arange(K, device=batch.ped_ref.device)
    valid = batch.ped_valid & batch.ped_mask[..., None]
    first = torch.where(valid, index, K).min(dim=-1).values
    last = torch.where(valid, index, -1).max(dim=-1).values
    return first, last


def reference_velocity(batch: ScenarioBatch, k: int) -> torch.Tensor:
    """Finite-difference reference velocity at index ``k`` (forward, backward at the end)."""
    K = batch.steps
    a, b = (k, k + 1) if k + 1 < K else (k - 1, k)
    return (batch.ped_ref[:, :, b] - batch.ped_ref[:, :, a]) / batch.dt


def clip_norm(vectors: torch.Tensor, limit) -> torch.Tensor:
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    limit = torch.as_tensor(limit, dtype=vectors.dtype, device=vectors.device)
    if limit.dim():
        limit = limit[..., None]
    return vectors * torch.clamp(limit / norm.clamp(min=1e-9), max=1.0)
