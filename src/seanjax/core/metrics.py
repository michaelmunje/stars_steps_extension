"""Social navigation metrics from recorded rollouts (per environment, ``[B]`` tensors).

Robot task: ``success``, ``collision`` (``ped_collision``, ``obstacle_collision``), ``timeout``,
``time_to_goal`` (NaN unless successful), ``normalized_time`` (episode time / limit, 1 on
failure), ``path_length``, ``path_efficiency`` (straight-line distance / path driven; 0 on failure).

Comfort and safety (SEAN 2.0 zones, measured surface to surface): ``min_ped_clearance``,
``intimate_violation_rate`` / ``personal_violation_rate`` (fraction of the robot's active steps
with a pedestrian closer than 0.45 m / 1.2 m), ``stops`` (moving -> stopped transitions),
``emergency_stops``, ``command_smoothness`` (mean |dv|/dt + |dw|/dt) and ``jerk`` (mean |d^2 v|/dt^2).

Pedestrian disruption, against the robot-absent counterfactual with identical initial conditions
and pedestrian model (only meaningful for reactive pedestrians; zero for replay):
``disruption_displacement`` (mean distance between matched pedestrian positions while both are in
the scene, averaged over pedestrians), ``disruption_max_displacement``, ``velocity_deviation``
(time-integrated |v - v_counterfactual|, mean per pedestrian), ``ped_delay`` (extra seconds in the
scene, mean per pedestrian), ``peds_affected`` (pedestrians displaced by more than
``affected_threshold``) and ``reference_deviation`` (mean distance to the recorded reference).
Minimising disruption is not the same as being socially appropriate, so these are reported
separately from comfort and task metrics.
"""
from __future__ import annotations

import torch

from seanjax.core.rollout import Rollout
from seanjax.core.state import ScenarioBatch

INTIMATE_DISTANCE = 0.45
PERSONAL_DISTANCE = 1.2


def robot_metrics(run: Rollout, batch: ScenarioBatch, robot_radius: float) -> dict[str, torch.Tensor]:
    B, K = run.robot_v.shape
    device = run.robot_v.device
    step = torch.arange(K, device=device)
    live = step[None] <= run.end_step[:, None]                                   # robot in the episode
    moving_steps = live[:, 1:].float()
    collision = run.ped_collision | run.obstacle_collision
    duration = run.end_step.float() * run.dt
    out = {
        "success": run.success.float(),
        "collision": collision.float(),
        "ped_collision": run.ped_collision.float(),
        "obstacle_collision": run.obstacle_collision.float(),
        "timeout": (~run.success & ~collision).float(),
        "time_to_goal": torch.where(run.success, duration, torch.full_like(duration, float("nan"))),
        "normalized_time": torch.where(run.success, duration / batch.duration, torch.ones_like(duration)),
    }
    steps = torch.linalg.norm(run.robot_pose[:, 1:, :2] - run.robot_pose[:, :-1, :2], dim=-1)
    out["path_length"] = (steps * moving_steps).sum(-1)
    straight = torch.linalg.norm(batch.robot_goal - batch.robot_start[:, :2], dim=-1)
    # 1 = straight to the goal; detours, loops and orbits lower it. Failed episodes count as 0.
    out["path_efficiency"] = torch.where(run.success, (straight / out["path_length"].clamp(min=1e-6)).clamp(max=1.0),
                                         torch.zeros_like(straight))

    rel = run.robot_pose[:, None, :, :2] - run.ped_pos                           # [B, P, K, 2]
    gap = torch.linalg.norm(rel, dim=-1) - batch.ped_radius[..., None] - robot_radius
    present = run.ped_active & live[:, None]
    gap = torch.where(present, gap, torch.full_like(gap, float("inf")))
    nearest = gap.min(dim=1).values                                              # [B, K]
    closest = nearest.min(dim=-1).values
    out["min_ped_clearance"] = torch.where(torch.isfinite(closest), closest, torch.full_like(closest, float("nan")))
    live_count = live.float().sum(-1).clamp(min=1)
    out["intimate_violation_rate"] = ((nearest < INTIMATE_DISTANCE) & live).float().sum(-1) / live_count
    out["personal_violation_rate"] = ((nearest < PERSONAL_DISTANCE) & live).float().sum(-1) / live_count

    v = run.robot_v
    was_moving = v[:, :-1] > 0.1
    now_stopped = v[:, 1:] < 0.05
    out["stops"] = (was_moving & now_stopped & live[:, 1:]).float().sum(-1)
    out["emergency_stops"] = (run.emergency & live).float().sum(-1)
    dv = (run.command[:, 1:, 0] - run.command[:, :-1, 0]).abs() / run.dt
    dw = (run.command[:, 1:, 1] - run.command[:, :-1, 1]).abs() / run.dt
    count = moving_steps.sum(-1).clamp(min=1)
    out["command_smoothness"] = ((dv + dw) * moving_steps).sum(-1) / count
    accel = (v[:, 1:] - v[:, :-1]) / run.dt
    jerk = (accel[:, 1:] - accel[:, :-1]).abs() / run.dt
    out["jerk"] = (jerk * live[:, 2:].float()).sum(-1) / live[:, 2:].float().sum(-1).clamp(min=1)
    return out


def disruption_metrics(run: Rollout, counterfactual: Rollout, batch: ScenarioBatch,
                       affected_threshold: float = 0.3) -> dict[str, torch.Tensor]:
    both = run.ped_active & counterfactual.ped_active                             # [B, P, K]
    displacement = torch.linalg.norm(run.ped_pos - counterfactual.ped_pos, dim=-1) * both
    seen = (run.ped_active | counterfactual.ped_active).any(-1) & batch.ped_mask   # [B, P]
    n = seen.float().sum(-1).clamp(min=1)
    per_ped_mean = displacement.sum(-1) / both.float().sum(-1).clamp(min=1)
    per_ped_max = displacement.amax(-1)
    vel = (run.ped_pos[:, :, 1:] - run.ped_pos[:, :, :-1]) / run.dt
    vel_cf = (counterfactual.ped_pos[:, :, 1:] - counterfactual.ped_pos[:, :, :-1]) / run.dt
    both_v = both[:, :, 1:] & both[:, :, :-1]
    velocity_deviation = (torch.linalg.norm(vel - vel_cf, dim=-1) * both_v).sum(-1) * run.dt
    delay = (run.ped_active.float().sum(-1) - counterfactual.ped_active.float().sum(-1)).clamp(min=0) * run.dt
    reference = torch.linalg.norm(run.ped_pos - batch.ped_ref[:, :, : run.steps], dim=-1)
    ref_valid = run.ped_active & batch.ped_valid[:, :, : run.steps]
    reference_deviation = (reference * ref_valid).sum(-1) / ref_valid.float().sum(-1).clamp(min=1)
    mean = lambda value: (value * seen).sum(-1) / n
    return {
        "disruption_displacement": mean(per_ped_mean),
        "disruption_max_displacement": (per_ped_max * seen).amax(-1),
        "velocity_deviation": mean(velocity_deviation),
        "ped_delay": mean(delay),
        "peds_affected": ((per_ped_max > affected_threshold) & seen).float().sum(-1),
        "reference_deviation": mean(reference_deviation),
    }


def all_metrics(run: Rollout, counterfactual: Rollout | None, batch: ScenarioBatch, robot_radius: float) -> dict:
    out = robot_metrics(run, batch, robot_radius)
    if counterfactual is not None:
        out.update(disruption_metrics(run, counterfactual, batch))
    return out


def summarize(metrics: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> dict[str, float]:
    """Mean of every metric over environments (NaN-aware, e.g. time_to_goal over successes)."""
    out = {}
    for key, value in metrics.items():
        v = value if mask is None else value[mask]
        finite = ~torch.isnan(v)
        out[key] = float(v[finite].mean()) if bool(finite.any()) else float("nan")
    return out
