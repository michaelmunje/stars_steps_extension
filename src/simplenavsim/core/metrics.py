"""Social navigation metrics from recorded rollouts (per environment, ``[B]`` tensors).

Task: ``success``, ``collision``, ``timeout``, ``time_to_goal`` (NaN unless successful),
``normalized_time`` (episode time / limit, 1 on failure), ``time_efficiency`` (straight-line time
at top speed / time to goal) and ``path_efficiency`` (straight-line distance / path driven), both 0
on failure and measured to the edge of the goal circle; ``loops`` (full turns beyond half a turn)
and ``command_smoothness`` (mean |dv|/dt + |dw|/dt).

Comfort (surface to surface): ``min_ped_clearance`` and ``intimate_violation_rate`` (fraction of
the robot's steps with a pedestrian closer than 0.45 m).

Disruption: ``disruption_local`` (distance between each pedestrian's position with and without
the robot, while within 3 m of the robot, averaged over pedestrians that came that close) and,
in the STARS crowd, ``robot_influence`` (how much the robot changes the pedestrians' forecasts).
"""
from __future__ import annotations

import math

import torch

from simplenavsim.core.geometry import FAR, obstacle_clearance, wrap_angle
from simplenavsim.core.rollout import Rollout
from simplenavsim.core.state import ScenarioBatch

INTIMATE_DISTANCE = 0.45
TTC_CAP = 10.0


def robot_metrics(run: Rollout, batch: ScenarioBatch, robot_radius: float, max_speed: float = 1.0,
                  goal_tolerance: float = 0.0) -> dict[str, torch.Tensor]:
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
        "timeout": (~run.success & ~collision).float(),
        "time_to_goal": torch.where(run.success, duration, torch.full_like(duration, float("nan"))),
        "normalized_time": torch.where(run.success, duration / batch.duration, torch.ones_like(duration)),
    }
    steps = torch.linalg.norm(run.robot_pose[:, 1:, :2] - run.robot_pose[:, :-1, :2], dim=-1)
    path_length = (steps * moving_steps).sum(-1)
    straight = (torch.linalg.norm(batch.robot_goal - batch.robot_start[:, :2], dim=-1) - goal_tolerance).clamp(min=1e-3)
    # 1 = straight to the goal; detours, loops and orbits lower it. Failed episodes count as 0.
    out["path_efficiency"] = torch.where(run.success, (straight / path_length.clamp(min=1e-6)).clamp(max=1.0),
                                         torch.zeros_like(straight))
    # 1 = drove straight to the goal at top speed; waiting, slow driving and detours lower it.
    out["time_efficiency"] = torch.where(run.success, (straight / max_speed / duration.clamp(min=1e-6)).clamp(max=1.0),
                                         torch.zeros_like(straight))

    # Loops: total heading change beyond half a turn, in full turns (a straight run with swerves ~0).
    turn = wrap_angle(run.robot_pose[:, 1:, 2] - run.robot_pose[:, :-1, 2]).abs()
    out["loops"] = torch.relu((turn * moving_steps).sum(-1) - math.pi) / (2 * math.pi)

    rel = run.robot_pose[:, None, :, :2] - run.ped_pos                           # [B, P, K, 2]
    gap = torch.linalg.norm(rel, dim=-1) - batch.ped_radius[..., None] - robot_radius
    present = run.ped_active & live[:, None]
    gap = torch.where(present, gap, torch.full_like(gap, float("inf")))
    nearest = gap.min(dim=1).values                                              # [B, K]
    closest = nearest.min(dim=-1).values
    out["min_ped_clearance"] = torch.where(torch.isfinite(closest), closest, torch.full_like(closest, float("nan")))
    live_count = live.float().sum(-1).clamp(min=1)
    out["intimate_violation_rate"] = ((nearest < INTIMATE_DISTANCE) & live).float().sum(-1) / live_count

    dv = (run.command[:, 1:, 0] - run.command[:, :-1, 0]).abs() / run.dt
    dw = (run.command[:, 1:, 1] - run.command[:, :-1, 1]).abs() / run.dt
    count = moving_steps.sum(-1).clamp(min=1)
    out["command_smoothness"] = ((dv + dw) * moving_steps).sum(-1) / count
    return out


def disruption_metrics(run: Rollout, counterfactual: Rollout, batch: ScenarioBatch,
                       local_radius: float = 3.0) -> dict[str, torch.Tensor]:
    both = run.ped_active & counterfactual.ped_active                             # [B, P, K]
    displacement = torch.linalg.norm(run.ped_pos - counterfactual.ped_pos, dim=-1) * both
    seen = (run.ped_active | counterfactual.ped_active).any(-1) & batch.ped_mask   # [B, P]
    # Only while the pedestrian is near the robot: learned crowds amplify any perturbation
    # everywhere over time, so global displacement mostly measures that divergence.
    near = torch.linalg.norm(run.ped_pos - run.robot_pose[:, None, :, :2], dim=-1) < local_radius
    local = both & near
    local_mean = (displacement * local).sum(-1) / local.float().sum(-1).clamp(min=1)
    ever_near = local.any(-1) & seen
    return {"disruption_local": (local_mean * ever_near).sum(-1) / ever_near.float().sum(-1).clamp(min=1)}


def influence_metrics(run: Rollout, batch: ScenarioBatch) -> dict[str, torch.Tensor]:
    """Robot-attributable change of learned-crowd forecasts (``stars_crowd``): per re-forecast, the
    mean distance between the 4.8 s forecasts with and without the robot in the graph. Immune to
    the closed-loop divergence that makes trajectory counterfactuals meaningless for learned crowds."""
    value = run.ped_influence                                                    # [B, P, K], 0 between re-forecasts
    record = value > 0
    seen = run.ped_active.any(-1) & batch.ped_mask
    per_ped = value.sum(-1) / record.float().sum(-1).clamp(min=1)                # mean over re-forecasts
    n = seen.float().sum(-1).clamp(min=1)
    return {"robot_influence": (per_ped * seen).sum(-1) / n}


def principles_metrics(run: Rollout, batch: ScenarioBatch, robot_radius: float, goal_tolerance: float = 0.0,
                       space_threshold: float = 0.5) -> dict[str, torch.Tensor]:
    """Table I of Francis et al., "Principles and Guidelines for Evaluating Social Robot Navigation
    Algorithms" (2023), per episode. Distances to people are surface to surface. ``SC`` is the share
    of the robot's steps with every pedestrian at least ``space_threshold`` away (personal space
    compliance at 0.5 m); ``TTC`` assumes constant velocities and is capped at 10 s (never on a
    collision course); ``FP`` counts 5 s windows with under 0.1 m of progress; ``ST`` is time below
    0.05 m/s; ``Aavg`` / ``Javg`` / ``CDavg`` average magnitudes (signed ones cancel out)."""
    B, K = run.robot_v.shape
    device, dt = run.robot_v.device, run.dt
    live = torch.arange(K, device=device)[None] <= run.end_step[:, None]
    nan = torch.full((B,), float("nan"), device=device)
    finite = lambda x: torch.where(torch.isfinite(x), x, nan)

    def stats(x, mask):                                      # min, mean magnitude, max over steps (NaN if none)
        lo = torch.where(mask, x, torch.full_like(x, math.inf)).amin(-1)
        hi = torch.where(mask, x, torch.full_like(x, -math.inf)).amax(-1)
        mean = (x.abs() * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return finite(lo), torch.where(mask.any(-1), mean, nan), finite(hi)

    xy = run.robot_pose[..., :2]
    path = (torch.linalg.norm(xy[:, 1:] - xy[:, :-1], dim=-1) * live[:, 1:]).sum(-1)
    straight = (torch.linalg.norm(batch.robot_goal - batch.robot_start[:, :2], dim=-1) - goal_tolerance).clamp(min=1e-3)
    success = run.success.float()
    out = {"S": success, "C": (run.ped_collision | run.obstacle_collision).float(),
           "WC": run.obstacle_collision.float(), "HC": run.ped_collision.float(),
           "TO": (~run.success & ~run.ped_collision & ~run.obstacle_collision).float(),
           "T": torch.where(run.success, run.end_step.float() * dt, nan), "PL": path,
           "SPL": success * straight / torch.maximum(straight, path)}
    to_goal = torch.linalg.norm(xy - batch.robot_goal[:, None], dim=-1)
    W = int(round(5.0 / dt))
    starts = torch.arange(0, K - W, W, device=device)
    out["FP"] = ((to_goal[:, starts] - to_goal[:, starts + W] < 0.1) & live[:, starts + W]).float().sum(-1)
    out["ST"] = ((run.robot_v.abs() < 0.05) & live).float().sum(-1) * dt
    acc = (run.robot_v[:, 1:] - run.robot_v[:, :-1]) / dt
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt
    for name, x, mask in (("V", run.robot_v, live), ("A", acc, live[:, 1:]), ("J", jerk, live[:, 2:])):
        out[f"{name}min"], out[f"{name}avg"], out[f"{name}max"] = stats(x, mask)
    clearance = obstacle_clearance(xy, batch.circles, batch.circle_mask, batch.segments, batch.segment_mask)[0]
    has_obstacle = live & (clearance < FAR / 2)
    out["CDmin"], out["CDavg"], _ = stats(clearance - robot_radius, has_obstacle)
    Kp = min(K, run.ped_pos.shape[2])
    ped, active = run.ped_pos[:, :, :Kp], run.ped_active[:, :, :Kp] & live[:, None, :Kp]
    rel = ped - xy[:, None, :Kp]                                                # [B, P, K, 2]
    radius = batch.ped_radius[:, :, None] + robot_radius
    gap = torch.where(active, torch.linalg.norm(rel, dim=-1) - radius, torch.full_like(radius.expand_as(active), math.inf))
    nearest = gap.amin(1)                                                       # [B, K]
    out["DHmin"] = finite(torch.where(live[:, :Kp], nearest, torch.full_like(nearest, math.inf)).amin(-1))
    out["SC"] = ((nearest >= space_threshold) & live[:, :Kp]).float().sum(-1) / live[:, :Kp].float().sum(-1)
    theta = run.robot_pose[:, None, :Kp, 2]
    robot_vel = run.robot_v[:, None, :Kp, None] * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    ped_vel = torch.zeros_like(ped)
    ped_vel[:, :, 1:] = (ped[:, :, 1:] - ped[:, :, :-1]) / dt
    v = ped_vel - robot_vel
    a, b = (v * v).sum(-1), 2 * (rel * v).sum(-1)
    c = (rel * rel).sum(-1) - radius ** 2
    disc = b * b - 4 * a * c
    t = (-b - disc.clamp(min=0).sqrt()) / (2 * a).clamp(min=1e-9)
    ttc = torch.where(c <= 0, torch.zeros_like(t), torch.where((disc >= 0) & (t >= 0) & (a > 1e-9), t, torch.full_like(t, math.inf)))
    out["TTC"] = torch.where(active, ttc, torch.full_like(ttc, math.inf)).amin(-1).amin(-1).clamp(max=TTC_CAP)
    return out


def all_metrics(run: Rollout, counterfactual: Rollout | None, batch: ScenarioBatch, robot_radius: float,
                max_speed: float = 1.0, goal_tolerance: float = 0.0) -> dict:
    out = robot_metrics(run, batch, robot_radius, max_speed, goal_tolerance)
    if run.ped_influence is not None:
        out.update(influence_metrics(run, batch))
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
