"""Realism metrics for one simulated window at one horizon (scored agents only, world metres).

* ADE / FDE: mean / final Euclidean displacement (m); SDE: mean squared displacement (m^2), reported
  separately (never mixed into ADE).
* nADE: ADE over non-linear agents, whose ground-truth future (from the last observed position) has a
  mean perpendicular distance from its start-end chord above ``NONLINEAR`` m.
* Collisions between scored agents, disc radius ``R`` (contact when centres are within 2R):
  unique colliding pairs / pairs, colliding pair-steps / pair-steps, agents in >= 1 collision, and
  hard / soft average collision severity (hACS / sACS, Mahzoon et al., IJCAI 2025: CS = 1 or
  (2R - d) / 2R for 0 < d <= 2R, summed over ordered pairs and steps, divided by agent-steps).
* Minimum separation between scored agents over the horizon.
* Goal: minimum distance to the oracle goal (MDE-goal), whether it came within ``REACHED`` m, and
  minimum distance to the ground-truth position at the horizon (MDE as in Mahzoon et al.).
* Kinematics (also for ground truth): per-step speed (m/s), acceleration (m/s^2) and absolute turning
  rate (deg/s, steps faster than 0.2 m/s), returned as samples for distribution comparison.
"""
from __future__ import annotations

import numpy as np

from .data import DT

R = 0.2
NONLINEAR = 0.2
REACHED = 0.5


def nonlinear(start, future) -> np.ndarray:
    """[A] ground-truth futures that are not straight lines (see module docstring)."""
    path = np.concatenate([start[:, None], future], 1)
    chord = path[:, -1] - path[:, 0]
    length = np.linalg.norm(chord, axis=-1, keepdims=True)
    unit = chord / np.maximum(length, 1e-9)
    rel = path - path[:, :1]
    perp = np.abs(rel[..., 0] * unit[:, None, 1] - rel[..., 1] * unit[:, None, 0])
    return perp.mean(1) > NONLINEAR


def collisions(traj) -> dict:
    """Collision statistics of ``traj [A, T, 2]`` (A agents simulated together)."""
    A, T = traj.shape[:2]
    out = {"pairs": A * (A - 1) // 2, "pair_steps": A * (A - 1) // 2 * T, "colliding_pairs": 0, "colliding_pair_steps": 0,
           "agents_in_collision": 0, "hacs_sum": 0.0, "sacs_sum": 0.0, "agent_steps": A * T, "min_separation": np.nan}
    if A < 2:
        return out
    d = np.linalg.norm(traj[:, None] - traj[None], axis=-1)                      # [A, A, T]
    off = ~np.eye(A, dtype=bool)
    hit = (d <= 2 * R) & (d > 0) & off[..., None]
    upper = np.triu(np.ones((A, A), bool), 1)
    out.update(colliding_pairs=int((hit.any(-1) & upper).sum()), colliding_pair_steps=int((hit & upper[..., None]).sum()),
               agents_in_collision=int(hit.any(-1).any(-1).sum()), hacs_sum=float(hit.sum()),
               sacs_sum=float(np.where(hit, (2 * R - d) / (2 * R), 0.0).sum()),
               min_separation=float(d[upper].min()))
    return out


def kinematics(start, traj) -> dict:
    """Per-step speed, acceleration and turning rate samples of ``traj [A, T, 2]`` starting from ``start [A, 2]``."""
    path = np.concatenate([start[:, None], traj], 1)
    vel = np.diff(path, axis=1) / DT
    speed = np.linalg.norm(vel, axis=-1)
    acc = np.linalg.norm(np.diff(vel, axis=1), axis=-1) / DT
    heading = np.arctan2(vel[..., 1], vel[..., 0])
    turn = np.abs((np.diff(heading, axis=1) + np.pi) % (2 * np.pi) - np.pi) / DT
    moving = (speed[:, 1:] > 0.2) & (speed[:, :-1] > 0.2)
    return {"speed": speed.ravel(), "acceleration": acc.ravel(), "turning": np.degrees(turn[moving])}


def window_metrics(pred, window, horizon: int):
    """Per-window sums (for micro averages / bootstrap) and per-agent rows of ``pred [N, >=H, 2]``."""
    s = window.scored(horizon)
    pred, gt = pred[s, :horizon], window.future[s, :horizon]
    start = window.obs[s, -1]
    err = np.linalg.norm(pred - gt, axis=-1)
    lin = nonlinear(start, gt)
    dgoal = np.linalg.norm(pred - window.goal[s, None], axis=-1).min(1)
    dend = np.linalg.norm(pred - gt[:, -1:], axis=-1).min(1)
    ade, fde, sde = err.mean(1), err[:, -1], (err ** 2).mean(1)
    row = {"agents": int(s.sum()), "cohort": len(s), "ade_sum": ade.sum(), "fde_sum": fde.sum(), "sde_sum": sde.sum(),
           "nonlinear_agents": int(lin.sum()), "nade_sum": ade[lin].sum(), "mde_goal_sum": dgoal.sum(),
           "goal_reached": int((dgoal < REACHED).sum()), "mde_end_sum": dend.sum(), **collisions(pred)}
    agents = [{"agent_id": int(i), "ade": a, "fde": f, "sde": q, "nonlinear": bool(n), "goal_at_boundary": bool(b),
               "min_goal_distance": g} for i, a, f, q, n, b, g in
              zip(window.agent_ids[s], ade, fde, sde, lin, window.goal_at_boundary[s], dgoal)]
    return row, agents, kinematics(start, pred)
