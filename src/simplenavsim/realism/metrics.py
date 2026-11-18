"""Realism metrics for one simulated window at one horizon (scored agents only, world metres).

* ADE / FDE: mean / final Euclidean displacement (m); SDE: mean squared displacement (m^2), reported
  separately (never mixed into ADE).
* nADE / lADE: ADE over non-linear / near-straight agents; non-linear means the ground-truth future (from
  the last observed position) has a mean perpendicular distance from its start-end chord above ``NONLINEAR`` m.
* Collisions between scored agents, disc radius ``R`` (contact when centres are within 2R):
  unique colliding pairs / pairs, colliding pair-steps / pair-steps, agents in >= 1 collision, and
  hard / soft average collision severity (hACS / sACS, Mahzoon et al., IJCAI 2025: CS = 1 or
  (2R - d) / 2R for 0 < d <= 2R, summed over ordered pairs and steps, divided by agent-steps).
* Close contacts: agents that come within 0.2 / 0.4 m (centre to centre) of another scored agent at some step,
  ignoring pairs that are a walking group in the observation (within 1 m and velocities within 0.3 m/s at
  the last observed step, the rule of ``stars_ma_model.observed_groups``; uses no future information).
* Minimum separation between scored agents over the horizon.
* Goal: minimum distance to the oracle goal (MDE-goal), whether it came within ``REACHED`` m, and
  minimum distance to the ground-truth position at the horizon (MDE as in Mahzoon et al.).
* Kinematics (also for ground truth): per-step speed (m/s), acceleration (m/s^2) and absolute turning
  rate (deg/s, steps faster than 0.2 m/s), and the distance to the nearest other scored agent at every step
  (m, clipped at ``NN_CLIP``), returned as samples for distribution comparison.
"""
from __future__ import annotations

import numpy as np

from .data import DT

R = 0.2
NONLINEAR = 0.2
REACHED = 0.5
CONTACTS = (0.2, 0.4)
NN_CLIP = 5.0


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


def observed_groups(obs) -> np.ndarray:
    """[A, A] pairs within 1 m with velocities within 0.3 m/s at the last observed step of ``obs [A, >=2, 2]``."""
    last, vel = obs[:, -1], (obs[:, -1] - obs[:, -2]) / DT
    return (np.linalg.norm(last[:, None] - last[None], axis=-1) < 1.0) & (np.linalg.norm(vel[:, None] - vel[None], axis=-1) < 0.3)


def contacts(traj, obs) -> tuple[dict, np.ndarray]:
    """Agents in a non-group close contact per threshold, and nearest-neighbour distances [A * T] of ``traj [A, T, 2]``."""
    A = len(traj)
    if A < 2:
        return {f"contact{round(c * 100)}_agents": 0 for c in CONTACTS}, np.empty(0)
    d = np.linalg.norm(traj[:, None] - traj[None], axis=-1)                       # [A, A, T]
    d[np.arange(A), np.arange(A)] = np.inf
    closest = np.where(observed_groups(obs), np.inf, d.min(-1))                     # [A, A], groups ignored
    return ({f"contact{round(c * 100)}_agents": int((closest < c).any(1).sum()) for c in CONTACTS},
            np.minimum(d.min(1), NN_CLIP).ravel())


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
    close, nn = contacts(pred, window.obs[s])
    row = {"agents": int(s.sum()), "cohort": len(s), "ade_sum": ade.sum(), "fde_sum": fde.sum(), "sde_sum": sde.sum(),
           "nonlinear_agents": int(lin.sum()), "nade_sum": ade[lin].sum(), "linear_agents": int((~lin).sum()),
           "lade_sum": ade[~lin].sum(), "mde_goal_sum": dgoal.sum(),
           "goal_reached": int((dgoal < REACHED).sum()), "mde_end_sum": dend.sum(), **collisions(pred), **close}
    agents = [{"agent_id": int(i), "ade": a, "fde": f, "sde": q, "nonlinear": bool(n), "goal_at_boundary": bool(b),
               "min_goal_distance": g} for i, a, f, q, n, b, g in
              zip(window.agent_ids[s], ade, fde, sde, lin, window.goal_at_boundary[s], dgoal)]
    return row, agents, {**kinematics(start, pred), "nn_distance": nn}
