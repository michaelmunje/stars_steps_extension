"""Score open-loop predictions on the common windows (``forecast_bench.windows``).

Prediction file ``preds/<set>/<method>.npz``: ``pred [E, K, 12, 2]`` float32 for the scored nodes
(``eval_mask``, in node order), same frame as the windows. Deterministic methods use K = 1.

Per scored agent (averaged over agents):
  min_ade / min_fde       best of K (K = 20: the standard minADE20 / minFDE20)
  mean_ade / mean_fde     average over the K samples (expected single-sample error)
  s0_ade / s0_fde         the first sample only
Per window (averaged over windows with >= 2 scored agents for collisions, all windows otherwise):
  min_joint_ade / fde     min over k of the window's mean agent ADE / FDE (sample k as a joint scene)
  col{20,40}_agent        P(agent is within 0.2 / 0.4 m of another scored agent's prediction of the same
                          sample at the same step), over (agent, sample) pairs
  col{20,40}_scene        P(some pair collides), over (window, sample) pairs
  The same collision rates on the ground-truth futures are the ``ground_truth`` row.
Kinematics: mean predicted speed (m/s, including the step from the last observation) vs GT, and the
1-D Wasserstein distance between the predicted and GT per-step speed distributions.
"""
from __future__ import annotations

import numpy as np

DT = 0.4
THRESHOLDS = (0.2, 0.4)


def scored(windows: dict) -> dict:
    """Scored-node views: last observed position, GT future, window id of every scored node."""
    mask = windows["eval_mask"]
    node_window = np.repeat(np.arange(len(windows["ptr"]) - 1), np.diff(windows["ptr"]))
    return dict(last=windows["obs"][mask, -1], gt=windows["fut"][mask], window=node_window[mask],
                obs=windows["obs"][mask])


def _collisions(pred: np.ndarray, window: np.ndarray, threshold: float, chunk: int = 4096):
    """pred [E, K, T, 2] -> (agent collides [E, K] bool, scene has >= 2 agents per agent [E])."""
    order = np.argsort(window, kind="stable")
    hit = np.zeros(pred.shape[:2], dtype=bool)
    bounds = np.flatnonzero(np.r_[True, window[order][1:] != window[order][:-1], True])
    multi = np.zeros(len(window), dtype=bool)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) < 2:
            continue
        multi[idx] = True
        p = pred[idx]                                   # [n, K, T, 2]
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)  # [n, n, K, T]
        d[np.arange(len(idx)), np.arange(len(idx))] = np.inf
        hit[idx] = (d < threshold).any(-1).any(1)
    return hit, multi


def speeds(last: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """[..., T, 2] future (with ``last`` [..., 2] before it) -> per-step speeds [..., T]."""
    last = np.broadcast_to(last[..., None, :], (*traj.shape[:-2], 1, 2))
    path = np.concatenate([last, traj], axis=-2)
    return np.linalg.norm(np.diff(path, axis=-2), axis=-1) / DT


def wasserstein(a: np.ndarray, b: np.ndarray, n: int = 1000) -> float:
    q = (np.arange(n) + 0.5) / n
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def score(windows: dict, pred: np.ndarray) -> dict[str, float]:
    s = scored(windows)
    gt, window = s["gt"], s["window"]
    if pred.shape[0] != len(gt) or pred.shape[2:] != gt.shape[1:]:
        raise ValueError(f"prediction {pred.shape} does not match {len(gt)} scored agents")
    err = np.linalg.norm(pred - gt[:, None], axis=-1)        # [E, K, T]
    ade, fde = err.mean(-1), err[..., -1]                     # [E, K]
    out = {"agents": len(gt), "windows": len(np.unique(window)), "k": pred.shape[1],
           "min_ade": ade.min(1).mean(), "min_fde": fde.min(1).mean(),
           "mean_ade": ade.mean(), "mean_fde": fde.mean(), "s0_ade": ade[:, 0].mean(), "s0_fde": fde[:, 0].mean()}
    # joint: per window, mean over agents for each k, then min over k
    counts = np.bincount(window)
    present = counts > 0
    for name, e in (("ade", ade), ("fde", fde)):
        per_window = np.zeros((len(counts), e.shape[1]))
        np.add.at(per_window, window, e)
        out[f"min_joint_{name}"] = (per_window[present] / counts[present, None]).min(1).mean()
    out.update(collision_rates(pred, window, prefix=""))
    sp, sp_gt = speeds(s["last"][:, None], pred), speeds(s["last"], gt)
    out.update(speed_mean=sp.mean(), speed_gt=sp_gt.mean(), speed_w1=wasserstein(sp.ravel(), sp_gt.ravel()))
    return {k: float(v) for k, v in out.items()}


def collision_rates(pred: np.ndarray, window: np.ndarray, prefix: str = "") -> dict[str, float]:
    out = {}
    for threshold in THRESHOLDS:
        hit, multi = _collisions(pred, window, threshold)
        tag = f"{prefix}col{int(round(threshold * 100))}"
        out[f"{tag}_agent"] = hit[multi].mean() if multi.any() else np.nan
        scene = np.zeros((window.max() + 1, pred.shape[1]), dtype=bool)
        np.logical_or.at(scene, window[multi], hit[multi])
        windows_multi = np.unique(window[multi])
        out[f"{tag}_scene"] = scene[windows_multi].mean() if len(windows_multi) else np.nan
    return out


def ground_truth(windows: dict) -> dict[str, float]:
    s = scored(windows)
    out = {"agents": len(s["gt"]), "windows": len(np.unique(s["window"])), "k": 1}
    out.update(collision_rates(s["gt"][:, None], s["window"]))
    sp = speeds(s["last"], s["gt"])
    out.update(speed_mean=sp.mean(), speed_gt=sp.mean(), speed_w1=0.0)
    return {k: float(v) for k, v in out.items()}
