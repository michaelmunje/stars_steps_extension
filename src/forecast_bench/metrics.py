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
Realism (same definitions as the closed-loop ETH/UCY benchmark, ``simplenavsim.realism.metrics``), pooled over
all K samples:
  mean_ade_straight / mean_ade_nonlinear   mean_ade over agents whose GT future stays within / departs from its
                          start-end chord by more than 0.2 m on average (``nonlinear``); nonlinear_share
  contact{20,40}_agent    like col{20,40}_agent but ignoring pairs that are a walking group in the observation
                          (within 1 m, velocities within 0.3 m/s at the last observed step; no future information)
  accel_w1 / turn_w1      W1 to GT of per-step acceleration (m/s^2) and turning rate (deg/s, steps > 0.2 m/s)
  nn_w1                   W1 to GT of the distance from each scored agent to its nearest other scored agent
                          at every step (same sample; windows with >= 2 scored agents; clipped at 5 m)
"""
from __future__ import annotations

import numpy as np

from simplenavsim.realism.metrics import kinematics, nonlinear

DT = 0.4
THRESHOLDS = (0.2, 0.4)
NN_CLIP = 5.0


def observed_groups(obs: np.ndarray) -> np.ndarray:
    """[A, A] walking-group pairs of ``obs [A, 8, 2]``: within 1 m and velocities within 0.3 m/s at the last
    observed step (``baselines.stars_ma_model.observed_groups``)."""
    last, vel = obs[:, -1], (obs[:, -1] - obs[:, -2]) / DT
    return (np.linalg.norm(last[:, None] - last[None], axis=-1) < 1.0) & (np.linalg.norm(vel[:, None] - vel[None], axis=-1) < 0.3)


def scored(windows: dict) -> dict:
    """Scored-node views: last observed position, GT future, window id of every scored node."""
    mask = windows["eval_mask"]
    node_window = np.repeat(np.arange(len(windows["ptr"]) - 1), np.diff(windows["ptr"]))
    return dict(last=windows["obs"][mask, -1], gt=windows["fut"][mask], window=node_window[mask],
                obs=windows["obs"][mask])


def _proximity(pred: np.ndarray, window: np.ndarray, obs: np.ndarray):
    """pred [E, K, T, 2] -> closest approach to any other scored agent of the window [E, K] (all pairs, and
    without observed-group pairs), nearest-neighbour distance [E, K, T], scene has >= 2 agents [E]."""
    order = np.argsort(window, kind="stable")
    closest = np.full(pred.shape[:2], np.inf)
    closest_ng = np.full(pred.shape[:2], np.inf)
    nn = np.full(pred.shape[:3], np.inf, dtype=np.float32)
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
        nn[idx] = d.min(1)
        pair = d.min(-1)                                # [n, n, K]
        closest[idx] = pair.min(1)
        closest_ng[idx] = np.where(observed_groups(obs[idx])[..., None], np.inf, pair).min(1)
    return closest, closest_ng, nn, multi


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
    sp, sp_gt = speeds(s["last"][:, None], pred), speeds(s["last"], gt)
    out.update(speed_mean=sp.mean(), speed_gt=sp_gt.mean(), speed_w1=wasserstein(sp.ravel(), sp_gt.ravel()))
    lin = ~nonlinear(s["last"], gt)
    out.update(mean_ade_straight=ade[lin].mean(), mean_ade_nonlinear=ade[~lin].mean(), nonlinear_share=1 - lin.mean())
    out.update(realism(s, pred, reference=realism(s, gt[:, None])))
    return {k: float(v) for k, v in out.items()}


def realism(s: dict, pred: np.ndarray, reference: dict | None = None) -> dict:
    """Contact rates and distribution samples of ``pred [E, K, T, 2]``; with ``reference`` (the GT's own
    output), the W1 distances to it instead of the samples."""
    closest, closest_ng, nn, multi = _proximity(pred, s["window"], s["obs"])
    out = {}
    for threshold in THRESHOLDS:
        tag = int(round(threshold * 100))
        out[f"col{tag}_agent"] = (closest[multi] < threshold).mean() if multi.any() else np.nan
        out[f"contact{tag}_agent"] = (closest_ng[multi] < threshold).mean() if multi.any() else np.nan
        scene = np.zeros((s["window"].max() + 1, pred.shape[1]), dtype=bool)
        np.logical_or.at(scene, s["window"][multi], closest[multi] < threshold)
        windows_multi = np.unique(s["window"][multi])
        out[f"col{tag}_scene"] = scene[windows_multi].mean() if len(windows_multi) else np.nan
    E, K, T = pred.shape[:3]
    kin = kinematics(np.repeat(s["last"], K, axis=0), pred.reshape(E * K, T, 2))
    samples = {"accel": kin["acceleration"], "turn": kin["turning"], "nn": np.minimum(nn[multi], NN_CLIP).ravel()}
    if reference is None:
        return {**out, "_samples": samples}
    return {**out, **{f"{k}_w1": wasserstein(v, reference["_samples"][k]) for k, v in samples.items()}}


def ground_truth(windows: dict) -> dict[str, float]:
    s = scored(windows)
    out = {"agents": len(s["gt"]), "windows": len(np.unique(s["window"])), "k": 1}
    out.update({k: v for k, v in realism(s, s["gt"][:, None]).items() if k != "_samples"})
    sp = speeds(s["last"], s["gt"])
    out.update(speed_mean=sp.mean(), speed_gt=sp.mean(), speed_w1=0.0, accel_w1=0.0, turn_w1=0.0, nn_w1=0.0,
               nonlinear_share=1 - (~nonlinear(s["last"], s["gt"])).mean())
    return {k: float(v) for k, v in out.items()}
