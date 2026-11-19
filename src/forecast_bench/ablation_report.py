"""Metrics and tables of the STARS ablation ladder on ETH/UCY leave-one-scene-out (agreed protocol).

Per rung, seed and held-out scene, from ``preds/ethucy_<fold>/<method>_s<seed>.npz`` (``pred [E, K, 12, 2]``
for the scored agents; K = 20 for stochastic models) and, for stochastic models, the most-likely prediction
``<method>_s<seed>_mode.npz`` (CVAE prior-mean latent, K = 1). At horizons 1.2 / 2.4 / 3.6 / 4.8 s (steps 3/6/9/12):

* ``ss_ade`` / ``ss_fde``  single-sample error: the error of one random sample, averaged over the K draws (what a
  simulator sees); for deterministic models the error of the prediction.
* ``ml_ade`` / ``ml_fde``  most-likely / deterministic prediction.
* ``nll``  KDE NLL (Trajectron++ / AgentFormer): a Gaussian KDE (Scott's bandwidth, as ``scipy.stats.gaussian_kde``,
  plus 1e-6 m^2 on the diagonal) fitted to the K samples of each agent and step; NLL of the true position, log-density
  clipped at -20 as Trajectron++, averaged over steps up to the horizon and over agents. n/a for deterministic models.
* ``col``  predicted collision rate at 0.2 m: share of (agent, sample) pairs that come within 0.2 m of another scored
  agent of the same window in the same sample at some step up to the horizon (windows with >= 2 scored agents);
  ``col_gt`` is the same on the ground truth.
* ``min_ade`` / ``min_fde``  best of K = 20 (published-methods comparison only).

All are per-agent micro averages over the scored agents of a scene; the 5-fold value is the mean over scenes; tables
give mean +- standard deviation over seeds.

    python -m forecast_bench.ablation_report --out results/stars_ablation_<date>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_bench.metrics import scored
from forecast_bench.windows import FOLDS, ROOT, load

BENCH = ROOT / "experiments/forecast_bench"
OFFICIAL = Path("/scratch/09865/mmunje/stars-forecast-benchmark/experiments/forecast_bench/preds")
HORIZONS = (3, 6, 9, 12)
SEEDS = (1, 2, 3)
LOG_PDF_FLOOR = -20.0
COLLISION = 0.2


def kde_nll(pred: np.ndarray, gt: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """``pred [E, K, T, 2]``, ``gt [E, T, 2]`` -> clipped KDE negative log-likelihood ``[E, T]``."""
    E, K, T, _ = pred.shape
    out = np.empty((E, T))
    factor = K ** (-1.0 / 6.0)                                  # Scott's rule, d = 2
    for a in range(0, E, chunk):
        p = pred[a:a + chunk].astype(np.float64).transpose(0, 2, 1, 3)    # [e, T, K, 2]
        g = gt[a:a + chunk].astype(np.float64)
        centred = p - p.mean(2, keepdims=True)
        cov = np.einsum("etki,etkj->etij", centred, centred) / (K - 1) * factor ** 2 + 1e-6 * np.eye(2)
        inv = np.linalg.inv(cov)
        logdet = np.log(np.linalg.det(cov))
        diff = g[:, :, None] - p                                           # [e, T, K, 2]
        maha = np.einsum("etki,etij,etkj->etk", diff, inv, diff)
        log_k = -0.5 * maha - np.log(2 * np.pi) - 0.5 * logdet[..., None]
        log_pdf = np.logaddexp.reduce(log_k, axis=2) - np.log(K)
        out[a:a + chunk] = -np.maximum(log_pdf, LOG_PDF_FLOOR)
    return out


def closest_by_horizon(pred: np.ndarray, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running minimum over steps of the distance to the nearest other scored agent of the window (same sample):
    ``[E, K, T]``, and whether the agent's window has >= 2 scored agents ``[E]``."""
    E, K, T, _ = pred.shape
    out = np.full((E, K, T), np.inf)
    order = np.argsort(window, kind="stable")
    bounds = np.flatnonzero(np.r_[True, window[order][1:] != window[order][:-1], True])
    multi = np.zeros(E, dtype=bool)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) < 2:
            continue
        multi[idx] = True
        p = pred[idx]
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)                   # [n, n, K, T]
        d[np.arange(len(idx)), np.arange(len(idx))] = np.inf
        out[idx] = np.minimum.accumulate(d.min(1), axis=-1)
    return out, multi


def fold_metrics(windows: dict, pred: np.ndarray, mode: np.ndarray | None) -> dict[str, float]:
    s = scored(windows)
    gt = s["gt"]
    err = np.linalg.norm(pred - gt[:, None], axis=-1)                       # [E, K, T]
    stochastic = pred.shape[1] > 1
    if mode is not None:
        ml = np.linalg.norm(mode - gt[:, None], axis=-1)
    else:   # deterministic: the prediction itself; stochastic without a most-likely file: n/a
        ml = np.full_like(err, np.nan) if stochastic else err
    nll = kde_nll(pred, gt) if stochastic else None
    closest, multi = closest_by_horizon(pred, s["window"])
    closest_gt, _ = closest_by_horizon(gt[:, None], s["window"])
    out = {"agents": len(gt), "k": pred.shape[1]}
    for h in HORIZONS:
        out[f"ss_ade@{h}"] = err[..., :h].mean(-1).mean()
        out[f"ss_fde@{h}"] = err[..., h - 1].mean()
        out[f"ml_ade@{h}"] = ml[..., :h].mean(-1).mean()
        out[f"ml_fde@{h}"] = ml[..., h - 1].mean()
        out[f"nll@{h}"] = nll[:, :h].mean() if stochastic else np.nan
        out[f"col@{h}"] = (closest[multi, :, h - 1] < COLLISION).mean()
        out[f"col_gt@{h}"] = (closest_gt[multi, :, h - 1] < COLLISION).mean()
        out[f"min_ade@{h}"] = err[..., :h].mean(-1).min(1).mean()
        out[f"min_fde@{h}"] = err[..., h - 1].min(1).mean()
    return {k: float(v) for k, v in out.items()}


def collect(methods: dict[str, str], root: Path, seeds=SEEDS, official: dict[str, Path] | None = None) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        windows = load(root / "windows" / f"ethucy_{fold}.npz")
        for rung, method in methods.items():
            for seed in seeds:
                path = root / "preds" / f"ethucy_{fold}" / f"{method}_s{seed}.npz"
                if not path.exists():
                    continue
                pred = np.load(path)["pred"]
                mode_path = path.with_name(f"{method}_s{seed}_mode.npz")
                mode = np.load(mode_path)["pred"] if mode_path.exists() else None
                if pred.shape[1] > 1 and mode is None:
                    print(f"warning: {mode_path} missing; most-likely metrics use the samples")
                rows.append({"rung": rung, "method": method, "seed": seed, "fold": fold,
                             **fold_metrics(windows, pred, mode)})
        for name, directory in (official or {}).items():
            path = directory / f"ethucy_{fold}" / f"{name}.npz"
            if path.exists():
                rows.append({"rung": name, "method": name, "seed": 0, "fold": fold,
                             **fold_metrics(windows, np.load(path)["pred"], None)})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """5-fold average per (rung, seed), then mean / std over seeds; only (rung, seed) with all 5 folds."""
    metrics = [c for c in df.columns if "@" in c]
    complete = df.groupby(["rung", "seed"]).fold.transform("nunique") == len(FOLDS)
    avg = df[complete].groupby(["rung", "seed"])[metrics].mean().reset_index()
    mean = avg.groupby("rung")[metrics].mean()
    std = avg.groupby("rung")[metrics].std(ddof=1)
    seeds = avg.groupby("rung").seed.nunique()
    return mean.join(std, rsuffix="_std").join(seeds.rename("seeds"))


def fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if np.isfinite(std) else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=BENCH)
    parser.add_argument("--rungs", type=Path, default=ROOT / "runs/stars_ablation/rungs.json",
                        help='{"L0 original STARS": "abl_L0", ...} (method prefix; files <prefix>_s<seed>.npz)')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = json.loads(args.rungs.read_text())
    official = {name: OFFICIAL for name in ("cv", "sgan_official", "trajectron_official", "agentformer_official",
                                            "socialvae_official", "moflow_official")}
    args.out.mkdir(parents=True, exist_ok=True)
    df = collect(methods, args.root, official=official)
    df.to_csv(args.out / "per_fold_seed.csv", index=False)
    summary = summarise(df)
    summary.to_csv(args.out / "summary.csv")
    print(summary[[c for c in summary.columns if c.endswith("@12")]].round(3).to_string())


if __name__ == "__main__":
    main()
