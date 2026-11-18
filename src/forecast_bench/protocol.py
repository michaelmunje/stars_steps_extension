"""ETH/UCY protocol helpers: LOSO configuration selection and block-bootstrap confidence intervals.

    python -m forecast_bench.protocol select      # per (kind, seed, fold): config with the best val minADE20
    python -m forecast_bench.protocol ci --out results/ethucy_protocol_<date>

``select`` reads ``loso/sweep/<kind>_s<seed>/<config>/<fold>/summary.json`` (``forecast_bench.finetune_stars_loo``)
and links the chosen run to ``loso/ep_<kind>_s<seed>/<fold>`` (what ``predict --method ep_<kind>_s<seed>`` loads).
Only the fold's val split (the other four scenes) is used; the held-out scene never is.

``ci``: 95% block-bootstrap CIs of the 5-fold average. Blocks are 50 consecutive window starts (20 s) of one
recording, resampled within each fold (2000 reps); seed-replicated methods (``ep_<kind>_s<seed>...``) use the
per-agent mean over seeds, and the seed standard deviation of the 5-fold average is reported next to it.
Paired differences use the same resampled blocks for both methods.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_bench.windows import FOLDS, ROOT, load

BENCH = ROOT / "experiments/forecast_bench"
BLOCK = 50
REPS = 2000


def select(root: Path = BENCH) -> pd.DataFrame:
    rows = []
    for summary in sorted((root / "loso/sweep").glob("*/*/*/summary.json")):
        s = json.loads(summary.read_text())
        kind_seed, config, fold = summary.parts[-4:-1]
        rows.append({"run": kind_seed, "config": config, "fold": fold, "val_min_ade": s["best_val_min_ade"],
                     "best_epoch": s["best_epoch"], "path": summary.parent})
    df = pd.DataFrame(rows)
    chosen = df.loc[df.groupby(["run", "fold"]).val_min_ade.idxmin()]
    for r in chosen.itertuples():
        link = root / "loso" / f"ep_{r.run}" / r.fold
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / "sweep" / r.run / r.config / r.fold)
    df["chosen"] = df.index.isin(chosen.index)
    df.drop(columns="path").to_csv(root / "loso/selection.csv", index=False)
    print(chosen.drop(columns="path").to_string(index=False))
    return df


def per_agent(windows: dict, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per scored agent: min/mean/first-sample ADE, FDE; window index; contact (< 0.4 / 0.2 m, groups ignored)
    averaged over samples (NaN for agents alone in their window)."""
    from forecast_bench.metrics import _proximity, scored

    s = scored(windows)
    err = np.linalg.norm(pred - s["gt"][:, None], axis=-1)
    ade, fde = err.mean(-1), err[..., -1]
    _, closest_ng, _, multi = _proximity(pred, s["window"], s["obs"])
    c40 = np.where(multi, (closest_ng < 0.4).mean(1), np.nan)
    c20 = np.where(multi, (closest_ng < 0.2).mean(1), np.nan)
    return {"min_ade": ade.min(1), "min_fde": fde.min(1), "mean_ade": ade.mean(1), "mean_fde": fde.mean(1),
            "s0_ade": ade[:, 0], "ade_k": ade, "fde_k": fde, "contact40": c40, "contact20": c20, "window": s["window"]}


def blocks(windows: dict) -> np.ndarray:
    """Block id per window: 50 consecutive window starts of one recording."""
    keys = windows["window_key"]
    rec = np.array([k.rsplit(":", 1)[0] for k in keys])
    frame = np.array([float(k.rsplit(":", 1)[1]) for k in keys])
    out = np.zeros(len(keys), dtype=np.int64)
    nxt = 0
    for r in dict.fromkeys(rec):
        idx = np.flatnonzero(rec == r)
        idx = idx[np.argsort(frame[idx], kind="stable")]
        out[idx] = nxt + np.arange(len(idx)) // BLOCK
        nxt = out[idx].max() + 1
    return out


METRICS = ["min_ade", "min_fde", "min_joint_ade", "min_joint_fde", "mean_ade", "mean_fde", "contact40", "contact20"]


def fold_stats(a: dict, n_windows: int) -> dict[str, np.ndarray]:
    """Per-window sums for every metric (so a block resample is a weighted sum)."""
    w = a["window"]
    out = {}
    for m in ("min_ade", "min_fde", "mean_ade", "mean_fde"):
        out[m] = (np.bincount(w, a[m], n_windows), np.bincount(w, None, n_windows))
    for m, k in (("min_joint_ade", "ade_k"), ("min_joint_fde", "fde_k")):
        per = np.zeros((n_windows, a[k].shape[1]))
        np.add.at(per, w, a[k])
        cnt = np.bincount(w, None, n_windows)
        joint = np.where(cnt > 0, (per / np.maximum(cnt, 1)[:, None]).min(1), 0.0)
        out[m] = (joint, (cnt > 0).astype(float))
    for m in ("contact40", "contact20"):
        ok = ~np.isnan(a[m])
        out[m] = (np.bincount(w[ok], a[m][ok], n_windows), np.bincount(w[ok], None, n_windows))
    return out


def load_method(method: str, fold: str, root: Path) -> np.ndarray | None:
    """Prediction of ``method`` on one fold; ``ep_<kind><suffix>`` = all seeds stacked [S, E, K, 12, 2]."""
    d = root / "preds" / f"ethucy_{fold}"
    if (d / f"{method}.npz").exists():
        with np.load(d / f"{method}.npz") as f:
            return f["pred"][None]
    m = re.fullmatch(r"ep_([a-z_]+?)(_os4|_mode)?", method)
    if not m:
        return None
    files = sorted(p for p in d.glob(f"ep_{m[1]}_s*{m[2] or ''}.npz")
                   if re.fullmatch(rf"ep_{m[1]}_s\d+{m[2] or ''}", p.stem))
    if not files:
        return None
    return np.stack([np.load(p)["pred"] for p in files])


def ci(methods: list[str], pairs: list[tuple[str, str]], root: Path = BENCH, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stats, seedstats = {}, {}
    weights = {}
    for fold in FOLDS:
        windows = load(root / "windows" / f"ethucy_{fold}.npz")
        nw = len(windows["ptr"]) - 1
        blk = blocks(windows)
        nb = blk.max() + 1
        # one multinomial resample of blocks per rep -> window weights, shared by every method (paired)
        counts = rng.multinomial(nb, np.full(nb, 1 / nb), size=REPS).astype(float)       # [R, nb]
        weights[fold] = counts[:, blk]                                                    # [R, W]
        for method in methods:
            pred = load_method(method, fold, root)
            if pred is None:
                continue
            per_seed = [fold_stats(per_agent(windows, p), nw) for p in pred]
            # expected value over seeds: average the per-window sums (same windows/agents in every seed)
            stats[method, fold] = {k: (np.mean([s[k][0] for s in per_seed], 0), per_seed[0][k][1]) for k in METRICS}
            seedstats[method, fold] = [{k: s[k][0].sum() / max(s[k][1].sum(), 1) for k in METRICS} for s in per_seed]
    rows, diffs = [], []

    def boot(method):
        out = {}
        for k in METRICS:
            point, reps = [], []
            for fold in FOLDS:
                num, den = stats[method, fold][k]
                point.append(num.sum() / den.sum())
                reps.append((weights[fold] @ num) / np.maximum(weights[fold] @ den, 1e-9))
            out[k] = (np.mean(point), np.mean(reps, 0))
        return out

    full = [m for m in methods if all((m, f) in stats for f in FOLDS)]
    booted = {m: boot(m) for m in full}
    for m in full:
        row = {"method": m, "seeds": len(seedstats[m, FOLDS[0]])}
        for k in METRICS:
            point, reps = booted[m][k]
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = point, *np.percentile(reps, [2.5, 97.5])
            per_seed = np.mean([[s[k] for s in seedstats[m, f]] for f in FOLDS], 0)
            row[f"{k}_seed_sd"] = per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan
            for f in FOLDS:
                num, den = stats[m, f][k]
                row[f"{k}@{f}"] = num.sum() / den.sum()
        rows.append(row)
    for a, b in pairs:
        if a not in booted or b not in booted:
            continue
        row = {"a": a, "b": b}
        for k in METRICS:
            d = booted[a][k][1] - booted[b][k][1]
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = booted[a][k][0] - booted[b][k][0], *np.percentile(d, [2.5, 97.5])
        diffs.append(row)
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "ci"])
    parser.add_argument("--root", type=Path, default=BENCH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--methods", default="socialvae_official,moflow_official,agentformer_official,trajectron_official,"
                        "sgan_official,cv,ep_zs,ep_zs_os4,ep_zs_mode,ep_loso,ep_loso_os4,ep_loso_mode,ep_scratch,"
                        "ep_scratch_os4,ep_scratch_mode,stars_noethucy,stars_loo")
    args = parser.parse_args()
    if args.stage == "select":
        select(args.root)
        return
    methods = args.methods.split(",")
    ours = ["ep_loso", "ep_loso_os4", "ep_loso_mode", "ep_zs", "ep_zs_os4"]
    pairs = [(o, b) for o in ours for b in methods if b not in ours]
    pairs += [("ep_loso", "ep_zs"), ("ep_loso_os4", "ep_zs_os4"), ("ep_loso_mode", "ep_zs_mode"),
              ("ep_loso_os4", "ep_loso"), ("ep_zs_os4", "ep_zs")]
    table, diff = ci(methods, pairs, args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "ci.csv", index=False)
    diff.to_csv(args.out / "ci_paired.csv", index=False)
    print(table[["method", "seeds"] + [c for k in METRICS[:3] for c in (k, f"{k}_lo", f"{k}_hi")]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
