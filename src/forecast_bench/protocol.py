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


HORIZONS = {1.2: 3, 2.4: 6, 3.6: 9, 4.8: 12}   # seconds -> prediction steps (0.4 s)
LOG_PDF_MIN = -20.0                           # Trajectron++'s clip of the KDE log density


def kde_nll(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """``pred [E, K, T, 2]``, ``gt [E, T, 2]`` -> NLL [E, T] of the truth under a Gaussian KDE of the K samples
    per agent and step (Scott's bandwidth, scipy.stats.gaussian_kde's rule), log density clipped at -20."""
    E, K, T, _ = pred.shape
    x = pred.transpose(0, 2, 1, 3).astype(np.float64)                     # [E, T, K, 2]
    centred = x - x.mean(2, keepdims=True)
    cov = np.einsum("etki,etkj->etij", centred, centred) / (K - 1) * K ** (-2 / 6) + 1e-8 * np.eye(2)
    inv, logdet = np.linalg.inv(cov), np.linalg.slogdet(cov)[1]
    d = gt[:, :, None].astype(np.float64) - x                              # [E, T, K, 2]
    maha = np.einsum("etki,etij,etkj->etk", d, inv, d)
    logk = -0.5 * maha - 0.5 * logdet[..., None] - np.log(2 * np.pi)
    m = logk.max(-1, keepdims=True)
    logpdf = (m + np.log(np.exp(logk - m).mean(-1, keepdims=True)))[..., 0]
    return -np.maximum(logpdf, LOG_PDF_MIN)


def primary_scores(windows: dict, pred: np.ndarray) -> dict[str, float]:
    """Single-sample ADE/FDE (mean over the K draws), KDE NLL (K > 1) and 0.2 m collision rate at each horizon."""
    from forecast_bench.metrics import _proximity, scored

    s = scored(windows)
    err = np.linalg.norm(pred - s["gt"][:, None], axis=-1)               # [E, K, T]
    nll = kde_nll(pred, s["gt"]) if pred.shape[1] > 1 else None
    out = {}
    for h, t in HORIZONS.items():
        out[f"ade@{h}"] = err[..., :t].mean()
        out[f"fde@{h}"] = err[..., t - 1].mean()
        out[f"nll@{h}"] = nll[:, :t].mean() if nll is not None else np.nan
        closest, _, _, multi = _proximity(pred[:, :, :t], s["window"], s["obs"])
        out[f"col20@{h}"] = (closest[multi] < 0.2).mean()
    return out


def primary(root: Path = BENCH, kinds=("zs", "loso", "scratch", "zs_rft", "loso_rft", "zs_ts", "loso_ts", "zs_tsf"),
            baselines=("socialvae_official", "moflow_official", "agentformer_official", "trajectron_official",
                       "sgan_official", "cv")) -> pd.DataFrame:
    """Per (row, fold, seed) primary metrics. Rows: ``ep_<kind>`` (sampled, mode from ``_mode``), ``ep_<kind>_os4``
    and the baselines (their own K draws; deterministic ones are their own mode)."""
    rows = []
    for fold in FOLDS:
        windows = load(root / "windows" / f"ethucy_{fold}.npz")
        d = root / "preds" / f"ethucy_{fold}"
        rows.append({"row": "ground_truth", "fold": fold, "seed": 0,
                     **{f"col20@{h}": v for h, v in _gt_collisions(windows).items()}})
        jobs = [(b, b, 0, b if b == "cv" else "_none") for b in baselines]   # no mode for sampled baselines
        for kind in kinds:
            for p in d.glob(f"ep_{kind}_s*.npz"):
                m = re.fullmatch(rf"ep_{kind}_s(\d+)(_os4)?", p.stem)
                if m:
                    jobs.append((f"ep_{kind}{m[2] or ''}", p.stem, int(m[1]), f"ep_{kind}_s{m[1]}_mode"))
        for row, stem, seed, mode in jobs:
            if not (d / f"{stem}.npz").exists():
                continue
            r = {"row": row, "fold": fold, "seed": seed,
                 **primary_scores(windows, np.load(d / f"{stem}.npz")["pred"].astype(np.float32))}
            if (d / f"{mode}.npz").exists():
                pm = primary_scores(windows, np.load(d / f"{mode}.npz")["pred"].astype(np.float32))
                r.update({f"mode_{k}": v for k, v in pm.items() if k[:3] in ("ade", "fde")})
            rows.append(r)
            print(fold, row, seed, f"{r['ade@4.8']:.3f}", flush=True)
    return pd.DataFrame(rows)


def _gt_collisions(windows: dict) -> dict[float, float]:
    from forecast_bench.metrics import _proximity, scored

    s = scored(windows)
    out = {}
    for h, t in HORIZONS.items():
        closest, _, _, multi = _proximity(s["gt"][:, None, :t], s["window"], s["obs"])
        out[h] = (closest[multi] < 0.2).mean()
    return out


def primary_table(df: pd.DataFrame) -> pd.DataFrame:
    """5-fold average per (row, seed), then mean and std over seeds."""
    avg = df.groupby(["row", "seed"]).mean(numeric_only=True).drop(columns=[], errors="ignore")
    full = df.groupby(["row", "seed"]).fold.nunique() == len(FOLDS)
    avg = avg[full]
    out = avg.groupby("row").agg(["mean", "std"])
    out[("seeds", "n")] = avg.groupby("row").size()
    return out


AGREED = [f"{m}@{h}" for h in HORIZONS for m in ("ade", "fde", "mode_ade", "mode_fde", "nll", "col20")]


def agreed_agent(windows: dict, path: Path) -> dict[str, np.ndarray]:
    """Per scored agent, one prediction file: single-sample ADE/FDE (mean over K), KDE NLL (mean over steps up to
    the horizon) and the share of samples within 0.2 m of another scored agent (NaN when alone) at each horizon;
    with the ``_mode`` file next to it, mode ADE/FDE."""
    from forecast_bench.metrics import _proximity, scored

    s = scored(windows)
    pred = np.load(path)["pred"].astype(np.float32)
    err = np.linalg.norm(pred - s["gt"][:, None], axis=-1)
    nll = kde_nll(pred, s["gt"]) if pred.shape[1] > 1 else None
    mode = path.with_name(path.stem + "_mode.npz")
    merr = np.linalg.norm(np.load(mode)["pred"][:, 0] - s["gt"], axis=-1) if mode.exists() else None
    out = {}
    for h, t in HORIZONS.items():
        out[f"ade@{h}"], out[f"fde@{h}"] = err[..., :t].mean((1, 2)), err[..., t - 1].mean(1)
        out[f"nll@{h}"] = nll[:, :t].mean(1) if nll is not None else np.full(len(err), np.nan)
        out[f"mode_ade@{h}"] = merr[:, :t].mean(1) if merr is not None else np.full(len(err), np.nan)
        out[f"mode_fde@{h}"] = merr[:, t - 1] if merr is not None else np.full(len(err), np.nan)
        closest, _, _, multi = _proximity(pred[:, :, :t], s["window"], s["obs"])
        out[f"col20@{h}"] = np.where(multi, (closest < 0.2).mean(1), np.nan)
    return out


def paired(comparisons: list[tuple[str, str, str]], root: Path = BENCH, seed: int = 0) -> pd.DataFrame:
    """``(name, a, b)`` with ``a``/``b`` = ``stem@s1,s2`` (e.g. ``ep_zs_ts@1,2,3``): per-agent metrics averaged over
    the listed seeds' prediction files, 5-fold mean of each, and the 95% block-bootstrap CI of a - b (same resampled
    blocks for both; per fold, metric = sum over agents / count over agents with a value)."""
    rng = np.random.default_rng(seed)
    cache, rows = {}, []
    folds = {}
    for fold in FOLDS:
        windows = load(root / "windows" / f"ethucy_{fold}.npz")
        blk = blocks(windows)
        nb = blk.max() + 1
        w_agent = blk[scored_window(windows)]
        weights = rng.multinomial(nb, np.full(nb, 1 / nb), size=REPS).astype(float)   # [R, nb]
        folds[fold] = (windows, w_agent, weights, nb)

    def spec(sp, fold):
        stem, seeds = sp.split("@")
        key = (sp, fold)
        if key not in cache:
            windows = folds[fold][0]
            per = [agreed_agent(windows, root / "preds" / f"ethucy_{fold}" /
                                (f"{stem}_s{x}.npz" if x else f"{stem}.npz")) for x in seeds.split(",")]
            cache[key] = {k: np.mean([p[k] for p in per], 0) for k in AGREED}
        return cache[key]

    for name, a, b in comparisons:
        row = {"comparison": name, "a": a, "b": b}
        for k in AGREED:
            pa, pb, diffs = [], [], []
            for fold in FOLDS:
                _, w_agent, weights, nb = folds[fold]
                xa, xb = spec(a, fold)[k], spec(b, fold)[k]
                ok = ~np.isnan(xa) & ~np.isnan(xb)
                if not ok.any():
                    break
                cnt = np.bincount(w_agent[ok], None, nb)
                sa, sb = np.bincount(w_agent[ok], xa[ok], nb), np.bincount(w_agent[ok], xb[ok], nb)
                pa.append(sa.sum() / cnt.sum()); pb.append(sb.sum() / cnt.sum())
                den = weights @ cnt
                diffs.append((weights @ sa - weights @ sb) / np.maximum(den, 1e-9))
            if len(pa) < len(FOLDS):
                continue
            d = np.mean(diffs, 0)
            row[f"{k}_a"], row[f"{k}_b"] = np.mean(pa), np.mean(pb)
            row[f"{k}_diff"] = np.mean(pa) - np.mean(pb)
            row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(row)
        print(name, f"ade@4.8 {row.get('ade@4.8_diff', np.nan):+.4f} [{row.get('ade@4.8_lo', np.nan):+.4f}, "
              f"{row.get('ade@4.8_hi', np.nan):+.4f}]", flush=True)
    return pd.DataFrame(rows)


def scored_window(windows: dict) -> np.ndarray:
    """Window index of every scored agent."""
    node_window = np.repeat(np.arange(len(windows["ptr"]) - 1), np.diff(windows["ptr"]))
    return node_window[windows["eval_mask"]]


TWO_STAGE = [("A' vs A (seeds 1-3)", "ep_zs_ts@1,2,3", "ep_zs@1,2,3"),
             ("B' vs B (seeds 1-3)", "ep_loso_ts@1,2,3", "ep_loso@1,2,3"),
             ("A' frozen vs A (seed 1)", "ep_zs_tsf@1", "ep_zs@1"),
             ("A' frozen vs A' fine-tuned (seed 1)", "ep_zs_tsf@1", "ep_zs_ts@1"),
             ("B vs A (seeds 1-3)", "ep_loso@1,2,3", "ep_zs@1,2,3"),
             ("B' vs A' (seeds 1-3)", "ep_loso_ts@1,2,3", "ep_zs_ts@1,2,3"),
             ("B vs LOSO from scratch (seeds 0-2 vs 1-3)", "ep_loso@1,2,3", "ep_scratch@0,1,2"),
             ("B' vs LOSO from scratch", "ep_loso_ts@1,2,3", "ep_scratch@0,1,2"),
             ("B vs SocialVAE", "ep_loso@1,2,3", "socialvae_official@"),
             ("B vs constant velocity", "ep_loso@1,2,3", "cv@")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "ci", "primary", "paired"])
    parser.add_argument("--root", type=Path, default=BENCH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--methods", default="socialvae_official,moflow_official,agentformer_official,trajectron_official,"
                        "sgan_official,cv,ep_zs,ep_zs_os4,ep_zs_mode,ep_loso,ep_loso_os4,ep_loso_mode,ep_scratch,"
                        "ep_scratch_os4,ep_scratch_mode,stars_noethucy,stars_loo")
    args = parser.parse_args()
    if args.stage == "select":
        select(args.root)
        return
    if args.stage == "paired":
        args.out.mkdir(parents=True, exist_ok=True)
        df = paired(TWO_STAGE, args.root)
        df.to_csv(args.out / "two_stage_paired.csv", index=False)
        return
    if args.stage == "primary":
        args.out.mkdir(parents=True, exist_ok=True)
        df = primary(args.root)
        df.to_csv(args.out / "primary_per_fold.csv", index=False)
        table = primary_table(df)
        table.to_csv(args.out / "primary.csv")
        print(table.round(3).to_string())
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
