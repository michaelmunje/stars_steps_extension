"""ETH/UCY closed-loop crowd-simulation realism benchmark: STARS vs CV, straight-to-goal, SFM and ORCA.

Every window's cohort (pedestrians present at the last observed step) is simulated jointly from 8
observed steps (3.2 s) for up to 12 s at 0.4 s, and scored at 2, 4, 4.8, 8 and 12 s against the
recording (``simplenavsim.realism``; protocol in ``docs/ethucy_benchmark.md``). Stages:

    python simplenavsim/scripts/ethucy_benchmark.py all --device cuda          # everything below, in order
    python simplenavsim/scripts/ethucy_benchmark.py calibrate --run results/<run_id>
    python simplenavsim/scripts/ethucy_benchmark.py run --run results/<run_id>
    python simplenavsim/scripts/ethucy_benchmark.py aggregate|plot|render|report --run results/<run_id>
    python simplenavsim/scripts/ethucy_benchmark.py smoke                       # tiny end-to-end check (CPU)

Rendering needs ffmpeg (system, or ``imageio_ffmpeg`` on PYTHONPATH).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from simplenavsim.realism import data as D
from simplenavsim.realism import metrics as M
from simplenavsim.realism import render as R
from simplenavsim.realism import sim as S

REPO = Path(__file__).resolve().parents[2]
# Adapter rows: name -> checkpoint it steers (inference-time goal-steering adapter, weight calibrated).
ADAPTERS = {"stars_steps_goal": "stars_steps", "stars_pooled_goal": "stars_pooled",
            "stars_noethucy_goal_adapter": "stars_noethucy_goal"}
STARS = ["stars_steps", "stars_pooled", "stars_steps_goal", "stars_pooled_goal",
         "stars_noethucy", "stars_noethucy_goal", "stars_noethucy_goal_adapter"]
BASELINES = ["cv", "goal_straight", "sfm", "orca", "sfm_default", "orca_default"]
METHODS = ["ground_truth", *STARS, *BASELINES]
STUDY_MODELS = ["ground_truth", *STARS, "cv", "goal_straight", "sfm", "orca"]   # preference export (schema v1.0)
LABEL = {"ground_truth": "Ground truth", "stars_steps": "STARS (STEPS-only, zero-shot)",
         "stars_pooled": "STARS (pooled, in-domain)", "stars_steps_goal": "STARS STEPS + goal adapter",
         "stars_pooled_goal": "STARS pooled + goal adapter",
         "stars_noethucy": "STARS (no ETH/UCY, zero-shot)", "stars_noethucy_goal": "STARS goal-conditioned (no ETH/UCY, zero-shot)",
         "stars_noethucy_goal_adapter": "STARS goal-conditioned + goal adapter", "cv": "Constant velocity", "goal_straight": "Straight to goal",
         "sfm": "Social force (calibrated)", "orca": "ORCA (calibrated)", "sfm_default": "Social force (defaults)",
         "orca_default": "ORCA (published defaults)"}
CAL_HORIZON = 20                       # calibrate on ADE at 8 s
GRIDS = {"sfm": {"ped_strength": [1.0, 2.1, 4.0], "ped_range": [0.2, 0.3, 0.5], "tau": [0.5, 1.0]},
         "orca": {"time_horizon": [1.0, 2.0, 5.0], "neighbor_dist": [3.0, 5.0, 10.0], "radius": [0.2, 0.3]},
         "goal_weight": [0.1, 0.2, 0.3, 0.5, 0.7]}
RATIOS = {"ade": ("ade_sum", "agents"), "fde": ("fde_sum", "agents"), "sde": ("sde_sum", "agents"),
          "nade": ("nade_sum", "nonlinear_agents"), "mde_goal": ("mde_goal_sum", "agents"),
          "goal_reached": ("goal_reached", "agents"), "mde_end": ("mde_end_sum", "agents"),
          "collision_pairs": ("colliding_pairs", "pairs"), "collision_pair_time": ("colliding_pair_steps", "pair_steps"),
          "agent_collision": ("agents_in_collision", "agents"), "hacs": ("hacs_sum", "agent_steps"),
          "sacs": ("sacs_sum", "agent_steps"), "min_separation": ("min_sep_sum", "sep_windows")}
BLOCK = 50                             # bootstrap block: 50 steps (20 s) of one recording
REPS = 2000


# ----------------------------------------------------------------------------- simulation


def simulators(cal: dict, device: str) -> dict:
    sfm_p, (orca_p, orca_r) = S.calibrated(cal.get("sfm"), "sfm"), S.calibrated(cal.get("orca"), "orca")
    sims = {"cv": S.cv, "goal_straight": S.goal_straight, "sfm_default": S.sfm,
            "sfm": lambda x, n, **k: S.sfm(x, n, params=sfm_p), "orca_default": S.orca,
            "orca": lambda x, n, **k: S.orca(x, n, params=orca_p, radius=orca_r)}
    for name, path in S.CHECKPOINTS.items():
        if path.exists():
            sims[name] = S.Stars(path, device=device)
    for name, base in ADAPTERS.items():
        if base in sims:
            sims[name] = sims[base].with_goal(cal.get("goal_weight", {}).get(base, 0.3))
    return sims


def chunks(wins, size: int = 32):
    """Windows grouped by recording and cohort size (padding stays small)."""
    order = sorted(wins, key=lambda w: (w.recording, len(w.agent_ids)))
    return [order[i:i + size] for i in range(0, len(order), size)]


def simulate(fn, wins, seed: int = 0) -> tuple[dict, float]:
    """``{scene_id: [N, steps, 2]}`` and wall time."""
    out, start = {}, time.perf_counter()
    for group in chunks(wins):
        traj = fn(S.pad(group), D.MAX_STEPS, seed=seed)
        for b, w in enumerate(group):
            out[w.scene_id] = traj[b, :len(w.agent_ids), :w.steps]
    return out, time.perf_counter() - start


def score(trajs: dict, wins, horizons=None, *, agents_out=None, kin_out=None, **tags) -> list[dict]:
    rows = []
    for w in wins:
        for label, h in (horizons or D.HORIZONS).items():
            if not w.scored(h).any():
                continue
            row, agents, kin = M.window_metrics(trajs[w.scene_id], w, h)
            sep = row.pop("min_separation")
            base = {**tags, "horizon_s": label, "scene_id": w.scene_id, "dataset": w.dataset, "recording": w.recording,
                    "start": w.start, "block": f"{w.recording}:{w.start // BLOCK}", "pooled_split": w.meta["pooled_split"]}
            rows.append({**base, **row, "min_sep_sum": 0.0 if np.isnan(sep) else sep, "sep_windows": int(not np.isnan(sep))})
            if agents_out is not None:
                agents_out += [{**base, **a} for a in agents]
            if kin_out is not None and label == 8.0:
                for k, v in kin.items():
                    kin_out.setdefault(k, []).append(v)
    return rows


def micro(rows, metric="ade") -> float:
    num, den = RATIOS[metric]
    frame = pd.DataFrame(rows)
    return float(frame[num].sum() / max(frame[den].sum(), 1))


def calibrate(args) -> dict:
    """Grid-search SFM, ORCA and the goal-adapter weight on the calibration recordings (ADE at 8 s)."""
    wins = D.benchmark_windows(recordings={"calibration": D.CALIBRATION}, stride=args.stride)[:args.max_windows or None]
    horizon = {8.0: CAL_HORIZON}
    log, best = [], {}
    for kind in ("sfm", "orca"):
        if args.append:
            break
        grid = GRIDS[kind]
        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())][:args.max_grid or None]
        results = []
        for combo in combos:
            fn = (lambda x, n, c=combo, **k: S.sfm(x, n, params=S.calibrated(c, "sfm"))) if kind == "sfm" else \
                 (lambda x, n, c=combo, **k: S.orca(x, n, params=S.calibrated(c, "orca")[0], radius=S.calibrated(c, "orca")[1]))
            rows = score(simulate(fn, wins)[0], wins, horizon)
            results.append((micro(rows), combo))
            log.append({"model": kind, **combo, "ade_8s": results[-1][0], "hacs_8s": micro(rows, "hacs")})
        best[kind] = min(results, key=lambda r: r[0])[1]
    best["goal_weight"] = {}
    for name in [ADAPTERS[m] for m in args.methods if m in ADAPTERS and S.CHECKPOINTS[ADAPTERS[m]].exists()]:
        model, results = S.Stars(S.CHECKPOINTS[name], device=args.device), []
        for weight in GRIDS["goal_weight"][:args.max_grid or None]:
            ade = np.mean([micro(score(simulate(model.with_goal(weight), wins, seed)[0], wins, horizon))
                           for seed in range(2)])
            results.append((ade, weight))
            log.append({"model": f"{name}_goal", "goal_weight": weight, "ade_8s": ade})
        best["goal_weight"][name] = min(results)[1]
    if args.append and (args.run / "calibration.json").exists():   # keep the earlier entries (same recordings)
        old = json.loads((args.run / "calibration.json").read_text())
        best = {**{k: old[k] for k in ("sfm", "orca")}, "goal_weight": {**old["goal_weight"], **best["goal_weight"]}}
        log = old["grid"] + [row for row in log if row["model"] not in ("sfm", "orca")]
    out = {"recordings": D.CALIBRATION, "windows": len(wins), "objective": "micro ADE at 8 s (seed 0; adapter: mean of seeds 0-1)",
           **best, "grid": log}
    (args.run / "calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))
    return out


def run(args) -> None:
    cal = json.loads((args.run / "calibration.json").read_text()) if (args.run / "calibration.json").exists() else {}
    wins = select_windows(args)
    sims = simulators(cal, args.device)
    tdir = args.run / "trajectories"
    tdir.mkdir(parents=True, exist_ok=True)
    rows, agents, runtime, kin = [], [], [], {}
    for method in args.methods:
        seeds = range(args.seeds) if method in STARS else [0]
        for seed in seeds:
            if method == "ground_truth":
                trajs, secs = {w.scene_id: w.future for w in wins}, 0.0
            else:
                trajs, secs = simulate(sims[method], wins, seed)
            np.savez_compressed(tdir / f"{method}_s{seed}.npz", **trajs)
            k = {}
            rows += score(trajs, wins, agents_out=agents, kin_out=k, method=method, seed=seed)
            for name, v in k.items():
                kin.setdefault(method, {}).setdefault(name, []).extend(v)
            steps = sum(len(w.agent_ids) * w.steps for w in wins)
            runtime.append({"method": method, "seed": seed, "seconds": secs, "windows": len(wins), "agent_steps": steps,
                            "ms_per_window": 1e3 * secs / len(wins), "us_per_agent_step": 1e6 * secs / steps,
                            "device": args.device if method.startswith("stars") else "cpu"})
            print(f"{method} seed {seed}: {secs:.1f} s, ADE@4.8s {micro([r for r in rows if r['method'] == method and r['seed'] == seed and r['horizon_s'] == 4.8]):.3f}",
                  flush=True)
    mdir = args.run / "metrics"
    mdir.mkdir(exist_ok=True)
    rows, agents, runtime = pd.DataFrame(rows), pd.DataFrame(agents), pd.DataFrame(runtime)
    if args.append:   # keep the other methods' results (same windows: checked through the scenario manifest)
        old = [json.loads(line)["init_hash"] for line in open(args.run / "scenario_manifest.jsonl")]
        assert old == [R.init_hash(w.agent_ids, w.obs, w.goal) for w in wins], "--append needs the same windows"
        keep = lambda f: f[~f.method.isin(args.methods)]
        rows = pd.concat([keep(pd.read_csv(mdir / "per_rollout.csv")), rows])
        agents = pd.concat([keep(pd.read_csv(mdir / "per_agent.csv.gz")), agents])
        runtime = pd.concat([keep(pd.read_csv(mdir / "runtime.csv")), runtime])
        previous = np.load(mdir / "kinematics_8s.npz")
        for key in previous.files:
            if key.split("/")[0] not in args.methods:
                kin.setdefault(key.split("/")[0], {})[key.split("/")[1]] = [previous[key]]
    rows.to_csv(mdir / "per_rollout.csv", index=False)
    agents.to_csv(mdir / "per_agent.csv.gz", index=False)
    runtime.to_csv(mdir / "runtime.csv", index=False)
    pd.DataFrame(D.horizon_counts(wins)).to_csv(mdir / "counts.csv", index=False)
    np.savez_compressed(mdir / "kinematics_8s.npz", **{f"{m}/{k}": np.concatenate(v) for m, d in kin.items() for k, v in d.items()})
    manifest = [{"scene_id": w.scene_id, "dataset": w.dataset, "recording": w.recording, "start_step": w.start,
                 "t0_step": w.t0, "steps": w.steps, "cohort": w.agent_ids.tolist(), "pooled_split": w.meta["pooled_split"],
                 "scored": {str(lab): w.agent_ids[w.scored(h)].tolist() for lab, h in D.HORIZONS.items()},
                 "goal_at_boundary": w.agent_ids[w.goal_at_boundary].tolist(),
                 "init_hash": R.init_hash(w.agent_ids, w.obs, w.goal)} for w in wins]
    with open(args.run / "scenario_manifest.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in manifest)


def select_windows(args):
    wins = D.benchmark_windows(scenes=args.scenes, stride=args.stride)
    if args.max_windows:
        keep, seen = [], {}
        for w in wins:
            seen[w.recording] = seen.get(w.recording, 0) + 1
            if seen[w.recording] <= args.max_windows:
                keep.append(w)
        wins = keep
    return wins


# ----------------------------------------------------------------------------- statistics


def _ratio(sums, metric):
    num, den = RATIOS[metric]
    return sums[num] / np.where(sums[den] > 0, sums[den], np.nan)


def aggregate(args) -> None:
    mdir = args.run / "metrics"
    raw = pd.read_csv(mdir / "per_rollout.csv")
    cols = sorted({c for pair in RATIOS.values() for c in pair})
    keys = ["method", "horizon_s", "scene_id", "dataset", "recording", "block", "pooled_split"]
    per_window = raw.groupby(keys, as_index=False)[cols].mean()          # expected value over STARS seeds
    rng = np.random.default_rng(0)
    agg, scene, paired = [], [], []
    for horizon, frame in per_window.groupby("horizon_s"):
        blocks = frame[frame.method == "ground_truth"][["dataset", "block"]].drop_duplicates()
        datasets = list(dict.fromkeys(blocks.dataset))
        draws = {d: rng.integers(0, (blocks.dataset == d).sum(), (REPS, (blocks.dataset == d).sum())) for d in datasets}
        stats = {}
        for method, mf in frame.groupby("method"):
            per_d, boot_d = {}, {}
            for d in datasets:
                bnames = blocks[blocks.dataset == d].block.tolist()
                bs = mf[mf.dataset == d].groupby("block")[cols].sum().reindex(bnames, fill_value=0)
                per_d[d] = bs.sum()
                arr = bs.to_numpy()
                boot_d[d] = pd.DataFrame(arr[draws[d]].sum(1), columns=cols)
            groups = {"micro": datasets, "heldout_eth_zara2": [d for d in ("eth", "zara2") if d in datasets]}
            stats[method] = {}
            for metric in RATIOS:
                point_d = {d: float(_ratio(per_d[d], metric)) for d in datasets}
                boot_dm = {d: _ratio(boot_d[d], metric).to_numpy() for d in datasets}
                for d in datasets:
                    scene.append({"method": method, "horizon_s": horizon, "dataset": d, "metric": metric, "value": point_d[d],
                                  **_ci(boot_dm[d]), "windows": int((mf.dataset == d).sum()),
                                  "agents": int(mf[mf.dataset == d].agents.sum()),
                                  "pooled_exposure": "/".join(sorted(set(mf[mf.dataset == d].pooled_split)))})
                values = {"macro": (np.mean(list(point_d.values())), np.nanmean(np.stack(list(boot_dm.values())), 0))}
                for name, ds in groups.items():
                    if ds:
                        tot = sum(per_d[d] for d in ds)
                        values[name] = (float(_ratio(tot, metric)), _ratio(sum(boot_d[d] for d in ds), metric).to_numpy())
                for level, (point, boot) in values.items():
                    stats[method][(metric, level)] = boot
                    agg.append({"method": method, "horizon_s": horizon, "level": level, "metric": metric, "value": point,
                                **_ci(boot), "windows": int(len(mf)), "agents": int(mf.agents.sum())})
        for a, b in comparisons(stats):
            for (metric, level) in stats[a]:
                if metric in ("ade", "fde", "collision_pair_time", "hacs", "sacs", "mde_goal") and level in ("micro", "macro"):
                    diff = stats[a][(metric, level)] - stats[b][(metric, level)]
                    point = next(r["value"] for r in agg[::-1] if r["method"] == a and r["horizon_s"] == horizon and r["metric"] == metric and r["level"] == level) \
                        - next(r["value"] for r in agg[::-1] if r["method"] == b and r["horizon_s"] == horizon and r["metric"] == metric and r["level"] == level)
                    paired.append({"horizon_s": horizon, "level": level, "metric": metric, "a": a, "b": b, "a_minus_b": point,
                                   **_ci(diff), "share_a_lower": float(np.nanmean(diff < 0))})
    pd.DataFrame(agg).to_csv(mdir / "aggregate.csv", index=False)
    pd.DataFrame(scene).to_csv(mdir / "per_scene.csv", index=False)
    pd.DataFrame(paired).to_csv(mdir / "paired.csv", index=False)
    kinematics_table(args)
    print(pd.DataFrame(agg).query("level == 'micro' and metric == 'ade'").pivot(index="method", columns="horizon_s", values="value").round(3))


def comparisons(stats):
    have = set(stats)
    pairs = [(s, b) for s in STARS for b in ("cv", "goal_straight", "sfm", "orca")]
    pairs += [("stars_steps_goal", "stars_steps"), ("stars_pooled_goal", "stars_pooled"), ("stars_pooled", "stars_steps"),
              ("stars_noethucy_goal", "stars_noethucy"), ("stars_noethucy", "stars_steps"),
              ("stars_noethucy_goal", "stars_steps_goal"), ("stars_noethucy_goal_adapter", "stars_noethucy_goal")]
    return [(a, b) for a, b in pairs if a in have and b in have]


def _ci(boot):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return {"ci_low": np.nan, "ci_high": np.nan}
    return {"ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))}


def wasserstein(a, b, n: int = 512) -> float:
    q = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))) if len(a) and len(b) else np.nan


def kinematics_table(args) -> None:
    kin = np.load(args.run / "metrics" / "kinematics_8s.npz")
    methods = sorted({k.split("/")[0] for k in kin.files}, key=METHODS.index)
    rows = []
    for m in methods:
        row = {"method": m}
        for q in ("speed", "acceleration", "turning"):
            v = kin[f"{m}/{q}"]
            row.update({f"{q}_mean": float(v.mean()), f"{q}_p90": float(np.quantile(v, 0.9)) if len(v) else np.nan,
                        f"{q}_w1_vs_gt": wasserstein(v, kin[f"ground_truth/{q}"])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.run / "metrics" / "kinematics.csv", index=False)


# ----------------------------------------------------------------------------- plots


COLORS = {"ground_truth": "#222222", "stars_steps": "#2b6cb0", "stars_pooled": "#6b46c1", "stars_steps_goal": "#63a4e0",
          "stars_pooled_goal": "#a78bdb",
          "stars_noethucy": "#319795", "stars_noethucy_goal": "#b83280", "stars_noethucy_goal_adapter": "#ed64a6", "cv": "#c05621", "goal_straight": "#d69e2e", "sfm": "#2f855a", "orca": "#c53030",
          "sfm_default": "#9ae6b4", "orca_default": "#feb2b2"}


def plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = args.run / "plots"
    pdir.mkdir(exist_ok=True)
    agg = pd.read_csv(args.run / "metrics" / "aggregate.csv")
    scene = pd.read_csv(args.run / "metrics" / "per_scene.csv")

    def lines(ax, metric, level="micro", methods=None, title=""):
        for m in methods or [m for m in METHODS if m in set(agg.method)]:
            f = agg[(agg.method == m) & (agg.metric == metric) & (agg.level == level)].sort_values("horizon_s")
            if f.empty:
                continue
            ax.plot(f.horizon_s, f.value, "o-", color=COLORS[m], label=LABEL[m], lw=1.6, ms=3,
                    ls="--" if m.endswith("_default") else "-")
            ax.fill_between(f.horizon_s, f.ci_low, f.ci_high, color=COLORS[m], alpha=0.12, lw=0)
        ax.set_xlabel("horizon (s)"), ax.set_title(title, fontsize=10), ax.grid(alpha=0.3)

    sims = [m for m in METHODS if m != "ground_truth" and m in set(agg.method)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lines(axes[0], "ade", methods=sims, title="ADE (m), micro average, 95% block-bootstrap CI")
    lines(axes[1], "fde", methods=sims, title="FDE (m)")
    axes[0].set_ylabel("metres"), axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(), fig.savefig(pdir / "ade_fde_vs_horizon.png", dpi=150), plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    lines(axes[0], "collision_pair_time", title="Colliding pair-steps / pair-steps (d <= 0.4 m)")
    lines(axes[1], "collision_pairs", title="Colliding pairs / pairs")
    lines(axes[2], "hacs", title="hACS (Mahzoon et al. 2025)")
    axes[2].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(pdir / "collision_vs_horizon.png", dpi=150), plt.close(fig)

    kin = np.load(args.run / "metrics" / "kinematics_8s.npz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (q, rng, unit) in zip(axes, [("speed", (0, 3), "m/s"), ("acceleration", (0, 6), "m/s²"), ("turning", (0, 180), "deg/s")]):
        for m in [m for m in METHODS if f"{m}/{q}" in kin.files and not m.endswith("_default")]:
            ax.hist(np.clip(kin[f"{m}/{q}"], *rng), bins=40, range=rng, density=True, histtype="step", color=COLORS[m],
                    lw=2.4 if m == "ground_truth" else 1.3, label=LABEL[m])
        ax.set_xlabel(f"{q} ({unit})"), ax.set_title(f"{q} distribution, 8 s horizon", fontsize=10), ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(pdir / "speed_distributions.png", dpi=150), plt.close(fig)

    for horizon in (4.8, 8.0):
        f = scene[(scene.metric == "ade") & (scene.horizon_s == horizon) & scene.method.isin(sims)]
        datasets = list(dict.fromkeys(f.dataset))
        fig, ax = plt.subplots(figsize=(12, 4.2))
        width = 0.8 / len(sims)
        for i, m in enumerate(sims):
            g = f[f.method == m].set_index("dataset").reindex(datasets)
            xs = np.arange(len(datasets)) + (i - len(sims) / 2 + 0.5) * width
            ax.bar(xs, g.value, width, color=COLORS[m], label=LABEL[m],
                   yerr=[g.value - g.ci_low, g.ci_high - g.value], error_kw={"lw": 0.7})
        exposure = f[f.method == "stars_pooled"].set_index("dataset").reindex(datasets).pooled_exposure.fillna("")
        ax.set_xticks(range(len(datasets)), [f"{d}\n(pooled: {e})" for d, e in zip(datasets, exposure)])
        ax.set_ylabel("ADE (m)"), ax.set_title(f"ADE per dataset at {horizon:g} s (95% CI)", fontsize=10)
        ax.legend(fontsize=7, ncol=3), ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(), fig.savefig(pdir / f"dataset_comparison_{horizon:g}s.png", dpi=150), plt.close(fig)


# ----------------------------------------------------------------------------- media + preference export


MEDIA_HORIZON = 8.0


def choose_scenarios(args, wins, count: int):
    """Deterministic random scenarios: per dataset, seeded draws among windows with >= 4 agents scored at 8 s
    and >= 60% of the cohort scored (so hidden context agents are few)."""
    h, rng, chosen = D.HORIZONS[MEDIA_HORIZON], np.random.default_rng(args.media_seed), []
    per = max(1, count // len({w.dataset for w in wins}))
    for d in dict.fromkeys(w.dataset for w in wins):
        pool = [w for w in wins if w.dataset == d and w.scored(h).sum() >= 4 and w.scored(h).mean() >= 0.6]
        pool = pool or [w for w in wins if w.dataset == d and w.scored(h).sum() >= 2]
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), min(per, len(pool)), replace=False))]
    return chosen


def panel(w, traj, h):
    s = w.scored(h)
    return np.concatenate([w.obs[s], traj[s, :h]], 1)


def render(args) -> None:
    wins = {w.scene_id: w for w in select_windows(args)}
    h = D.HORIZONS[MEDIA_HORIZON]
    chosen = choose_scenarios(args, list(wins.values()), args.scenarios)
    tdir = args.run / "trajectories"
    load = lambda m: np.load(tdir / f"{m}_s0.npz")
    trajs = {m: load(m) for m in STUDY_MODELS if (tdir / f"{m}_s0.npz").exists()}
    media, pref = args.run / "media", args.run / "preference_export"
    for sub in ("gifs", "mp4", "comparison_grids", "case_studies"):
        (media / sub).mkdir(parents=True, exist_ok=True)
    (pref / "clips").mkdir(parents=True, exist_ok=True)
    cfg = R.render_config()
    scenarios, index = [], []
    for w in chosen:
        paths = {m: panel(w, t[w.scene_id], h) for m, t in trajs.items()}
        view = R.bounds(paths["ground_truth"])
        sid = scenario_id(w)
        ih = R.init_hash(w.agent_ids[w.scored(h)], w.obs[w.scored(h)], w.goal[w.scored(h)])
        clips = []
        for m, p in paths.items():
            cid = "c_" + hashlib.sha256(f"{args.run.name}/{sid}/{m}/0".encode()).hexdigest()[:6]
            R.animate([p], view, MEDIA_HORIZON, pref / "clips" / f"{cid}.mp4", pref / "clips" / f"{cid}.gif")
            clips.append({"clip_id": cid, "model": m, "seed": 0, "mp4": f"clips/{cid}.mp4", "gif": f"clips/{cid}.gif",
                          "duration_s": MEDIA_HORIZON, "fps": cfg["fps"], "init_hash": ih,
                          "render_config_hash": cfg["render_config_hash"]})
        models = list(paths)
        R.animate([paths[m] for m in models], view, MEDIA_HORIZON, media / "comparison_grids" / f"{sid}.mp4",
                  media / "comparison_grids" / f"{sid}.gif", titles=[LABEL[m] for m in models], shape=(3, 3),
                  size=(1440, 1080), suptitle=f"{sid}: same initial state, {MEDIA_HORIZON:g} s closed loop (seed 0)")
        pair = [m for m in ("ground_truth", "stars_steps", "stars_pooled") if m in paths]
        R.animate([paths[m] for m in pair], view, MEDIA_HORIZON, media / "mp4" / f"{sid}_real_vs_stars.mp4",
                  media / "gifs" / f"{sid}_real_vs_stars.gif", titles=[LABEL[m] for m in pair], shape=(1, len(pair)),
                  size=(480 * len(pair), 420))
        rng = rng_window(w)
        scenarios.append({"scenario_id": sid, "dataset": w.dataset, "recording": w.recording, "start_frame": int(w.t0),
                          "start_time_s": round(w.t0 * D.DT, 3), "num_pedestrians": int(w.scored(h).sum()),
                          "horizon_s": MEDIA_HORIZON, "sim_dt": D.DT,
                          "goal_rule": "last observed position in the continuous recording (oracle)", "bounds_m": view,
                          "init_hash": ih, "counterfactual": None, "clips": clips})
        index.append({"scenario_id": sid, "scene_id": w.scene_id, "selection": "random (seeded)", **rng})
    # Case studies (labelled, kept apart from the random examples): worst STARS-STEPS ADE at 8 s.
    per = pd.read_csv(args.run / "metrics" / "per_rollout.csv").query("method == 'stars_steps' and horizon_s == @MEDIA_HORIZON and agents >= 3")
    worst = per.groupby("scene_id").apply(lambda f: f.ade_sum.sum() / f.agents.sum(), include_groups=False).sort_values()[::-1]
    for scene_id in worst.index[:args.case_studies]:
        w = wins[scene_id]
        paths = {m: panel(w, t[w.scene_id], h) for m, t in trajs.items()}
        sid = scenario_id(w)
        R.animate([paths[m] for m in paths], R.bounds(paths["ground_truth"]), MEDIA_HORIZON,
                  media / "case_studies" / f"{sid}.mp4", media / "case_studies" / f"{sid}.gif",
                  titles=[LABEL[m] for m in paths], shape=(3, 3), size=(1440, 1080),
                  suptitle=f"Case study (largest STARS-STEPS ADE at 8 s: {worst[scene_id]:.2f} m): {sid}")
        index.append({"scenario_id": sid, "scene_id": scene_id, "selection": "case study: worst stars_steps ADE at 8 s",
                      "stars_steps_ade_8s": float(worst[scene_id])})
    manifest = {"schema_version": "1.0", "benchmark_run_id": args.run.name,
                "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "study_type": "factual",
                "render": cfg, "scenarios": scenarios}
    (pref / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (media / "index.json").write_text(json.dumps({"render_style": R.STYLE, "scenarios": index}, indent=1))
    problems = validate_manifest(pref / "manifest.json")
    print(f"{len(scenarios)} scenarios, {sum(len(s['clips']) for s in scenarios)} clips; manifest problems: {problems or 'none'}")


def rng_window(w):
    return {"recording": w.recording, "start_step": int(w.start)}


def scenario_id(w) -> str:
    rec = "" if len(D.SCENES[w.dataset]) == 1 else w.recording.replace("students", "s") + "-"
    return f"{w.dataset}-{rec}{w.t0:04d}"


def validate_manifest(path: Path) -> list[str]:
    """Problems with a preference manifest against schema v1.0 (empty = valid)."""
    m, root, problems = json.loads(Path(path).read_text()), Path(path).parent, []
    for key in ("schema_version", "benchmark_run_id", "created_utc", "study_type", "render", "scenarios"):
        if key not in m:
            problems.append(f"missing {key}")
    if m.get("schema_version") != "1.0" or m.get("study_type") not in ("factual", "counterfactual"):
        problems.append("bad schema_version / study_type")
    render = m.get("render", {})
    for key in ("width", "height", "fps", "trail_s", "show_goals", "show_ids", "background", "render_config_hash"):
        if key not in render:
            problems.append(f"render.{key} missing")
    ids, clip_ids = set(), set()
    for s in m.get("scenarios", []):
        for key in ("scenario_id", "dataset", "recording", "start_frame", "start_time_s", "num_pedestrians", "horizon_s",
                    "sim_dt", "goal_rule", "bounds_m", "init_hash", "counterfactual", "clips"):
            if key not in s:
                problems.append(f"{s.get('scenario_id')}: missing {key}")
        if s.get("dataset") not in D.SCENES or s.get("scenario_id") in ids or len(s.get("bounds_m", [])) != 4:
            problems.append(f"{s.get('scenario_id')}: bad dataset / duplicate id / bounds")
        ids.add(s.get("scenario_id"))
        for c in s.get("clips", []):
            if c.get("model") not in STUDY_MODELS or c.get("clip_id") in clip_ids:
                problems.append(f"{c.get('clip_id')}: bad model or duplicate clip_id")
            clip_ids.add(c.get("clip_id"))
            if c.get("init_hash") != s.get("init_hash") or c.get("render_config_hash") != render.get("render_config_hash"):
                problems.append(f"{c.get('clip_id')}: hash mismatch")
            for kind in ("mp4", "gif"):
                rel = c.get(kind, "")
                if rel != f"clips/{c.get('clip_id')}.{kind}" or not (root / rel).is_file():
                    problems.append(f"{c.get('clip_id')}: {kind} path missing or not named by clip_id")
                if any(name in rel for name in METHODS):
                    problems.append(f"{c.get('clip_id')}: model name in file name")
    return problems


# ----------------------------------------------------------------------------- provenance + report


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def git(*cmd) -> str:
    return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True).stdout.strip()


def provenance(args) -> None:
    args.run.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("func",)}
    config.update({"horizons_s": list(D.HORIZONS), "obs_steps": D.OBS, "dt": D.DT, "max_steps": D.MAX_STEPS,
                   "scenes": D.SCENES, "calibration_recordings": D.CALIBRATION, "collision_radius_m": M.R,
                   "nonlinear_threshold_m": M.NONLINEAR, "goal_reached_m": M.REACHED, "bootstrap": {"block_steps": BLOCK, "reps": REPS},
                   "calibration_grids": GRIDS, "sim_radius_m": S.RADIUS, "arrival_m": S.ARRIVAL, "substeps": S.SUBSTEPS})
    (args.run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = {"python": sys.version, "platform": platform.platform(), "host": platform.node(), "torch": torch.__version__,
           "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "ffmpeg": shutil.which("ffmpeg") or str(R.matplotlib.rcParams["animation.ffmpeg_path"])}
    (args.run / "environment.json").write_text(json.dumps(env, indent=1))
    files = [REPO / "datasets" / D.SOURCE.get(n, "ucy") / "converted" / f"{n}.traj.parquet"
             for n in [*sum(D.SCENES.values(), []), *D.CALIBRATION]]
    meta = {"run_id": args.run.name, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "src", "simplenavsim")), "command": sys.argv,
            "checkpoints": {k: {"path": str(v.relative_to(REPO)), "sha256": sha256(v)} for k, v in S.CHECKPOINTS.items()
                            if v.exists()},
            "data": {str(f.relative_to(REPO)): sha256(f) for f in files},
            "pooled_checkpoint_exposure": D.POOLED_SPLIT}
    (args.run / "metadata.json").write_text(json.dumps(meta, indent=1))


def md(frame) -> str:
    rows = [list(map(str, frame.columns))] + [[f"{v:.4g}" if isinstance(v, float) else str(v) for v in r] for r in frame.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def fmt(r):
    return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"


def report(args) -> None:
    mdir = args.run / "metrics"
    agg, scene = pd.read_csv(mdir / "aggregate.csv"), pd.read_csv(mdir / "per_scene.csv")
    paired, counts = pd.read_csv(mdir / "paired.csv"), pd.read_csv(mdir / "counts.csv")
    kin, runtime = pd.read_csv(mdir / "kinematics.csv"), pd.read_csv(mdir / "runtime.csv")
    cal = json.loads((args.run / "calibration.json").read_text()) if (args.run / "calibration.json").exists() else {}
    methods = [m for m in METHODS if m in set(agg.method)]

    def table(metric, level, rows=methods):
        f = agg[(agg.metric == metric) & (agg.level == level)]
        hs = sorted(f.horizon_s.unique())
        lines = ["| method | " + " | ".join(f"{h:g} s" for h in hs) + " |", "|---|" + "---|" * len(hs)]
        for m in rows:
            g = f[f.method == m].set_index("horizon_s")
            lines.append(f"| {LABEL[m]} | " + " | ".join(fmt(g.loc[h]) if h in g.index else "" for h in hs) + " |")
        return "\n".join(lines)

    sims = [m for m in methods if m != "ground_truth"]
    c = counts.groupby("horizon_s")[["windows", "cohort_agents", "scored_agents", "excluded_short_history",
                                      "excluded_track_ends_by_horizon", "scored_goal_at_boundary", "late_arrivals_not_simulated"]].sum()
    best = agg[(agg.metric == "ade") & (agg.level == "micro") & agg.method.isin(sims)].sort_values("value").groupby("horizon_s").first()
    win = paired[(paired.metric == "ade") & (paired.level == "micro") & paired.b.isin(["cv", "goal_straight", "sfm", "orca"])]
    sig_better = win[win.ci_high < 0]
    out = [f"# ETH/UCY closed-loop realism benchmark: `{args.run.name}`", "",
           "Generated by `simplenavsim/scripts/ethucy_benchmark.py report`. Protocol: `docs/ethucy_benchmark.md`. "
           "All averages are 95% blocked-bootstrap CIs (blocks = 20 s of one recording, resampled within dataset, "
           f"{REPS} reps); STARS rows are expected values over {int(runtime[runtime.method == 'stars_steps'].seed.nunique() or 0)} seeds "
           "(not best-of-K). **stars_pooled is in-domain, not zero-shot**: it trained on hotel, zara1 and students001 and "
           "used students003 for checkpoint selection; only eth and zara2 are held out (`heldout_eth_zara2` rows). "
           "The `stars_steps_goal`, `stars_pooled_goal` and `*_adapter` rows use an inference-time goal-steering adapter. "
           "`stars_noethucy` and `stars_noethucy_goal` trained without any ETH or UCY data (zero-shot on every scene); "
           "`stars_noethucy_goal` is goal-conditioned (the oracle goal is a model input).", "",
           "## Headline", ""]
    for h, r in best.iterrows():
        out.append(f"- {h:g} s: lowest micro ADE is {LABEL[r.method]} ({fmt(r)} m).")
    out.append(f"- Paired micro-ADE comparisons where a STARS variant is significantly lower than a baseline (CI of the "
               f"difference < 0): {len(sig_better)} of {len(win)}" + (": " + ", ".join(
                   f"{a} < {b} at {h:g} s" for a, b, h in sig_better[["a", "b", "horizon_s"]].itertuples(index=False)) if len(sig_better) else "."))
    out += ["", "## Windows and agents per horizon", "", md(c.reset_index()), "",
            "Windows start every 10 steps (4 s); one agent can be scored in several overlapping windows (the block "
            "bootstrap accounts for this). Context agents (short history or track ending before the horizon) are simulated "
            "but not scored. Late arrivals are not simulated (fixed cohort).", "",
            "## ADE (m), micro average over agents", "", table("ade", "micro"), "",
            "## FDE (m), micro", "", table("fde", "micro"), "",
            "## ADE (m), macro average over the five datasets", "", table("ade", "macro"), "",
            "## ADE (m), held-out recordings of the pooled checkpoint (eth + zara2)", "", table("ade", "heldout_eth_zara2"), "",
            "## Squared displacement error (m², mean over steps), micro", "", table("sde", "micro"), "",
            "## nADE (m, non-linear agents), micro", "", table("nade", "micro"), "",
            "## Colliding pair-steps / pair-steps (d ≤ 0.4 m), micro", "", table("collision_pair_time", "micro"), "",
            "## Colliding pairs / pairs, micro", "", table("collision_pairs", "micro"), "",
            "## hACS / sACS, micro", "", table("hacs", "micro"), "", table("sacs", "micro"), "",
            "## Minimum separation (m, mean over windows of the window minimum)", "", table("min_separation", "micro"), "",
            "## Minimum distance to the oracle goal (m) and goal reached within 0.5 m", "", table("mde_goal", "micro"), "",
            table("goal_reached", "micro"), "",
            "## Per-dataset ADE at 4.8 s", ""]
    f = scene[(scene.metric == "ade") & (scene.horizon_s == 4.8)]
    ds = list(dict.fromkeys(f.dataset))
    exp = f[f.method == methods[-1]].set_index("dataset").pooled_exposure.reindex(ds).fillna("")
    out += ["| method | " + " | ".join(f"{d} (pooled: {exp[d]})" for d in ds) + " |", "|---|" + "---|" * len(ds)]
    for m in methods:
        g = f[f.method == m].set_index("dataset")
        out.append(f"| {LABEL[m]} | " + " | ".join(fmt(g.loc[d]) if d in g.index else "" for d in ds) + " |")
    out += ["", "## Paired comparisons (micro ADE, a − b, 95% CI)", "",
            md(paired[(paired.metric == "ade") & (paired.level == "micro")][["horizon_s", "a", "b", "a_minus_b", "ci_low", "ci_high"]]
               .round(3)), "",
            "## Kinematics at 8 s (W1 = Wasserstein-1 distance to the ground-truth distribution)", "",
            md(kin.round(3)), "",
            "## Runtime", "", md(runtime.round(3)), "",
            "## Calibration (classical baselines and goal adapter; calibration recordings only)", "",
            "```json", json.dumps({k: v for k, v in cal.items() if k != "grid"}, indent=1), "```", "",
            "## Files", "", "- `metrics/{per_rollout.csv, per_agent.csv.gz, per_scene.csv, aggregate.csv, paired.csv, "
            "kinematics.csv, runtime.csv, counts.csv}`", "- `plots/`, `media/` (`index.json` lists the scenarios), "
            "`preference_export/manifest.json` (schema v1.0), `trajectories/<method>_s<seed>.npz` (keyed by scene id)",
            "- `scenario_manifest.jsonl`: every window, its cohort, scored agents per horizon and init hash"]
    (args.run / "report.md").write_text("\n".join(out) + "\n")


# ----------------------------------------------------------------------------- CLI


def smoke(args) -> None:
    """Tiny end-to-end run: one dataset, few windows, two seeds, coarse calibration, one scenario."""
    args.run = Path(args.run or tempfile.mkdtemp(prefix="ethucy_smoke_"))
    args.scenes, args.max_windows, args.seeds, args.max_grid, args.scenarios, args.case_studies = ["zara1"], 6, 2, 1, 1, 0
    for stage in (provenance, calibrate, run, aggregate, plot, render, report):
        stage(args)
    print(f"smoke ok: {args.run}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("stage", choices=["all", "calibrate", "run", "aggregate", "plot", "render", "report", "smoke"])
    parser.add_argument("--run", type=Path, default=None, help="results/<run_id> (default: results/ethucy_<date>_<commit>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device for STARS")
    parser.add_argument("--scenes", nargs="+", choices=list(D.SCENES), default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", type=int, default=5, help="STARS seeds (expected value over them)")
    parser.add_argument("--stride", type=int, default=10, help="window start stride (steps)")
    parser.add_argument("--max-windows", type=int, default=0, help="per recording (0 = all)")
    parser.add_argument("--max-grid", type=int, default=0, help="calibration grid points per model (0 = all)")
    parser.add_argument("--scenarios", type=int, default=10, help="random media / preference scenarios")
    parser.add_argument("--case-studies", type=int, default=3)
    parser.add_argument("--media-seed", type=int, default=2026)
    parser.add_argument("--append", action="store_true",
                        help="calibrate/run: add --methods to an existing --run (other methods' rows and calibration kept)")
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        return smoke(args)
    if args.run is None:
        args.run = REPO / "results" / f"ethucy_{dt.date.today().isoformat()}_{git('rev-parse', '--short', 'HEAD')}"
    stages = {"calibrate": [calibrate], "run": [run], "aggregate": [aggregate], "plot": [plot], "render": [render],
              "report": [report], "all": [provenance, calibrate, run, aggregate, plot, render, report]}[args.stage]
    if args.stage in ("calibrate", "run"):
        stages = [provenance, *stages]
    for stage in stages:
        with torch.no_grad():
            stage(args)


if __name__ == "__main__":
    main()
