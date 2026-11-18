"""Closed-loop ETH/UCY validation for checkpoint selection (never test): the realism simulator on the
pooled-split *validation* recording (students003), stride 10, as in simplenavsim/scripts/ethucy_benchmark.py.

Usage: python runs/stars_rollout_ft/cl_val.py --out FILE.json [--oversample M] [--seeds S] NAME=CKPT ... [cv]
Reports micro ADE/FDE at 4.8/8/12 s, collision pairs and HACS, and speed / acceleration / turning
(mean, W1 to the ground truth) at 8 s.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "simplenavsim" / "scripts")]

import ethucy_benchmark as B  # noqa: E402
from simplenavsim.realism import data as D  # noqa: E402
from simplenavsim.realism import sim as S  # noqa: E402

VAL = {"univ": ["students003"]}


def evaluate(fn, wins, seeds) -> dict:
    rows, kin = [], {}
    for seed in seeds:
        trajs, secs = B.simulate(fn, wins, seed)
        k = {}
        rows += B.score(trajs, wins, kin_out=k, seed=seed)
        for name, v in k.items():
            kin.setdefault(name, []).extend(v)
    out = {}
    for label in D.HORIZONS:
        sel = [r for r in rows if r["horizon_s"] == label]
        for metric in ("ade", "fde", "collision_pairs", "hacs"):
            out[f"{metric}@{label}"] = B.micro(sel, metric)
    out["kin"] = {q: np.concatenate(v) for q, v in kin.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="NAME=CHECKPOINT, or cv / orca_default / ground_truth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oversample", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    wins = D.benchmark_windows(recordings=VAL, stride=args.stride)
    gt_kin = {}
    B.score({w.scene_id: w.future for w in wins}, wins, kin_out=gt_kin)
    gt_kin = {q: np.concatenate(v) for q, v in gt_kin.items()}
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        if path:
            fn = S.Stars(Path(path), device=args.device)
            if args.oversample > 1:
                fn = fn.with_oversample(args.oversample)
            seeds = range(args.seeds)
        else:
            fn = {"cv": S.cv, "orca_default": S.orca}[name]
            seeds = [0]
        res = evaluate(fn, wins, seeds)
        kin = res.pop("kin")
        for q in ("speed", "acceleration", "turning"):
            res[f"{q}_mean@8.0"] = float(kin[q].mean())
            res[f"{q}_w1@8.0"] = B.wasserstein(kin[q], gt_kin[q])
        res["gt_speed_mean@8.0"] = float(gt_kin["speed"].mean())
        res.update(checkpoint=path, oversample=args.oversample if path else 1, seeds=len(seeds), windows=len(wins))
        results[f"{name}" + (f"_os{args.oversample}" if path and args.oversample > 1 else "")] = res
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
