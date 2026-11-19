"""Pick a ladder rung's configuration on seed-1 validation (mean over the 5 folds of each fold's best validation
single-sample ADE), write chosen/<rung>.flags and link the chosen seed-1 run and predictions as <rung>_s1.

    python runs/stars_ablation/select_config.py L4 "lr3e-4=--lr 3e-4" "lr1e-3=--lr 1e-3"
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOSO = ROOT / "experiments/forecast_bench/ablation/loso"
PREDS = ROOT / "experiments/forecast_bench/preds"
FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
rung, options = sys.argv[1], dict(o.split("=", 1) for o in sys.argv[2:])
scores = {}
for name in options:
    vals = [json.loads((LOSO / f"{rung}_{name}_s1" / f / "summary.json").read_text())["best_val_min_ade"] for f in FOLDS]
    scores[name] = float(np.mean(vals))
best = min(scores, key=scores.get)
out = ROOT / "runs/stars_ablation/chosen"
out.mkdir(exist_ok=True)
(out / f"{rung}.flags").write_text(options[best] + "\n")
(out / f"{rung}.json").write_text(json.dumps({"chosen": best, "flags": options[best], "val_ss_ade": scores}, indent=1))
link = LOSO / f"{rung}_s1"
if link.is_symlink():
    link.unlink()
link.symlink_to(f"{rung}_{best}_s1")
for fold in FOLDS:
    for suffix in ("", "_mode"):
        src = PREDS / f"ethucy_{fold}" / f"abl_{rung}_{best}_s1{suffix}.npz"
        dst = src.with_name(f"abl_{rung}_s1{suffix}.npz")
        if src.exists():
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.name)
print(rung, scores, "->", best)
