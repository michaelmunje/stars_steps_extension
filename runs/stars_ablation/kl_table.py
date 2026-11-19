"""KL trade-off table for L6 (seed 1): validation single-sample ADE (mean of the folds' best epochs) and test metrics
at 4.8 s (5-scene mean): single-sample ADE, most-likely ADE, KDE NLL, minADE20, collisions, and the sample spread
(per agent, RMS distance of the 20 samples to their mean at 4.8 s; median over agents, mean over scenes).

    python runs/stars_ablation/kl_table.py 0.03 0.08 0.15 0.3 0.5 1.0
"""
import json
import sys
from pathlib import Path

import numpy as np

from forecast_bench.ablation_report import fold_metrics
from forecast_bench.windows import FOLDS, ROOT, load

B = ROOT / "experiments/forecast_bench"
rows = ["| KL weight | val single-sample ADE | test single-sample ADE / FDE | most-likely ADE | KDE NLL | minADE20 | collisions @0.2 m | sample spread at 4.8 s (m) |",
        "|---|---|---|---|---|---|---|---|"]
for k in sys.argv[1:]:
    name = f"L6_kl{k}_s1"
    try:
        val = np.mean([json.loads((B / "ablation/loso" / name / f / "summary.json").read_text())["best_val_min_ade"] for f in FOLDS])
    except FileNotFoundError:
        continue
    m, spread = [], []
    for f in FOLDS:
        w = load(B / "windows" / f"ethucy_{f}.npz")
        p = np.load(B / "preds" / f"ethucy_{f}" / f"abl_{name}.npz")["pred"]
        mode = np.load(B / "preds" / f"ethucy_{f}" / f"abl_{name}_mode.npz")["pred"]
        m.append(fold_metrics(w, p, mode))
        last = p[:, :, -1]
        spread.append(np.median(np.sqrt(((last - last.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))))
    a = lambda key: np.mean([x[key] for x in m])
    rows.append(f"| {k} | {val:.3f} | {a('ss_ade@12'):.3f} / {a('ss_fde@12'):.3f} | {a('ml_ade@12'):.3f} | {a('nll@12'):.2f} | "
                f"{a('min_ade@12'):.3f} | {a('col@12'):.3f} | {np.mean(spread):.3f} |")
print("\n".join(rows))
