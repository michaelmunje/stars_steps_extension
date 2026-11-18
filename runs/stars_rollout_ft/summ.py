"""Per-epoch validation summary of the rft_* runs: closed-loop (12 steps) ADE/FDE/speed/turn, open-loop min-of-20 ADE."""
import json
import sys
from pathlib import Path

RUNS = Path("/scratch/09865/mmunje/stars-forecasting/experiments/stars_ma/runs")
names = sys.argv[1:] or sorted(p.name for p in RUNS.glob("rft_*") if (p / "metrics.jsonl").exists())
keys = ["selection", "val_cl_ade", "val_cl_fde", "val_cl_speed", "val_cl_gt_speed", "val_cl_turn", "val_agent_ade",
        "val_cl_ucy/ade", "val_cl_eth/ade", "train_rft", "train_forecast"]
for name in names:
    print(f"== {name}")
    print("ep  " + " ".join(f"{k.replace('val_', '')[:12]:>12}" for k in keys))
    for line in open(RUNS / name / "metrics.jsonl"):
        d = json.loads(line)
        if "selection" in d:
            print(f"{d['epoch']:<3} " + " ".join(f"{d.get(k, float('nan')):12.4f}" for k in keys))
