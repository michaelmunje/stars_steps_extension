"""Aggregate `evaluation.json` results from independently-launched multi-seed runs.

Combines per-seed evaluation results into mean/std/standard-error summaries for
every numeric metric present. Run directories missing `evaluation.json` raise
rather than being silently dropped, since a missing seed changes `n_seeds`
without any other diagnostic (see `aggregate_transfer.py` for the matching
pattern used for transfer evaluations).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from baselines.status import atomic_json


def aggregate(run_dirs: list[Path]) -> dict:
    rows = []
    for directory in run_dirs:
        path = directory / "evaluation.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation result: {path}")
        result = json.loads(path.read_text())
        config = json.loads((directory / "config.json").read_text())
        rows.append({"run_dir": str(directory), **result, "config": config})
    if not rows:
        raise ValueError("At least one evaluation directory is required")

    keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"k", "n_predictions"}
    })
    summary = {"n_seeds": len(rows), "runs": rows, "metrics": {}}
    for key in keys:
        values = np.asarray([row[key] for row in rows if row.get(key) is not None], float)
        n = len(values)
        summary["metrics"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if n > 1 else 0.0,
            "standard_error": float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "n": n,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = aggregate(args.run_dirs)
    atomic_json(args.output, summary)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
