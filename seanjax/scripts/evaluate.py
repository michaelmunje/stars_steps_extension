#!/usr/bin/env python3
"""Compare policies on held-out scenarios (default: the untouched test split).

    python seanjax/scripts/evaluate.py --policy checkpoints/seanjax/cmaes/best_policy.yaml --split test

Always includes the two baselines with default parameters (goal_seeking, social) unless
--no-baselines; extra policies (names or YAML files) via --compare. Writes a JSON summary with
per-kind breakdowns and a comparison bar chart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seanjax.cli import DEFAULT_CONFIG, build, check_backend, setup_logging
from seanjax.core.metrics import summarize
from seanjax.core.scenarios import scenario_set
from seanjax.visualization.plots import plot_comparison

HEADLINE = ("objective", "success", "collision", "time_to_goal", "min_ped_clearance", "personal_violation_rate",
            "disruption_displacement", "ped_delay", "peds_affected", "stops", "command_smoothness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="lightweight", choices=("lightweight", "isaac_lab"))
    parser.add_argument("--policy", action="append", default=[], help="Policy YAML (repeatable)")
    parser.add_argument("--compare", nargs="*", default=[], help="More policies: names or YAML files")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--per-kind", type=int, help="Scenarios per kind (default: config test_per_kind)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pedestrians", choices=("replay", "social_force", "reactive_replay", "orca"))
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/seanjax/evaluate"))
    args = parser.parse_args()
    setup_logging()
    check_backend(args.backend)
    entries = ([] if args.no_baselines else ["goal_seeking", "social"]) + args.compare + args.policy
    config, *_ = build(args.config)
    cm = config.get("cmaes") or {}
    per_kind = args.per_kind or config.get("test_per_kind", 10)
    results, summaries = {}, {}
    for entry in entries:
        config, env, policy, evaluator, pool, device = build(args.config, policy=entry, pedestrian_model=args.pedestrians,
                                                             device=args.device)
        scenarios = scenario_set(args.split, cm.get("kinds"), per_kind, dt=cm.get("dt", 0.1),
                                 duration=cm.get("duration", 30.0), pool=pool)
        label = (f"{Path(entry).parent.name}/{Path(entry).stem}" if entry.endswith((".yaml", ".yml"))
                 else f"{entry} (default)")
        metrics, *_ = evaluator.evaluate_detailed(config["policy_parameters"], scenarios)
        kinds = [s.kind for s in scenarios]
        per = {kind: summarize(metrics, torch.tensor([k == kind for k in kinds])) for kind in sorted(set(kinds))}
        summaries[label] = summarize(metrics)
        results[label] = {"policy": policy.name, "parameters": config["policy_parameters"], "overall": summaries[label],
                          "per_kind": per, "n_scenarios": len(scenarios)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"evaluation_{args.split}.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_comparison(summaries, args.out_dir / f"comparison_{args.split}.png")
    width = max(len(k) for k in summaries) + 2
    print(f"{'policy':{width}s}" + "".join(f"{m[:12]:>13s}" for m in HEADLINE))
    for label, s in summaries.items():
        print(f"{label:{width}s}" + "".join(f"{s.get(m, float('nan')):13.3f}" for m in HEADLINE))
    print(f"saved {args.out_dir}")


if __name__ == "__main__":
    main()
