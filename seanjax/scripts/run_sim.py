#!/usr/bin/env python3
"""Run one scenario with a policy on the lightweight backend; print metrics, save a plot and GIF.

    python seanjax/scripts/run_sim.py --backend lightweight --scenario crossing --policy social
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seanjax.cli import DEFAULT_CONFIG, build, check_backend, setup_logging
from seanjax.core.scenarios import KINDS, collate, make_scenario
from seanjax.core.rollout import rollout
from seanjax.core.metrics import all_metrics, summarize
from seanjax.visualization.plots import animate_rollout, plot_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="lightweight", choices=("lightweight", "isaac_lab"))
    parser.add_argument("--scenario", default="crossing", choices=KINDS)
    parser.add_argument("--policy", default="social", help="social | goal_seeking | a saved policy YAML")
    parser.add_argument("--pedestrians", choices=("replay", "social_force", "reactive_replay", "orca", "stars_crowd"),
                        help="Pedestrian model (default: from the config)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/seanjax/sim"))
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    setup_logging()
    check_backend(args.backend)
    config, env, policy, evaluator, pool, device = build(args.config, policy=args.policy,
                                                         pedestrian_model=args.pedestrians, device=args.device)
    from seanjax.core.scenarios import SPLIT_OFFSETS

    scenario = make_scenario(args.scenario, SPLIT_OFFSETS[args.split] + args.seed, duration=args.duration, pool=pool,
                             split=args.split)
    batch = collate([scenario], device)
    params = policy.parameter_tensor(config["policy_parameters"], 1, device)
    run = rollout(env, batch, policy, params)
    counterfactual = rollout(env, batch, robot_present=False)
    metrics = summarize(all_metrics(run, counterfactual, batch, env.robot.radius))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{scenario.name}-{policy.name}-{env.pedestrians.name}"
    plot_rollout(batch, run, counterfactual=counterfactual, path=f"{stem}.png")
    if not args.no_gif:
        animate_rollout(batch, run, counterfactual=counterfactual, path=f"{stem}.gif")
    Path(f"{stem}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: round(v, 3) for k, v in metrics.items()}, indent=1))
    print(f"saved {stem}.png" + ("" if args.no_gif else f", {stem}.gif"))


if __name__ == "__main__":
    main()
