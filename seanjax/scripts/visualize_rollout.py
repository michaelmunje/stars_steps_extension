#!/usr/bin/env python3
"""Plot and animate a policy on one scenario, with the robot-absent counterfactual pedestrians.

    python seanjax/scripts/visualize_rollout.py --checkpoint checkpoints/seanjax/cmaes/best_policy.yaml \
        --scenario crossing

With --run-dir, also redraws that CMA-ES run's convergence plot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from seanjax.cli import DEFAULT_CONFIG, build, setup_logging
from seanjax.core.rollout import rollout
from seanjax.core.scenarios import KINDS, SPLIT_OFFSETS, collate, make_scenario
from seanjax.visualization.plots import animate_rollout, plot_convergence, plot_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="social", help="Policy YAML, or social / goal_seeking")
    parser.add_argument("--scenario", default="crossing", choices=KINDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pedestrians", choices=("replay", "social_force", "reactive_replay", "orca", "stars_crowd"))
    parser.add_argument("--run-dir", type=Path, help="CMA-ES output directory to plot convergence for")
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/seanjax/visualize"))
    args = parser.parse_args()
    setup_logging()
    config, env, policy, evaluator, pool, device = build(args.config, policy=args.checkpoint,
                                                         pedestrian_model=args.pedestrians, device=args.device)
    scenario = make_scenario(args.scenario, SPLIT_OFFSETS[args.split] + args.seed, pool=pool, split=args.split)
    batch = collate([scenario], device)
    run = rollout(env, batch, policy, policy.parameter_tensor(config["policy_parameters"], 1, device))
    counterfactual = rollout(env, batch, robot_present=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{scenario.name}-{Path(args.checkpoint).stem}"
    plot_rollout(batch, run, counterfactual=counterfactual, path=f"{stem}.png")
    animate_rollout(batch, run, counterfactual=counterfactual, path=f"{stem}.gif")
    print(f"saved {stem}.png, {stem}.gif")
    if args.run_dir:
        plot_convergence(args.run_dir / "generations.jsonl", args.out_dir / "convergence.png")
        print(f"saved {args.out_dir / 'convergence.png'}")


if __name__ == "__main__":
    main()
