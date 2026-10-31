#!/usr/bin/env python3
"""Tune the social navigation policy with CMA-ES (or ``--optimizer ppo``).

    python simplenavsim/scripts/train_cmaes.py --config simplenavsim/configs/cmaes_crowd_v4.yaml --device cuda --seed 42

Writes generations.jsonl, best_policy.yaml (validation-selected), default_policy.yaml,
checkpoint.pkl (resume with --resume), summary.json and convergence.png to --out-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simplenavsim.cli import DEFAULT_CONFIG, build, cmaes_config, setup_logging
from simplenavsim.optimization.cmaes import CMAESTrainer
from simplenavsim.visualization.plots import plot_convergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer", choices=("cmaes", "ppo"))
    parser.add_argument("--policy", help="Start from a saved policy YAML (its values become the initial mean)")
    parser.add_argument("--pedestrians", choices=("replay", "social_force", "reactive_replay", "stars_crowd"))
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/simplenavsim/cmaes"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    setup_logging()
    config, env, policy, evaluator, pool, device = build(args.config, policy=args.policy,
                                                         pedestrian_model=args.pedestrians, device=args.device)
    cfg = cmaes_config(config, num_generations=args.num_generations, population_size=args.population_size,
                       seed=args.seed, optimizer=args.optimizer)
    if args.policy:
        cfg.initial = {**config["policy_parameters"], **cfg.initial}
    print(f"device {device}, pedestrians {env.pedestrians.name}, policy {policy.name}, optimizing {cfg.optimize}")
    trainer = CMAESTrainer(cfg, evaluator, args.out_dir, pool=pool)
    summary = trainer.run(resume=args.resume)
    plot_convergence(args.out_dir / "generations.jsonl", args.out_dir / "convergence.png")
    print(json.dumps({"best_val_objective": summary["best"]["val_objective"],
                      "best_generation": summary["best"]["generation"], "best_parameters": summary["best"]["values"]},
                     indent=1))
    print(f"best policy: {args.out_dir / 'best_policy.yaml'}")


if __name__ == "__main__":
    main()
