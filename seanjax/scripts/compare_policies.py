#!/usr/bin/env python3
"""Side-by-side rollouts: one row per scenario, one column per policy, same seeds and pedestrians.

    python seanjax/scripts/compare_policies.py --config seanjax/configs/cmaes_stars_best.yaml \
        --policies goal_seeking social checkpoints/seanjax/cmaes/best_policy.yaml \
        --scenarios crossing head_on group_toward dense_flow --output runs/seanjax/compare.png

Each panel shows the robot path, pedestrians with the robot present (solid) and absent (dashed),
and the outcome with the episode's disruption and closest approach.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seanjax.cli import DEFAULT_CONFIG, build, setup_logging
from seanjax.core.metrics import all_metrics
from seanjax.core.rollout import rollout
from seanjax.core.scenarios import KINDS, SPLIT_OFFSETS, collate, make_scenario


def _label(entry: str) -> str:
    return f"{Path(entry).parent.name}/{Path(entry).stem}" if entry.endswith((".yaml", ".yml")) else entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policies", nargs="+", required=True, help="Policy names or YAML files")
    parser.add_argument("--scenarios", nargs="+", default=["crossing", "head_on", "group_toward", "dense_flow"],
                        choices=KINDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--pedestrians")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_logging()
    rows, cols = len(args.scenarios), len(args.policies)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    pool = None
    for c, entry in enumerate(args.policies):
        config, env, policy, evaluator, pool, device = build(args.config, policy=entry, pedestrian_model=args.pedestrians,
                                                             device=args.device)
        scenarios = [make_scenario(k, SPLIT_OFFSETS[args.split] + args.seed, pool=pool, split=args.split)
                     for k in args.scenarios]
        batch = collate(scenarios, device)
        run = rollout(env, batch, policy, policy.parameter_tensor(config["policy_parameters"], batch.size, device))
        cf = rollout(env, batch, robot_present=False)
        metrics = all_metrics(run, cf, batch, env.robot.radius)
        for r in range(rows):
            ax = axes[r, c]
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            for seg in batch.segments[r][batch.segment_mask[r]].cpu().numpy():
                ax.plot([seg[0], seg[2]], [seg[1], seg[3]], color="black", lw=2)
            pos, active = run.ped_pos[r].cpu().numpy(), run.ped_active[r].cpu().numpy()
            cpos, cactive = cf.ped_pos[r].cpu().numpy(), cf.ped_active[r].cpu().numpy()
            for p in np.flatnonzero(batch.ped_mask[r].cpu().numpy()):
                if cactive[p].any():
                    ax.plot(*cpos[p, cactive[p]].T, ls="--", lw=0.8, color=colors[p % 20], alpha=0.6)
                if active[p].any():
                    ax.plot(*pos[p, active[p]].T, lw=1.2, color=colors[p % 20])
            end = int(run.end_step[r])
            robot = run.robot_pose[r, : end + 1, :2].cpu().numpy()
            ax.plot(*robot.T, lw=2.6, color="#1f4e9c")
            ax.plot(*batch.robot_start[r, :2].cpu().numpy(), "s", color="#1f4e9c")
            ax.plot(*batch.robot_goal[r].cpu().numpy(), "*", color="#2e8b57", ms=12)
            outcome = ("success" if bool(run.success[r]) else
                       "collision" if bool(run.ped_collision[r] | run.obstacle_collision[r]) else "timeout")
            ax.set_title(f"{_label(entry)} | {args.scenarios[r]}\n{outcome} {end * run.dt:.1f}s, "
                         f"disrupt {float(metrics['disruption_displacement'][r]):.2f} m, "
                         f"min gap {float(np.nan_to_num(metrics['min_ped_clearance'][r].cpu(), nan=99)):.2f} m",
                         fontsize=8)
            ax.set_aspect("equal"), ax.grid(alpha=0.3), ax.tick_params(labelsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(args.output)


if __name__ == "__main__":
    main()
