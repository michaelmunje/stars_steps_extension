"""GIF grid of trained policies on the fixed crowd-conditions scenes (rows = policy, columns = scene).

Each row is a saved policy YAML (social / subgoal; ``ppo`` arc-choice checkpoints need
``--arc-module``: `git show 4b91eae:src/simplenavsim/policies/ppo.py > arc_ppo.py`) run in the config's crowd. Also writes a table
over ``--per-kind`` test scenes per kind.

    python simplenavsim/scripts/render_policies.py --device cuda \\
        --policy v4=checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml \\
        --policy subgoal=checkpoints/simplenavsim/subgoal_s1/best_policy.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from crowd_conditions import METRICS, recorded_rollout, snapshot  # noqa: E402

from simplenavsim.cli import build, setup_logging
from simplenavsim.core.config import make_controller, make_predictor
from simplenavsim.core.metrics import all_metrics, summarize
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import collate, scenario_set
from simplenavsim.visualization.plots import animate_grid


def load(path, args, device):
    """``(env, policy, values)`` for a saved policy YAML."""
    if yaml.safe_load(Path(path).read_text())["policy"] != "ppo":
        config, env, policy, _, _, _ = build(args.config, policy=path, device=device)
        return env, policy, {} if policy.name == "subgoal" else config["policy_parameters"]
    spec = importlib.util.spec_from_file_location("arc_ppo", args.arc_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, env, _, _, _, _ = build(args.config, device=device)
    controller = make_controller(config["policy"], "social", env.robot, make_predictor(config["policy"].get("predictor"), device))
    return env, module.PPOPolicy.load(path, controller, device), {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("simplenavsim/configs/cmaes_crowd_v4.yaml"))
    parser.add_argument("--policy", action="append", required=True, help="label=path/to/policy.yaml")
    parser.add_argument("--arc-module", type=Path, help="policies/ppo.py from commit 4b91eae (arc-choice PPO)")
    parser.add_argument("--kinds", nargs="*", default=["crossing", "group_toward", "dense_flow"])
    parser.add_argument("--per-kind", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/simplenavsim/policy_grid"))
    args = parser.parse_args()
    setup_logging()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenes = scenario_set("test", args.kinds, args.per_kind)
    fixed = [next(s for s in scenes if s.kind == kind) for kind in args.kinds]
    batch, fixed_batch = collate(scenes, device), collate(fixed, device)
    rows, snaps, results = [], [], {}
    for item in args.policy:
        label, path = item.split("=", 1)
        env, policy, values = load(path, args, device)
        cf = rollout(env, batch, robot_present=False)
        run = rollout(env, batch, policy, policy.parameter_tensor(values, batch.size, device))
        metrics = all_metrics(run, cf, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
        fixed_run, plans = recorded_rollout(policy, env, fixed_batch, values, device)
        fixed_cf = rollout(env, fixed_batch, robot_present=False)
        if yaml.safe_load(Path(path).read_text())["policy"] == "ppo":
            plans = []                      # the controller's plan is not the arc PPO chose
        snaps.append(snapshot(fixed_batch, fixed_run, plans, fixed_cf))
        results[label] = summarize(metrics)
        rows.append(f"| {label} | " + " | ".join(f"{results[label].get(m, float('nan')):.3f}" for m in METRICS) + " |")
        print(label, json.dumps({m: round(results[label][m], 3) for m in ("success", "collision")}), flush=True)
    table = ["| Policy | " + " | ".join(METRICS) + " |", "|" + " --- |" * (len(METRICS) + 1), *rows]
    (args.out_dir / "results.md").write_text(f"# Policies, {len(scenes)} test scenes ({', '.join(args.kinds)})\n\n"
                                             + "\n".join(table) + "\n")
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (args.out_dir / "snapshots.pkl").write_bytes(pickle.dumps(snaps))
    animate_grid(snaps, [p.split("=", 1)[0] for p in args.policy], args.out_dir / "grid_policies.gif",
                 title="rows = policy (STARS crowd); dashed line = plan, dashed circle = pedestrian without the robot")
    print(f"saved {args.out_dir}")


if __name__ == "__main__":
    main()
