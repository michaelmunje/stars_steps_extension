"""Random vs CMA-ES-tuned controller weights, 10 seeds each, scored with standard social-navigation metrics.

Random, seed s: one weight set drawn uniformly from the CMA-ES search range (log-uniform for
log-scale weights) with seed s. CMA-ES, seed s: ``train_cmaes.py --seed s --random-init s``, i.e.
CMA-ES started from random seed s's weights (a paired before/after comparison), saved under
``checkpoints/simplenavsim/seeds_random_init/cmaes_s<s>/``.

Every policy runs on the test split (the config's 9 kinds x 30 scenes) in the STARS crowd it is
tuned in. Metrics are Table I of Francis et al., "Principles and Guidelines for Evaluating Social
Robot Navigation Algorithms" (``core.metrics.principles_metrics``); the table gives the mean over
seeds of each seed's scene average, with its standard error, and the paired change (CMA-ES minus
random, per seed) with its standard error. One GIF per policy and seed shows the
first test scene of six kinds.

    python simplenavsim/scripts/seed_study.py --policies random_s0 cmaes_s0 --device cuda
    python simplenavsim/scripts/seed_study.py --render
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import CONDITIONS, KIND_NAMES, make_policy, recorded_rollout, snapshot  # noqa: E402

from simplenavsim.cli import recording_pool, setup_logging
from simplenavsim.core.config import CONFIG_DIR, load_config, make_env
from simplenavsim.core.metrics import all_metrics, principles_metrics, summarize
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import collate, scenario_set
from simplenavsim.optimization.cmaes import random_weights
from simplenavsim.optimization.evaluator import Objective

SEEDS = range(10)
GROUPS = {"random": "Random weights", "cmaes": "CMA-ES"}
POLICIES = [f"{g}_s{s}" for g in GROUPS for s in SEEDS]
FIXED = ("crossing", "head_on", "group_toward", "narrow_passage", "dense_flow", "recorded")
METRICS = [  # (key, name, unit, better)
    ("S", "Success rate", "", "high"), ("C", "Collision rate", "", "low"), ("HC", "Human collisions", "", "low"),
    ("WC", "Wall collisions", "", "low"), ("TO", "Timeouts", "", "low"),
    ("FP", "Failures to progress (5 s, 0.1 m)", "per episode", "low"), ("ST", "Stalled time (< 0.05 m/s)", "s", "low"),
    ("T", "Time to reach goal (successes)", "s", "low"), ("PL", "Path length", "m", ""),
    ("SPL", "Success weighted by path length", "", "high"),
    ("Vavg", "Average speed", "m/s", ""), ("Vmax", "Maximum speed", "m/s", ""),
    ("Aavg", "Average acceleration magnitude", "m/s^2", "low"), ("Amin", "Minimum acceleration", "m/s^2", ""),
    ("Amax", "Maximum acceleration", "m/s^2", ""), ("Javg", "Average jerk magnitude", "m/s^3", "low"),
    ("Jmin", "Minimum jerk", "m/s^3", ""), ("Jmax", "Maximum jerk", "m/s^3", ""),
    ("CDmin", "Minimum clearance to walls (scenes with walls)", "m", "high"),
    ("CDavg", "Average clearance to walls (scenes with walls)", "m", ""),
    ("DHmin", "Minimum distance to a human", "m", "high"),
    ("SC", "Personal space compliance (>= 0.5 m)", "", "high"),
    ("TTC", "Minimum time to collision (constant velocity, capped at 10 s)", "s", "high"),
    ("objective", "Training objective J (not Table I)", "", "high"),
]


def run_policy(label, config, env, scenes, fixed, device):
    group, _, seed = label.partition("_s")
    path = f"checkpoints/simplenavsim/seeds_random_init/cmaes_s{seed}/best_policy.yaml" if group == "cmaes" else None
    policy, values = make_policy("social", path, True, env, config["policy"], device)
    if group == "random":
        values = {**values, **random_weights(policy.parameters, config["cmaes"]["optimize"], int(seed))}
    batch, fixed_batch = collate(scenes, device), collate(fixed, device)
    run = rollout(env, batch, policy, policy.parameter_tensor(values, batch.size, device))
    metrics = principles_metrics(run, batch, env.robot.radius, env.sim.goal_tolerance)
    metrics["objective"] = Objective(**config["objective"])(
        all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance))
    kinds = [s.kind for s in scenes]
    fixed_run, plans = recorded_rollout(policy, env, fixed_batch, values, device)
    return {"weights": values, "all": summarize(metrics),
            "per_kind": {k: summarize(metrics, torch.tensor([x == k for x in kinds])) for k in dict.fromkeys(kinds)},
            "animation": snapshot(fixed_batch, fixed_run, plans, rollout(env, fixed_batch, robot_present=False))}


def mean_se(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    se = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5 / len(values) ** 0.5 if len(values) > 1 else 0.0
    return m, se


def render(out_dir: Path, docs_dir: Path) -> None:
    from simplenavsim.visualization.plots import animate_grid

    results = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(out_dir.glob("*.pkl"))}
    docs_dir.mkdir(parents=True, exist_ok=True)
    groups = {g: [results[f"{g}_s{s}"] for s in SEEDS if f"{g}_s{s}" in results] for g in GROUPS}
    cell = lambda m, se: "—" if math.isnan(m) else f"{m:.3f} ± {se:.3f}"
    paired = [s for s in SEEDS if f"random_s{s}" in results and f"cmaes_s{s}" in results]
    lines = ["# Random vs CMA-ES-tuned controller weights (STARS crowd, test split)\n",
             "Random seed s: one weight set drawn uniformly from the CMA-ES search range. CMA-ES seed s: "
             "`train_cmaes.py --config cmaes_crowd_principles.yaml --seed s --random-init s`, i.e. CMA-ES started "
             "from random seed s's weights (200 generations, population 32, about 2 h on one GPU), objective "
             "J = SPL - 2 C + SC - 0.3 time/30 s. Narrow passages are 2.0-2.8 m wide.\n",
             f"{len(groups['random'])} random and {len(groups['cmaes'])} CMA-ES seeds; each seed is evaluated on "
             "the same 270 test scenes (9 kinds x 30). Cells: mean over seeds ± standard error; *Change* is the "
             "per-seed difference CMA-ES minus its own random start, averaged over seeds. Metrics from Table I of "
             "Francis et al., *Principles and Guidelines for Evaluating Social Robot Navigation Algorithms* (2023); "
             "distances are measured edge to edge; T averages successful episodes only; CD averages scenes with "
             "walls only. Aggregated time (AT) is omitted: our pedestrians have no cooperative goals.\n",
             "| Metric | Unit | Better | " + " | ".join(GROUPS.values()) + " | Change (paired) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for key, name, unit, better in METRICS:
        cells = [cell(*mean_se([r["all"][key] for r in groups[g]])) for g in GROUPS]
        change = mean_se([results[f"cmaes_s{s}"]["all"][key] - results[f"random_s{s}"]["all"][key] for s in paired])
        lines.append(f"| {name} (`{key}`) | {unit} | {better} | " + " | ".join(cells) + f" | {cell(*change)} |")
    kinds = list(groups["random"][0]["per_kind"]) if groups["random"] else []
    lines += ["\n## Success / collision rate by scene kind (mean over seeds)\n",
              "| Kind | " + " | ".join(GROUPS.values()) + " |", "| --- | --- | --- |"]
    for k in kinds:
        cells = [f"{mean_se([r['per_kind'][k]['S'] for r in groups[g]])[0]:.2f} / "
                 f"{mean_se([r['per_kind'][k]['C'] for r in groups[g]])[0]:.2f}" if groups[g] else "—" for g in GROUPS]
        lines.append(f"| {KIND_NAMES[k]} | " + " | ".join(cells) + " |")
    (docs_dir / "README.md").write_text("\n".join(lines) + "\n")
    (docs_dir / "results.json").write_text(json.dumps({k: {f: v[f] for f in ("weights", "all", "per_kind")}
                                                       for k, v in results.items()}, indent=1))
    for label, res in results.items():
        group, _, seed = label.partition("_s")
        snap = res["animation"]
        kinds = [n.rsplit("-", 1)[0] for n in snap["names"]]
        (docs_dir / group).mkdir(exist_ok=True)
        start = " (from random seed {seed})".format(seed=seed) if group == "cmaes" else ""
        animate_grid([snap], [f"{GROUPS[group]}\nseed {seed}"], [KIND_NAMES[k] for k in kinds],
                     docs_dir / group / f"seed_{seed}.gif", title=f"{GROUPS[group]}, seed {seed}{start} (STARS crowd)")
    print(f"saved {docs_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "cmaes_crowd_principles.yaml")
    parser.add_argument("--policies", nargs="*", default=POLICIES, choices=POLICIES)
    parser.add_argument("--per-kind", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/simplenavsim/seed_study"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/results/seed_study"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.render:
        return render(args.out_dir, args.docs_dir)
    config, device = load_config(args.config), torch.device(args.device)
    pedestrians = copy.deepcopy(CONDITIONS["stars_reactive"])
    pedestrians["stars_crowd"]["device"] = str(device)
    env = make_env(config["simulator"], pedestrians)
    cm, pool = config["cmaes"], recording_pool(config)
    scenes = scenario_set("test", cm["kinds"], args.per_kind, dt=cm["dt"], duration=cm["duration"], pool=pool)
    fixed = [next(s for s in scenes if s.kind == k) for k in FIXED if any(s.kind == k for s in scenes)]
    for label in args.policies:
        result = run_policy(label, config, env, scenes, fixed, device)
        (args.out_dir / f"{label}.pkl").write_bytes(pickle.dumps(result))
        print(label, json.dumps({k: round(result["all"][k], 3) for k in ("S", "C", "SPL", "objective")}), flush=True)


if __name__ == "__main__":
    main()
