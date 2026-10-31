"""Evaluate policies on the test split under every pedestrian condition; GIF grids from the saved runs.

Conditions (how pedestrians move):
    replay            scripted paths, no reaction
    social_force      scripted goals, social forces
    reactive_replay   scripted paths tracked with social forces (reactive demonstrations)
    stars_blind       STARS forecaster drives pedestrians, robot not in its graph
    stars_reactive    STARS forecaster, robot in its graph (the crowd every policy was trained in)

Policies (``POLICIES``): goal-seeking, the untuned social controller, v4 (the latest CMA-ES result)
with a constant-velocity or STARS predictor, PPO over the same parameters, and learned subgoals.
Scenes: the config's kinds x ``--per-kind`` test scenes (9 x 30 = 270), each scored by the config's
objective and every metric against a robot-absent run of the same condition. ``robot_influence``
needs the STARS crowd with the robot in its graph, so elsewhere the objective has no disruption term.
The GIFs show the first test scene of each kind. Crowd realism
is measured on recorded STEPS test scenes with the robot absent.

    python simplenavsim/scripts/evaluate.py --conditions replay social_force --device cuda
    python simplenavsim/scripts/evaluate.py --policy mine=checkpoints/simplenavsim/my_run/best_policy.yaml
    python simplenavsim/scripts/evaluate.py --render     # tables and GIF grids from the saved runs

Runs are saved as ``condition_<name>.pkl``; ``--render`` writes ``original.md`` (stars_reactive, the
evaluation the policies were selected by), ``conditions.md``, ``results.json`` and the GIF grids in ``FIGURES``.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from simplenavsim.cli import ROOT, recording_pool, setup_logging
from simplenavsim.core.config import CONFIG_DIR, load_config, load_yaml, make_controller, make_env, make_predictor
from simplenavsim.core.metrics import all_metrics, summarize
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import collate, scenario_set
from simplenavsim.optimization.evaluator import Objective
from simplenavsim.policies.base import load_policy_file
from simplenavsim.policies.ppo import SubgoalPolicy

FORCES = load_yaml(CONFIG_DIR / "pedestrians.yaml")["social_force"]
STARS = load_yaml(CONFIG_DIR / "pedestrians_stars.yaml")["stars_crowd"]
CONDITIONS = {
    "replay": {"model": "replay"},
    "social_force": {"model": "social_force", "social_force": FORCES},
    "reactive_replay": {"model": "reactive_replay", "social_force": FORCES},
    "stars_blind": {"model": "stars_crowd", "stars_crowd": {**STARS, "include_robot": False}},
    "stars_reactive": {"model": "stars_crowd", "stars_crowd": STARS},
}
CKPT = "checkpoints/simplenavsim"
POLICIES = {  # label: (controller, policy YAML, STARS predictor)
    "goal_seeking": ("goal_seeking", None, False),
    "social_default": ("social", None, False),
    "v4_cv": ("social", f"{CKPT}/cmaes_crowd_v4/best_policy.yaml", False),
    "v4": ("social", f"{CKPT}/cmaes_crowd_v4/best_policy.yaml", True),
    "ppo_params": ("social", f"{CKPT}/ppo_params_s42/best_policy.yaml", True),
    "subgoal_ppo": ("subgoal", f"{CKPT}/subgoal_s1/best_policy.yaml", True),
}
KIND_NAMES = {"crossing": "Crossing", "head_on": "Head-on", "overtaking": "Overtaking a slow walker",
              "following": "Following", "group_toward": "Oncoming group", "narrow_passage": "Narrow passage",
              "stationary_group": "Standing group", "dense_flow": "Two-way corridor", "recorded": "Recorded STEPS crowd"}
CONDITION_NAMES = {"replay": "Scripted paths\n(ignore robot)", "social_force": "Social forces",
                   "reactive_replay": "Scripted paths\n+ social forces", "stars_blind": "STARS forecaster\n(ignores robot)",
                   "stars_reactive": "STARS forecaster\n(reacts to robot)"}
POLICY_NAMES = {"v4": "CMA-ES\n(tuned controller)", "ppo_params": "PPO\n(same parameters)",
                "subgoal_ppo": "PPO subgoals\n(tuned controller)"}
ENCOUNTERS = ("crossing", "head_on", "overtaking", "following")
CROWDS = ("group_toward", "stationary_group", "narrow_passage", "dense_flow", "recorded")
FIGURES = {  # file: (title, rows ("conditions" or policies), scene kinds)
    "cmaes_conditions_encounters": ("CMA-ES-tuned controller under five pedestrian models: one-on-one encounters",
                                    "conditions", ENCOUNTERS),
    "cmaes_conditions_crowds": ("CMA-ES-tuned controller under five pedestrian models: groups and crowds",
                                "conditions", CROWDS),
    "policies": ("Tuned policies in the STARS crowd they were trained in", ("v4", "ppo_params", "subgoal_ppo"),
                 ("head_on", "group_toward", "narrow_passage", "dense_flow", "recorded")),
    "cmaes_vs_ppo": ("CMA-ES vs PPO tuning the same controller parameters (STARS crowd)", ("v4", "ppo_params"),
                     ("head_on", "group_toward", "narrow_passage", "dense_flow", "recorded")),
}
COLUMNS = ("objective", "success", "collision", "robot_influence", "intimate_violation_rate", "time_efficiency",
           "normalized_time", "path_efficiency", "command_smoothness", "loops", "min_ped_clearance",
           "disruption_local")


def make_policy(name, path, stars, env, policy_config, device):
    predictor = make_predictor(policy_config["predictor"] if stars else None, device)
    controller = make_controller(policy_config, "social" if name == "subgoal" else name, env.robot, predictor)
    if name == "subgoal":
        return SubgoalPolicy.load(ROOT / path, controller, device), {}
    values = controller.default_parameters()
    if path:
        values.update(load_policy_file(ROOT / path)[1])
    return controller, values


def recorded_rollout(policy, env, batch, values, device):
    """Rollout that also keeps the controller's plan at every step (for the animation)."""
    plans, act = [], policy.act

    def recording_act(obs, params):
        command = act(obs, params)
        if "plan" in policy.last_info:
            plans.append((policy.last_info["plan"].float().cpu().numpy(), policy.last_info["plan_steps"].cpu().numpy()))
        return command

    policy.act = recording_act
    try:
        run = rollout(env, batch, policy, policy.parameter_tensor(values, batch.size, device))
    finally:
        policy.act = act
    return run, plans


def snapshot(batch, run, plans, counterfactual):
    """What ``plots.animate_grid`` needs, as numpy."""
    n = lambda t: t.detach().float().cpu().numpy() if t.dtype.is_floating_point else t.detach().cpu().numpy()
    return {"names": batch.names, "dt": run.dt, "robot_pose": n(run.robot_pose), "robot_v": n(run.robot_v),
            "ped_pos": n(run.ped_pos), "ped_active": n(run.ped_active), "ped_mask": n(batch.ped_mask),
            "ped_radius": n(batch.ped_radius), "segments": n(batch.segments), "segment_mask": n(batch.segment_mask),
            "goal": n(batch.robot_goal), "start": n(batch.robot_start), "end_step": n(run.end_step),
            "success": n(run.success), "collision": n(run.ped_collision | run.obstacle_collision), "plans": plans,
            "ped_ghost": n(counterfactual.ped_pos), "ghost_active": n(counterfactual.ped_active)}


def realism(env, pool, per_kind, device):
    """Robot-absent fidelity to recorded STEPS tracks: position error, speed, pedestrian overlaps."""
    batch = collate(scenario_set("test", ["recorded"], per_kind, pool=pool), device)
    run = rollout(env, batch, robot_present=False)
    K = min(run.ped_pos.shape[2], batch.ped_ref.shape[2])
    pos, ref = run.ped_pos[:, :, :K], batch.ped_ref[:, :, :K]
    both = run.ped_active[:, :, :K] & batch.ped_valid[:, :, :K]
    error = torch.linalg.norm(pos - ref, dim=-1)
    age = both.long().cumsum(-1)                                                # steps since appearing
    speed = lambda xy: torch.linalg.norm(xy[:, :, 1:] - xy[:, :, :-1], dim=-1) / batch.dt
    moving = both[:, :, 1:] & both[:, :, :-1]
    gap = (torch.linalg.norm(pos[:, :, None] - pos[:, None], dim=-1)
           - (batch.ped_radius[:, :, None] + batch.ped_radius[:, None])[..., None])
    P = pos.shape[1]
    pair = both[:, :, None] & both[:, None] & ~torch.eye(P, dtype=torch.bool, device=pos.device)[None, :, :, None]
    return {"position_error": float(error[both].mean()),
            "error_at_5s": float(error[both & (age == 50)].mean()),
            "error_at_10s": float(error[both & (age == 100)].mean()),
            "speed_sim": float(speed(pos)[moving].mean()), "speed_real": float(speed(ref)[moving].mean()),
            "overlap_rate": float((gap < 0)[pair].float().mean())}


def run_condition(name, args, config, device):
    pedestrians = {k: dict(v) if isinstance(v, dict) else v for k, v in CONDITIONS[name].items()}
    if "stars_crowd" in pedestrians:
        pedestrians["stars_crowd"]["device"] = str(device)
    env = make_env(config["simulator"], pedestrians)
    objective, cm, pool = Objective(**config["objective"]), config["cmaes"], recording_pool(config)
    scenes = scenario_set("test", cm["kinds"], args.per_kind, dt=cm["dt"], duration=cm["duration"], pool=pool)
    kinds = [s.kind for s in scenes]
    fixed = [next(s for s in scenes if s.kind == kind) for kind in dict.fromkeys(kinds)]
    batch, fixed_batch = collate(scenes, device), collate(fixed, device)
    cf, fixed_cf = rollout(env, batch, robot_present=False), rollout(env, fixed_batch, robot_present=False)
    out = {"policies": {}, "animations": {}}
    for label, (policy_name, path, stars) in POLICIES.items():
        policy, values = make_policy(policy_name, path, stars, env, config["policy"], device)
        run = rollout(env, batch, policy, policy.parameter_tensor(values, batch.size, device))
        metrics = all_metrics(run, cf, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
        metrics["objective"] = objective(metrics)
        out["policies"][label] = {"all": summarize(metrics),
                                  "per_kind": {k: summarize(metrics, torch.tensor([x == k for x in kinds]))
                                               for k in dict.fromkeys(kinds)}}
        fixed_run, plans = recorded_rollout(policy, env, fixed_batch, values, device)
        out["animations"][label] = snapshot(fixed_batch, fixed_run, plans, fixed_cf)
        print(name, label, json.dumps({m: round(out["policies"][label]["all"][m], 3)
                                       for m in ("objective", "success", "collision")}), flush=True)
    out["realism"] = realism(env, pool, args.realism_scenes, device) if pool else {}
    return out


def table(rows, head):
    """Markdown table: ``rows`` are ``(*labels, summary)``."""
    lines = ["| " + " | ".join(head + COLUMNS) + " |", "|" + " --- |" * (len(head) + len(COLUMNS))]
    for *labels, r in rows:
        cells = ["—" if r.get(m) is None or r[m] != r[m] else f"{r[m]:.3f}" for m in COLUMNS]
        lines.append("| " + " | ".join(labels + cells) + " |")
    return "\n".join(lines)


def columns(snap, kinds):
    """The scenes of ``kinds`` (in that order) from a snapshot of one scene per kind."""
    idx = [[name.rsplit("-", 1)[0] for name in snap["names"]].index(k) for k in kinds]
    out = {k: (v[idx] if hasattr(v, "shape") and v.ndim and len(v) == len(snap["names"]) else v) for k, v in snap.items()}
    out["names"] = [snap["names"][i] for i in idx]
    out["plans"] = [(plan[idx], steps[idx]) for plan, steps in snap["plans"]]
    return out


def render(out_dir: Path, docs_dir: Path) -> None:
    from simplenavsim.visualization.plots import animate_grid

    results = {name: pickle.loads((out_dir / f"condition_{name}.pkl").read_bytes())
               for name in CONDITIONS if (out_dir / f"condition_{name}.pkl").exists()}
    reference = results["stars_reactive"]
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "original.md").write_text(
        f"# Policies in the training crowd (stars_reactive), {len(reference['policies']['v4']['per_kind'])} scene kinds\n\n"
        "The evaluation the policies were selected by (decision 99): the config objective and its terms.\n\n"
        + table([(p, r["all"]) for p, r in reference["policies"].items()], ("Policy",)) + "\n")
    v4_kinds = list(reference["policies"]["v4"]["per_kind"])
    keys = ("position_error", "error_at_5s", "error_at_10s", "speed_sim", "speed_real", "overlap_rate")
    lines = [f"# Policies x pedestrian conditions, {len(v4_kinds)} scene kinds\n",
             "`robot_influence` exists only in stars_reactive; elsewhere the objective has no disruption term.\n",
             table([(c, p, r["all"]) for c, res in results.items() for p, r in res["policies"].items()],
                   ("Condition", "Policy")),
             "\n## v4 objective by scene kind\n",
             "| Condition | " + " | ".join(v4_kinds) + " |", "|" + " --- |" * (1 + len(v4_kinds))]
    lines += [f"| {c} | " + " | ".join(f"{res['policies']['v4']['per_kind'][k]['objective']:.3f}" for k in v4_kinds)
              + " |" for c, res in results.items()]
    lines += ["\n## Crowd realism (recorded STEPS test scenes, robot absent)\n",
              "| Condition | " + " | ".join(keys) + " |", "|" + " --- |" * (len(keys) + 1)]
    lines += [f"| {c} | " + " | ".join(f"{res['realism'][k]:.3f}" for k in keys) + " |" for c, res in results.items()
              if res["realism"]]
    (docs_dir / "conditions.md").write_text("\n".join(lines) + "\n")
    (docs_dir / "results.json").write_text(json.dumps({c: {k: v for k, v in r.items() if k != "animations"}
                                                       for c, r in results.items()}, indent=1))
    available = [name.rsplit("-", 1)[0] for name in reference["animations"]["v4"]["names"]]
    for name, (title, rows, kinds) in FIGURES.items():
        kinds = [k for k in kinds if k in available]
        if rows == "conditions":
            snaps, labels = [res["animations"]["v4"] for res in results.values()], [CONDITION_NAMES[c] for c in results]
        else:
            snaps, labels = [reference["animations"][p] for p in rows], [POLICY_NAMES[p] for p in rows]
        animate_grid([columns(snap, kinds) for snap in snaps], labels, [KIND_NAMES[k] for k in kinds],
                     docs_dir / f"{name}.gif", title=title)
    print(f"saved {docs_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "cmaes_crowd_v4.yaml")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--policy", action="append", default=[], metavar="LABEL=YAML",
                        help="Also evaluate a saved policy (social or subgoal YAML from train_cmaes / train_ppo)")
    parser.add_argument("--per-kind", type=int, default=30)
    parser.add_argument("--realism-scenes", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/simplenavsim/evaluation"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/results/evaluation"))
    parser.add_argument("--render", action="store_true", help="Only build tables and GIFs from saved runs")
    args = parser.parse_args()
    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.render:
        return render(args.out_dir, args.docs_dir)
    config = load_config(args.config)
    for item in args.policy:
        label, path = item.split("=", 1)
        POLICIES[label] = (load_yaml(Path(path))["policy"], str(Path(path).resolve()), True)
    for name in args.conditions:
        result = run_condition(name, args, config, torch.device(args.device))
        (args.out_dir / f"condition_{name}.pkl").write_bytes(pickle.dumps(result))


if __name__ == "__main__":
    main()
