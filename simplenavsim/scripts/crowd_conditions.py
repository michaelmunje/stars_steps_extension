"""Does the forecaster help? The same policies under six pedestrian conditions.

Rows (pedestrian conditions):
    replay            scripted paths, no reaction
    social_force      scripted goals, social forces
    reactive_replay   scripted paths tracked with social forces (reactive demonstrations)
    stars_blind       STARS forecaster drives pedestrians, robot not in its graph
    stars_reactive    STARS forecaster, robot in its graph (the reference crowd)
    stars_social      STARS forecasts as intents, social forces for motion

Policies: goal-seeking, the untuned social controller (constant-velocity predictor), and v4 with a
constant-velocity or a STARS predictor. Scenes: ``--kinds`` of the test split, ``--per-kind`` each;
the first of each kind is the fixed scene shown in the GIF grid. Crowd realism is measured on
recorded STEPS test scenes with the robot absent.

    python simplenavsim/scripts/crowd_conditions.py --conditions replay social_force --device cuda
    python simplenavsim/scripts/crowd_conditions.py --render       # tables + GIF grids from saved runs
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
from simplenavsim.policies.base import load_policy_file

FORCES = load_yaml(CONFIG_DIR / "pedestrians.yaml")["social_force"]
STARS = load_yaml(CONFIG_DIR / "pedestrians_stars.yaml")["stars_crowd"]
CONDITIONS = {
    "replay": {"model": "replay"},
    "social_force": {"model": "social_force", "social_force": FORCES},
    "reactive_replay": {"model": "reactive_replay", "social_force": FORCES},
    "stars_blind": {"model": "stars_crowd", "stars_crowd": {**STARS, "include_robot": False}},
    "stars_reactive": {"model": "stars_crowd", "stars_crowd": STARS},
    "stars_social": {"model": "stars_social", "stars_crowd": STARS, "social_force": FORCES},
}
V4 = "checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml"
POLICIES = {"goal_seeking": ("goal_seeking", None, False), "social_default": ("social", None, False),
            "v4_cv": ("social", V4, False), "v4_stars": ("social", V4, True)}
ANIMATED = ("v4_stars", "v4_cv")
METRICS = ("success", "collision", "intimate_violation_rate", "min_ped_clearance", "time_efficiency",
           "path_efficiency", "loops", "disruption_local")


def make_policy(name, path, stars, env, policy_config, device):
    predictor = make_predictor(policy_config["predictor"] if stars else None, device)
    controller = make_controller(policy_config, name, env.robot, predictor)
    values = controller.default_parameters()
    if path:
        values.update(load_policy_file(ROOT / path)[1])
    return controller, values


def recorded_rollout(controller, env, batch, values, device):
    """Rollout that also keeps the controller's plan at every step (for the animation)."""
    plans, act = [], controller.act

    def recording_act(obs, params):
        command = act(obs, params)
        info = controller.last_info
        plans.append((info["plan"].float().cpu().numpy(), info["plan_steps"].cpu().numpy()))
        return command

    controller.act = recording_act
    try:
        run = rollout(env, batch, controller, controller.parameter_tensor(values, batch.size, device))
    finally:
        controller.act = act
    return run, plans


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
    policy_config = config["policy"]
    scenes = scenario_set("test", args.kinds, args.per_kind)
    fixed = [next(s for s in scenes if s.kind == kind) for kind in args.kinds]
    batch, fixed_batch = collate(scenes, device), collate(fixed, device)
    counterfactual = rollout(env, batch, robot_present=False)
    fixed_cf = rollout(env, fixed_batch, robot_present=False)
    out = {"policies": {}, "animations": {}}
    for label, (policy_name, path, stars) in POLICIES.items():
        controller, values = make_policy(policy_name, path, stars, env, policy_config, device)
        run = rollout(env, batch, controller, controller.parameter_tensor(values, batch.size, device))
        metrics = all_metrics(run, counterfactual, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
        kinds = [s.kind for s in scenes]
        fixed_run, plans = recorded_rollout(controller, env, fixed_batch, values, device)
        fixed_metrics = all_metrics(fixed_run, fixed_cf, fixed_batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
        out["policies"][label] = {
            "all": summarize(metrics),
            "per_kind": {k: summarize(metrics, torch.tensor([x == k for x in kinds])) for k in args.kinds},
            "fixed": {s.name: {m: float(fixed_metrics[m][i]) for m in METRICS + ("time_to_goal",)}
                      for i, s in enumerate(fixed)}}
        if label in ANIMATED:
            out["animations"][label] = snapshot(fixed_batch, fixed_run, plans)
    out["realism"] = realism(env, recording_pool(config), args.realism_scenes, device)
    return out


def snapshot(batch, run, plans):
    """What ``plots.animate_grid`` needs, as numpy."""
    n = lambda t: t.detach().float().cpu().numpy() if t.dtype.is_floating_point else t.detach().cpu().numpy()
    return {"names": batch.names, "dt": run.dt, "robot_pose": n(run.robot_pose), "robot_v": n(run.robot_v),
            "ped_pos": n(run.ped_pos), "ped_active": n(run.ped_active), "ped_mask": n(batch.ped_mask),
            "ped_radius": n(batch.ped_radius), "segments": n(batch.segments), "segment_mask": n(batch.segment_mask),
            "goal": n(batch.robot_goal), "start": n(batch.robot_start), "end_step": n(run.end_step),
            "success": n(run.success), "collision": n(run.ped_collision | run.obstacle_collision), "plans": plans}


def render(out_dir: Path) -> None:
    from simplenavsim.visualization.plots import animate_grid

    results = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(out_dir.glob("condition_*.pkl"))}
    results = {name: results[f"condition_{name}"] for name in CONDITIONS if f"condition_{name}" in results}
    rows = ["# Pedestrian conditions × policies\n",
            f"Scenes: {', '.join(next(iter(results.values()))['policies']['v4_stars']['per_kind'])} "
            "(test split). Means over all scenes; the fixed-scene outcomes are those in the GIF grids.\n",
            "| Condition | Policy | " + " | ".join(METRICS) + " |", "|" + " --- |" * (len(METRICS) + 2)]
    for cond, res in results.items():
        for label, r in res["policies"].items():
            rows.append(f"| {cond} | {label} | " + " | ".join(f"{r['all'].get(m, float('nan')):.3f}" for m in METRICS) + " |")
    rows += ["\n## Fixed scenes (v4 + STARS predictor)\n", "| Condition | " + " | ".join(
        next(iter(results.values()))["policies"]["v4_stars"]["fixed"]) + " |",
             "|" + " --- |" * (1 + len(next(iter(results.values()))["policies"]["v4_stars"]["fixed"]))]
    for cond, res in results.items():
        cells = [("reached" if f["success"] else ("COLLISION" if f["collision"] else "timeout"))
                 + (f" {f['time_to_goal']:.1f} s" if f["success"] else "") + f", gap {f['min_ped_clearance']:.2f} m"
                 for f in res["policies"]["v4_stars"]["fixed"].values()]
        rows.append(f"| {cond} | " + " | ".join(cells) + " |")
    keys = ("position_error", "error_at_5s", "error_at_10s", "speed_sim", "speed_real", "overlap_rate")
    rows += ["\n## Crowd realism (recorded STEPS test scenes, robot absent)\n",
             "| Condition | " + " | ".join(keys) + " |", "|" + " --- |" * (len(keys) + 1)]
    for cond, res in results.items():
        rows.append(f"| {cond} | " + " | ".join(f"{res['realism'][k]:.3f}" for k in keys) + " |")
    (out_dir / "results.md").write_text("\n".join(rows) + "\n")
    (out_dir / "results.json").write_text(json.dumps({c: {k: v for k, v in r.items() if k != "animations"}
                                                      for c, r in results.items()}, indent=1))
    for label in ANIMATED:
        animate_grid([results[c]["animations"][label] for c in results], list(results),
                     out_dir / f"grid_{label}.gif", title=f"{label}: rows = pedestrian condition")
    print(f"saved {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "cmaes_crowd_v4.yaml")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--kinds", nargs="*", default=["crossing", "group_toward", "dense_flow"])
    parser.add_argument("--per-kind", type=int, default=30)
    parser.add_argument("--realism-scenes", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/simplenavsim/crowd_conditions"))
    parser.add_argument("--render", action="store_true", help="Only build tables and GIFs from saved runs")
    args = parser.parse_args()
    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.render:
        return render(args.out_dir)
    config = load_config(args.config)
    for name in args.conditions:
        result = run_condition(name, args, config, torch.device(args.device))
        (args.out_dir / f"condition_{name}.pkl").write_bytes(pickle.dumps(result))
        print(name, json.dumps({k: round(v["all"]["success"], 3) for k, v in result["policies"].items()}))


if __name__ == "__main__":
    main()
