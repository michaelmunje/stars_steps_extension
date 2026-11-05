"""Failure cases of the agent-judged CMA-ES policy, with the other policies on the same scenes.

Failure case = a held-out test scene where the agent-judged policy times out or collides; up to six, at
most two per kind, worst hand-crafted reward first. One GIF of all of them (rows = policies) and one per
scene; ``scenes.json`` gives every policy's outcome, and how many of the 10 random-weight and 10 CMA-ES
seeds reach the goal there.

    python docs/results/agent_judge/scripts/failure_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "simplenavsim/scripts")
from evaluate import recorded_rollout, snapshot  # noqa: E402
from judged_cmaes import PRINCIPLES_J, JudgedSearch  # noqa: E402

from simplenavsim.core.config import CONFIG_DIR, load_config  # noqa: E402
from simplenavsim.core.metrics import all_metrics  # noqa: E402
from simplenavsim.core.rollout import rollout  # noqa: E402
from simplenavsim.core.scenarios import collate, scenario_set  # noqa: E402
from simplenavsim.optimization.cmaes import random_weights  # noqa: E402
from simplenavsim.policies.base import load_policy_file  # noqa: E402
from simplenavsim.visualization.plots import animate_grid  # noqa: E402

A, D = Path("runs/simplenavsim/agent_judge"), Path("docs/results/agent_judge")
JUDGED = "Agent-judged\nCMA-ES"
SEEDS = "checkpoints/simplenavsim/seeds_random_init/cmaes_s{}/best_policy.yaml"
KIND_NAMES = {"head_on": "Head-on", "crossing": "Crossing", "group_toward": "Oncoming group",
              "narrow_passage": "Narrow passage", "stationary_group": "Standing group", "dense_flow": "Two-way corridor",
              "overtaking": "Overtaking"}


def outcomes(env, policy, batch, values, device):
    run = rollout(env, batch, policy, policy.parameter_tensor(values, batch.size, device))
    m = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
    label = ["collision" if c else "reached" if s else "timed out" for s, c in zip(m["S"], m["C"])]
    return label, PRINCIPLES_J(m).numpy()


def main(count: int = 6) -> None:
    search = JudgedSearch(A / "exp1_judged")
    env, policy = search.stack()
    base, device = search.state["base"], search.device
    names = load_config(CONFIG_DIR / "cmaes_crowd_principles.yaml")["cmaes"]["optimize"]
    kinds = search.kinds("test_kinds") + ["overtaking"]
    scenes = scenario_set("test", kinds, 30, dt=search.j["dt"], duration=search.j["duration"])
    batch = collate(scenes, device)
    shown = {"Random weights\n(seed 0)": {**base, **random_weights(policy.parameters, names, 0)},
             "CMA-ES\n(seed 0)": {**base, **load_policy_file(SEEDS.format(0))[1]},
             JUDGED: {**base, **load_policy_file(D / "judged_cmaes_policy.yaml")[1]}}
    judged, reward = outcomes(env, policy, batch, shown[JUDGED], device)
    failed = [i for i, o in enumerate(judged) if o != "reached"]
    chosen, per_kind = [], {}
    for i in sorted(failed, key=lambda i: reward[i]):
        if per_kind.get(scenes[i].kind, 0) < 2:
            chosen.append(i)
            per_kind[scenes[i].kind] = per_kind.get(scenes[i].kind, 0) + 1
        if len(chosen) == count:
            break
    cases = [scenes[i] for i in chosen]
    fixed = collate(cases, device)
    reached = {"random weights (10 seeds)": np.zeros(len(cases), int), "CMA-ES (10 seeds)": np.zeros(len(cases), int)}
    for s in range(10):
        for group, values in (("random weights (10 seeds)", {**base, **random_weights(policy.parameters, names, s)}),
                              ("CMA-ES (10 seeds)", {**base, **load_policy_file(SEEDS.format(s))[1]})):
            label, _ = outcomes(env, policy, fixed, values, device)
            reached[group] += np.array([o == "reached" for o in label])
    cf = rollout(env, fixed, robot_present=False)
    snaps, shown_outcomes = [], {}
    for name, values in shown.items():
        run, plans = recorded_rollout(policy, env, fixed, values, device)
        snaps.append(snapshot(fixed, run, plans, cf))
        shown_outcomes[name] = ["collision" if c else "reached" if s else "timed out"
                                for s, c in zip(run.success.tolist(), (run.ped_collision | run.obstacle_collision).tolist())]
    cols = [f"{KIND_NAMES[s.kind]} (#{s.seed - 2_000_000})" for s in cases]
    out = D / "failure_cases"
    out.mkdir(exist_ok=True)
    animate_grid(snaps, list(shown), cols, out / "failure_cases.gif",
                 title="Failure cases of the agent-judged CMA-ES policy (held-out scenes), other policies on the same scenes")
    for c in range(len(cases)):
        one = [{k: (v[[c]] if hasattr(v, "shape") and v.ndim and len(v) == len(cases) else v) for k, v in s.items()}
               for s in snaps]
        for s, full in zip(one, snaps):
            s["names"], s["plans"] = [full["names"][c]], [(p[[c]], st[[c]]) for p, st in full["plans"]]
        animate_grid(one, list(shown), [cols[c]], out / f"{cases[c].name}.gif", title=f"Failure case: {cols[c]}")
    summary = {s.name: {**{name.replace("\n", " "): shown_outcomes[name][c] for name in shown},
                        **{f"{group} reaching the goal": int(reached[group][c]) for group in reached}}
               for c, s in enumerate(cases)}
    (out / "scenes.json").write_text(json.dumps({
        "selection": f"held-out scenes where the agent-judged policy fails ({len(failed)} of {len(scenes)}), "
                     "worst hand-crafted reward first, <= 2 per kind", "scenes": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
