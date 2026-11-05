"""GIFs of the hardest held-out test scenes for the start weights and the best policy of each experiment.

Hard = lowest Principles J averaged over the three policies (ties: smallest minimum gap), at most two
scenes per kind; one GIF of all of them (rows = policies) and one per scene.

    python docs/results/agent_judge/scripts/hard_scenes.py
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

from simplenavsim.core.metrics import all_metrics  # noqa: E402
from simplenavsim.core.rollout import rollout  # noqa: E402
from simplenavsim.core.scenarios import collate, scenario_set  # noqa: E402
from simplenavsim.policies.base import load_policy_file  # noqa: E402
from simplenavsim.visualization.plots import animate_grid  # noqa: E402

A, D = Path("runs/simplenavsim/agent_judge"), Path("docs/results/agent_judge")
POLICIES = {"Start\n(default weights)": None, "A: agent-judged\nCMA-ES": D / "judged_cmaes_policy.yaml",
            "B: cost function\n+ agent checks": D / "costfn_agent_policy.yaml"}
KIND_NAMES = {"head_on": "Head-on", "crossing": "Crossing", "group_toward": "Oncoming group",
              "narrow_passage": "Narrow passage", "stationary_group": "Standing group", "dense_flow": "Two-way corridor",
              "overtaking": "Overtaking"}


def main(count: int = 6) -> None:
    search = JudgedSearch(A / "exp1_judged")
    env, policy = search.stack()
    kinds = search.kinds("test_kinds") + ["overtaking"]
    scenes = scenario_set("test", kinds, 30, dt=search.j["dt"], duration=search.j["duration"])
    batch = collate(scenes, search.device)
    values = {name: {**search.state["base"], **(load_policy_file(path)[1] if path else {})} for name, path in POLICIES.items()}
    J, gap, outcome = [], [], []
    for name, v in values.items():
        run = rollout(env, batch, policy, policy.parameter_tensor(v, batch.size, search.device))
        m = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
        J.append(PRINCIPLES_J(m).numpy())
        gap.append(torch.nan_to_num(m["DHmin"], nan=9.0).numpy())
        outcome.append(["collision" if c else "reached" if s else "timed out" for s, c in zip(m["S"], m["C"])])
    J, gap = np.mean(J, 0), np.min(gap, 0)
    order = np.lexsort((gap, J))
    chosen, per_kind = [], {}
    for i in order:
        if per_kind.get(scenes[i].kind, 0) < 2:
            chosen.append(int(i))
            per_kind[scenes[i].kind] = per_kind.get(scenes[i].kind, 0) + 1
        if len(chosen) == count:
            break
    hard = [scenes[i] for i in chosen]
    fixed = collate(hard, search.device)
    cf = rollout(env, fixed, robot_present=False)
    snaps = []
    for v in values.values():
        run, plans = recorded_rollout(policy, env, fixed, v, search.device)
        snaps.append(snapshot(fixed, run, plans, cf))
    cols = [f"{KIND_NAMES[s.kind]} (#{s.seed - 2_000_000})" for s in hard]
    out = D / "hard_scenes"
    out.mkdir(exist_ok=True)
    animate_grid(snaps, list(POLICIES), cols, out / "hard_scenes.gif",
                 title="Hardest held-out scenes: start weights vs A (agent-judged) vs B (cost function + agent)")
    for c in range(len(hard)):
        one = [{k: (v[[c]] if hasattr(v, "shape") and v.ndim and len(v) == len(hard) else v) for k, v in s.items()}
               for s in snaps]
        for s, full in zip(one, snaps):
            s["names"], s["plans"] = [full["names"][c]], [(p[[c]], st[[c]]) for p, st in full["plans"]]
        animate_grid(one, list(POLICIES), [cols[c]], out / f"{hard[c].name}.gif", title=cols[c])
    summary = {s.name: {name.replace("\n", " "): outcome[p][i] for p, name in enumerate(POLICIES)}
               for s, i in zip(hard, chosen)}
    (out / "scenes.json").write_text(json.dumps({"selection": "lowest mean Principles J over the 3 policies, <= 2 per kind",
                                                 "scenes": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
