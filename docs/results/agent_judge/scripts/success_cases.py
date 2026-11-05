"""Success cases of the agent-judged CMA-ES policy, with the other policies on the same scenes.

Success case = a held-out test scene where the agent-judged policy reaches the goal without coming within
0.45 m of anyone, ranked by how far its hand-crafted reward beats the average of the 10 CMA-ES seeds there;
up to six, at most two per kind. Same layout and outputs as ``failure_cases.py``.

    python docs/results/agent_judge/scripts/success_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "simplenavsim/scripts")
from evaluate import recorded_rollout, snapshot  # noqa: E402
from failure_cases import JUDGED, KIND_NAMES, SEEDS, A, D, outcomes  # noqa: E402
from judged_cmaes import JudgedSearch  # noqa: E402

from simplenavsim.core.config import CONFIG_DIR, load_config  # noqa: E402
from simplenavsim.core.metrics import all_metrics  # noqa: E402
from simplenavsim.core.rollout import rollout  # noqa: E402
from simplenavsim.core.scenarios import collate, scenario_set  # noqa: E402
from simplenavsim.optimization.cmaes import random_weights  # noqa: E402
from simplenavsim.policies.base import load_policy_file  # noqa: E402
from simplenavsim.visualization.plots import animate_grid  # noqa: E402


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
    run = rollout(env, batch, policy, policy.parameter_tensor(shown[JUDGED], batch.size, device))
    m = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
    judged, reward = outcomes(env, policy, batch, shown[JUDGED], device)
    close = m["intimate_violation_rate"].numpy() > 0
    cmaes = np.mean([outcomes(env, policy, batch, {**base, **load_policy_file(SEEDS.format(s))[1]}, device)[1]
                     for s in range(10)], 0)
    good = [i for i, o in enumerate(judged) if o == "reached" and not close[i]]
    chosen, per_kind = [], {}
    for i in sorted(good, key=lambda i: cmaes[i] - reward[i]):
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
        r, plans = recorded_rollout(policy, env, fixed, values, device)
        snaps.append(snapshot(fixed, r, plans, cf))
        shown_outcomes[name] = ["collision" if c else "reached" if s else "timed out"
                                for s, c in zip(r.success.tolist(), (r.ped_collision | r.obstacle_collision).tolist())]
    cols = [f"{KIND_NAMES[s.kind]} (#{s.seed - 2_000_000})" for s in cases]
    out = D / "success_cases"
    out.mkdir(exist_ok=True)
    animate_grid(snaps, list(shown), cols, out / "success_cases.gif",
                 title="Success cases of the agent-judged CMA-ES policy (held-out scenes), other policies on the same scenes")
    for c in range(len(cases)):
        one = [{k: (v[[c]] if hasattr(v, "shape") and v.ndim and len(v) == len(cases) else v) for k, v in s.items()}
               for s in snaps]
        for s, full in zip(one, snaps):
            s["names"], s["plans"] = [full["names"][c]], [(p[[c]], st[[c]]) for p, st in full["plans"]]
        animate_grid(one, list(shown), [cols[c]], out / f"{cases[c].name}.gif", title=f"Success case: {cols[c]}")
    summary = {s.name: {**{name.replace("\n", " "): shown_outcomes[name][c] for name in shown},
                        "hand-crafted reward: agent-judged": round(float(reward[i]), 3),
                        "hand-crafted reward: CMA-ES (10-seed mean)": round(float(cmaes[i]), 3),
                        **{f"{group} reaching the goal": int(reached[group][c]) for group in reached}}
               for c, (s, i) in enumerate(zip(cases, chosen))}
    (out / "scenes.json").write_text(json.dumps({
        "selection": "held-out scenes where the agent-judged policy reaches the goal without coming within 0.45 m of "
                     "anyone, largest margin of its hand-crafted reward over the 10-seed CMA-ES mean first, <= 2 per kind",
        "scenes": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
