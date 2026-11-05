"""Final evaluation of the agent-judge experiments: held-out test scenes, every metric, both cost functions,
judge labels on one picture per kind, a GIF.

    python runs/simplenavsim/agent_judge/final_eval.py start=<yaml or 'defaults'> exp1=<yaml> exp2=<yaml> ...
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "simplenavsim/scripts")
from judged_cmaes import PRINCIPLES_J, SIM_KEYS, JudgedSearch  # noqa: E402

from simplenavsim.core.metrics import all_metrics, summarize  # noqa: E402
from simplenavsim.core.rollout import rollout  # noqa: E402
from simplenavsim.core.scenarios import collate, scenario_set  # noqa: E402
from simplenavsim.optimization.judge import ClaudeCodeJudge  # noqa: E402
from simplenavsim.policies.base import load_policy_file  # noqa: E402
from simplenavsim.visualization.plots import animate_grid  # noqa: E402

A = Path("runs/simplenavsim/agent_judge")
OUT = A / "final"
PER_KIND = 30


def main() -> None:
    specs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    search = JudgedSearch(A / "exp1_judged")            # its config: the setting both experiments used
    search.out_dir = OUT
    OUT.mkdir(exist_ok=True)
    env, policy = search.stack()
    policies = {name: {**search.state["base"], **({} if path == "defaults" else load_policy_file(path)[1])}
                for name, path in specs.items()}
    kinds = search.kinds("test_kinds") + ["overtaking"]
    scenes = scenario_set("test", kinds, PER_KIND, dt=search.j["dt"], duration=search.j["duration"])
    batch = collate(scenes, search.device)
    results = {}
    for name, values in policies.items():
        run = rollout(env, batch, policy, policy.parameter_tensor(values, batch.size, search.device))
        m = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
        m["hand_objective"], m["principles_J"] = search.objective(m), PRINCIPLES_J(m)
        per_seed = {k: v for k, v in summarize(m).items() if k in SIM_KEYS + ("hand_objective", "principles_J")}
        results[name] = {"weights": values, "all": per_seed}
        print(name, json.dumps({k: round(per_seed[k], 3) for k in ("S", "C", "SPL", "SC", "principles_J",
                                                                     "hand_objective")}), flush=True)
    # One held-out scene per kind: pictures (the judge's view), a judged sheet, a GIF.
    fixed = [next(s for s in scenes if s.kind == k) for k in kinds]
    records, sim = search.simulate(list(policies.values()), fixed)
    episodes = search.render(OUT, records, sim, list(policies), fixed, "test_", "Held-out test scenes (rows: policies)")
    verdicts = ClaudeCodeJudge(model="sonnet", timeout=900).label(
        [{**e, "image": str(OUT / e["image"])} for e in episodes], OUT)
    for name in policies:
        labels = [verdicts[e["id"]] for e in episodes if e["candidate"] == name]
        results[name]["judge_test"] = {"fitness": float(np.mean([v.fitness for v in labels])),
                                       "labels": {e["scene"]: [verdicts[e["id"]].label, verdicts[e["id"]].reason]
                                                  for e in episodes if e["candidate"] == name}}
    snaps = []
    S = len(fixed)
    n = lambda t: t.detach().float().cpu().numpy() if t.dtype.is_floating_point else t.detach().cpu().numpy()
    for run, b, count in search._last:
        for r in range(count):
            rows = slice(r * S, (r + 1) * S)
            snaps.append({"names": b.names[rows], "dt": run.dt, "robot_pose": n(run.robot_pose[rows]),
                          "robot_v": n(run.robot_v[rows]), "ped_pos": n(run.ped_pos[rows]),
                          "ped_active": n(run.ped_active[rows]), "ped_mask": n(b.ped_mask[rows]),
                          "ped_radius": n(b.ped_radius[rows]), "segments": n(b.segments[rows]),
                          "segment_mask": n(b.segment_mask[rows]), "goal": n(b.robot_goal[rows]),
                          "start": n(b.robot_start[rows]), "end_step": n(run.end_step[rows]),
                          "success": n(run.success[rows]),
                          "collision": n((run.ped_collision | run.obstacle_collision)[rows]), "plans": []})
    animate_grid(snaps, list(policies), [s.kind.replace("_", " ") for s in fixed], OUT / "policies.gif",
                 title="Starting weights vs agent-judged CMA-ES vs cost-function CMA-ES (held-out test scenes)")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=lambda x: None if isinstance(x, float)
                                                 and math.isnan(x) else x))
    print("saved", OUT)


if __name__ == "__main__":
    with torch.no_grad():
        main()
