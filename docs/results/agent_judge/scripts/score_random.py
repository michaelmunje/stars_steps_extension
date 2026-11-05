"""Score the seed study's 10 random weight sets (the CMA-ES starting points) on the 210 held-out scenes of the
CPU setting, and add them to cost_comparison.json."""
import json, sys
import torch
sys.path.insert(0, "simplenavsim/scripts")
from judged_cmaes import PRINCIPLES_J, JudgedSearch
from simplenavsim.core.config import CONFIG_DIR, load_config
from simplenavsim.core.metrics import all_metrics, summarize
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import collate, scenario_set
from simplenavsim.optimization.cmaes import random_weights

D = "docs/results/agent_judge/"
s = JudgedSearch("runs/simplenavsim/agent_judge/exp1_judged")
env, policy = s.stack()
batch = collate(scenario_set("test", s.kinds("test_kinds") + ["overtaking"], 30, dt=s.j["dt"], duration=s.j["duration"]), s.device)
names = load_config(CONFIG_DIR / "cmaes_crowd_principles.yaml")["cmaes"]["optimize"]
out = json.load(open(D + "cost_comparison.json"))
with torch.no_grad():
    for i in range(10):
        v = {**policy.default_parameters(), **random_weights(policy.parameters, names, i)}
        run = rollout(env, batch, policy, policy.parameter_tensor(v, batch.size, s.device))
        m = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
        m["pj"] = PRINCIPLES_J(m)
        out[f"random s{i}"] = a = summarize(m)
        print(f"random s{i}  J {a['pj']:.3f}  S {a['S']:.3f}  C {a['C']:.3f}  <0.45m {a['intimate_violation_rate']:.3f}", flush=True)
json.dump(out, open(D + "cost_comparison.json", "w"), indent=1)
