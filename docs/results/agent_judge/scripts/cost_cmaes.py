"""Experiment 2: CMA-ES on an editable cost function, checked by an agent judge every so often.

Each generation: CMA-ES proposes candidates, every candidate drives the same scenes (JudgedSearch.ask:
same simulation, pictures and logged metrics as the judged run), and fitness is the cost function in
``objective.json`` ({"weights": {metric: weight}}) applied to every episode's logged metrics. About every
``--judge-minutes``, the judge labels the three best candidates' episodes; the run then waits up to
``--revise-minutes`` for ``revised_<generation>`` (written once objective.json has been reconsidered)
and goes on with whatever objective.json now says. Every objective in force is logged with its
generation in ``objective_history.jsonl``; ``evals.jsonl`` has the judge's labels beside the cost.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "simplenavsim/scripts")
from judged_cmaes import JudgedSearch  # noqa: E402

from simplenavsim.optimization.judge import ClaudeCodeJudge  # noqa: E402
from simplenavsim.policies.base import save_policy  # noqa: E402


def cost(sim: dict, weights: dict) -> float:
    value = 0.0
    for key, w in weights.items():
        x = sim.get(key)
        value += w * (0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else x)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--judge-minutes", type=float, default=12)
    parser.add_argument("--revise-minutes", type=float, default=6)
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()
    out = args.out_dir
    search = JudgedSearch(out)
    objective_path, history = out / "objective.json", out / "objective_history.jsonl"
    end, next_judge = time.time() + 60 * args.minutes, time.time() + 60 * args.judge_minutes
    last_weights = None
    while time.time() < end and search.state["generation"] < search.j["generations"] and not (out / "STOP").exists():
        if not search.state["pending"]:
            search.ask()
        g, gdir, meta, sim, episodes = search.pending_episodes()
        weights = json.loads(objective_path.read_text())["weights"]
        if weights != last_weights:
            with open(history, "a") as stream:
                stream.write(json.dumps({"generation": g, "time": time.strftime("%H:%M:%S"),
                                         **json.loads(objective_path.read_text())}) + "\n")
            last_weights = weights
        names = [c["candidate"] for c in meta["candidates"]]
        per = {n: [cost(sim[e["id"]], weights) for e in episodes if e["candidate"] == n] for n in names}
        fitness = np.array([np.mean(per[n]) for n in names])
        if time.time() >= next_judge:
            top = [names[i] for i in np.argsort(-fitness)[:3]]
            chosen = [e for e in episodes if e["candidate"] in top]
            verdicts = ClaudeCodeJudge(model=args.model, timeout=900).label(chosen, gdir)
            (gdir / f"judge_transcript_g{g:02d}.json").write_text((gdir / "judge_transcript.json").read_text())
            with open(out / "evals.jsonl", "a") as stream:
                for e in chosen:
                    v = verdicts[e["id"]]
                    stream.write(json.dumps({"generation": g, "episode": e["id"], "candidate": e["candidate"],
                                             "cost": cost(sim[e["id"]], weights), "label": v.label,
                                             "fitness": v.fitness, "reason": v.reason, "sim": sim[e["id"]],
                                             "weights": weights}) + "\n")
            print(f"{time.strftime('%H:%M:%S')} judged generation {g}; waiting for revised_{g}", flush=True)
            deadline = time.time() + 60 * args.revise_minutes
            while time.time() < deadline and not (out / f"revised_{g}").exists():
                time.sleep(10)
            weights = json.loads(objective_path.read_text())["weights"]   # rescore with any revision
            per = {n: [cost(sim[e["id"]], weights) for e in episodes if e["candidate"] == n] for n in names}
            fitness = np.array([np.mean(per[n]) for n in names])
            next_judge = time.time() + 60 * args.judge_minutes
        es = search.state["es"]
        es.tell(search.state["pending"]["solutions"], (-fitness).tolist())
        best = int(fitness.argmax())
        mean_values = search.decode(es.mean)
        record = {"generation": g, "fitness": fitness.tolist(), "fitness_best": float(fitness.max()),
                  "fitness_mean": float(fitness.mean()), "sigma": float(es.sigma), "weights": weights,
                  "principles_J_mean": float(np.mean([sim[e["id"]]["principles_J"] for e in episodes])),
                  "best_parameters": search.state["pending"]["candidates"][best], "mean_parameters": mean_values}
        with open(out / "generations.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")
        with open(out / "labels.jsonl", "a") as stream:   # every episode's metrics, as in the judged run
            for e in episodes:
                i = names.index(e["candidate"])
                stream.write(json.dumps({"generation": g, "episode": e["id"], "candidate": e["candidate"],
                                         "scene": e["scene"], "kind": e["kind"],
                                         "parameters": search.state["pending"]["candidates"][i],
                                         "cost": cost(sim[e["id"]], weights), "sim": sim[e["id"]]}) + "\n")
        save_policy(out / "mean_policy.yaml", search.policy, mean_values, generation=g)
        search.state["generation"], search.state["pending"] = g, None
        search._save()
        print(f"{time.strftime('%H:%M:%S')} gen {g:3d} cost best {fitness.max():+.3f} mean {fitness.mean():+.3f} "
              f"sigma {es.sigma:.3f}", flush=True)
    print(f"{time.strftime('%H:%M:%S')} exp2 stopped at generation {search.state['generation']}", flush=True)


if __name__ == "__main__":
    main()
