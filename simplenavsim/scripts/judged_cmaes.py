#!/usr/bin/env python3
"""CMA-ES with a judge's labels as fitness: the AutoEval loop, with an agent in place of PaliGemma.

Each generation: CMA-ES proposes candidate weight sets, every candidate drives the same scenes,
every episode is rendered top-down, a judge labels each picture (``judge.LABELS``), and the mean
label fitness of a candidate is what CMA-ES maximises. The hand-written objective J is computed
for every episode too, only so the two can be compared (``report``).

The state lives in ``--out-dir`` (``state.pkl``), so the loop can stop between proposing and
scoring and an agent or a person can do the labelling:

    S=simplenavsim/scripts/judged_cmaes.py C=simplenavsim/configs/judged_cmaes_poc.yaml
    python $S ask   --config $C --out-dir runs/judged    # roll out, render gen_01/sheet.png, pending.json
    python $S judge --out-dir runs/judged --judge claude-code   # an agent writes gen_01/labels.json
    python $S tell  --out-dir runs/judged                # labels.json -> fitness -> CMA-ES update
    python $S run   --config $C --out-dir runs/judged --judge claude-code   # ask/judge/tell for every generation
    python $S report --out-dir runs/judged               # convergence, judge vs J, test scenes, GIF

Only ``ask``, ``run`` and ``report`` simulate (and load STARS on ``--device``); ``judge`` and ``tell``
need neither a GPU nor the weights, so on a cluster whose compute nodes have no internet the
rollouts run on a GPU node and the judging on a login node (``scripts/horizon/judged_loop.sh``).
With ``common_crowd`` (default) each candidate is its own batch, so every candidate of a
generation meets the same crowd draws (the STARS crowd draws its CVAE intents per batch row).

Per generation ``gen_XX/``: ``sheet.png`` (candidates x scenes, what the judge sees),
``episodes/*.png`` (one picture per episode), ``pending.json``, ``sim.json`` (simulator metrics and J,
hidden from the judge), ``labels.json``, ``scored_sheet.png`` (the sheet with the labels).
Across generations: ``labels.jsonl`` (parameters -> label, one line per episode),
``generations.jsonl``, ``mean_policy.yaml``, ``best_policy.yaml``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cma
import numpy as np
from simplenavsim.core.config import CONFIG_DIR, device_from, load_config, make_controller, make_env, make_predictor
from simplenavsim.core.metrics import all_metrics
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import collate, make_scenario
from simplenavsim.optimization.evaluator import Objective
from simplenavsim.optimization.judge import FITNESS, LABELS, RUBRIC, episode_records, make_judge, render_episode, \
    render_sheet, time_scale, view_for
from simplenavsim.policies.base import load_policy_file, save_policy

DEFAULT_CONFIG = CONFIG_DIR / "judged_cmaes_poc.yaml"
SIM_KEYS = ("success", "collision", "timeout", "min_ped_clearance", "intimate_violation_rate", "path_efficiency",
            "normalized_time", "loops", "command_smoothness", "robot_influence", "disruption_local",
            # Principles (Francis et al. Table I), core.metrics.principles_metrics
            "S", "C", "HC", "WC", "TO", "FP", "ST", "T", "PL", "SPL", "Vavg", "Vmax", "Aavg", "Amin", "Amax",
            "Javg", "Jmin", "Jmax", "CDmin", "CDavg", "DHmin", "SC", "TTC")
# The Principles objective of cmaes_crowd_principles.yaml, logged beside the config's objective (not fitness).
PRINCIPLES_J = Objective(weights={"SPL": 1.0, "C": -2.0, "SC": 1.0, "normalized_time": -0.3})
SPLIT_SEED = {"train": 0, "test": 2_000_000}


class JudgedSearch:
    def __init__(self, out_dir: Path, config_path: Path | None = None, *, random_init: int | None = None,
                 initial_policy: str | None = None, device: str | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        saved = self.out_dir / "config.json"
        if config_path is None and not saved.exists():
            config_path = DEFAULT_CONFIG
        if config_path is not None:
            config = load_config(config_path)
            if saved.exists() and json.loads(saved.read_text()) != json.loads(json.dumps(config)):
                raise ValueError(f"{self.out_dir} was started with a different config ({saved})")
            saved.write_text(json.dumps(config, indent=1))
        self.config = json.loads(saved.read_text())
        self.j = self.config["judged"]
        self.device = device_from(device or self.j.get("device") or (self.config.get("simulator") or {}).get("device"))
        pc = self.config["policy"]
        # The parameter specs and policy files need only the controller; the simulator, the crowd model
        # and the predictor (STARS weights, GPU) are built on first use, so judge / tell run anywhere.
        self.policy = make_controller(pc, pc.get("name", "social"))
        self._built, self._pool, self._last = None, False, []
        self.objective = Objective(**self.config["objective"])
        specs = {p.name: p for p in self.policy.parameters}
        self.specs = [specs[name] for name in self.j["optimize"]]
        self.base = {**self.policy.default_parameters(), **(pc.get("parameters") or {}), **self.j.get("initial", {})}
        if initial_policy:
            self.base.update(load_policy_file(initial_policy)[1])
        if random_init is not None:
            rng = np.random.default_rng(random_init)
            self.base.update({s.name: s.from_unit(float(rng.random())) for s in self.specs})
        self.state = self._load()

    # ------------------------------------------------------------------ state

    @property
    def state_path(self) -> Path:
        return self.out_dir / "state.pkl"

    def _load(self) -> dict:
        if self.state_path.exists():
            return pickle.loads(self.state_path.read_bytes())
        es = cma.CMAEvolutionStrategy([s.to_unit(self.base[s.name]) for s in self.specs], self.j["sigma0"],
                                      {"popsize": self.j["population_size"], "bounds": [0.0, 1.0],
                                       "seed": self.j["seed"], "verbose": -9})
        state = {"es": es, "generation": 0, "pending": None, "base": dict(self.base), "best": None}
        self._save(state)
        return state

    def _save(self, state=None) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(state or self.state))
        tmp.replace(self.state_path)

    def decode(self, unit) -> dict[str, float]:
        values = dict(self.state["base"])
        values.update({s.name: s.from_unit(float(u)) for s, u in zip(self.specs, unit)})
        return values

    def gen_dir(self, g: int) -> Path:
        return self.out_dir / f"gen_{g:02d}"

    # ------------------------------------------------------------------ simulation

    def stack(self):
        """``(env, controller)`` on ``self.device``, built once (loads the STARS weights if configured)."""
        if self._built is None:
            pedestrians = {k: dict(v) if isinstance(v, dict) else v for k, v in self.config["pedestrians"].items()}
            if "stars_crowd" in pedestrians:
                pedestrians["stars_crowd"]["device"] = str(self.device)
            env = make_env(self.config["simulator"], pedestrians)
            pc = self.config["policy"]
            policy = make_controller(pc, pc.get("name", "social"), env.robot,
                                     make_predictor(pc.get("predictor"), self.device))
            self._built = (env, policy)
        return self._built

    def pool(self):
        """The STEPS RecordingPool for ``recorded`` scenes (``recorded:`` in the config), or None."""
        if self._pool is False:
            from simplenavsim.cli import recording_pool

            self._pool = recording_pool(self.config) if "recorded" in self.j["kinds"] + self.j.get("test_kinds", []) \
                else None
        return self._pool

    def kinds(self, key: str) -> list[str]:
        kinds = list(self.j[key])
        return kinds if self.pool() is not None else [k for k in kinds if k != "recorded"]

    def scenes(self, g: int):
        kinds, S = self.kinds("kinds"), self.j["scenes_per_generation"]
        return [make_scenario(kinds[(S * (g - 1) + i) % len(kinds)], SPLIT_SEED["train"] + g - 1, dt=self.j["dt"],
                              duration=self.j["duration"], pool=self.pool(), split="train") for i in range(S)]

    def simulate(self, candidates: list[dict], scenes) -> tuple[list[dict], list[dict]]:
        """Every candidate on every scene: per-row (candidate-major) picture records and simulator metrics.
        With ``common_crowd`` each candidate is its own batch, so all meet the same crowd draws."""
        env, policy = self.stack()
        base = collate(scenes, self.device)
        groups = [[c] for c in candidates] if self.j.get("common_crowd", True) else [candidates]
        records, sim, self._last = [], [], []
        for group in groups:
            batch = base.repeat(len(group))
            rows = [values for values in group for _ in scenes]
            run = rollout(env, batch, policy, policy.parameter_tensor(rows, batch.size, self.device))
            metrics = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
            hand, principles = self.objective(metrics), PRINCIPLES_J(metrics)
            sim += [{**{k: float(metrics[k][b]) for k in SIM_KEYS if k in metrics}, "hand_objective": float(hand[b]),
                     "principles_J": float(principles[b])} for b in range(batch.size)]
            records += episode_records(run, batch, env.robot.radius, env.sim.goal_tolerance)
            self._last.append((run, batch, len(group)))
        return records, sim

    def render(self, out: Path, records, sim, labels: list[str], scenes, prefix: str, title: str) -> list[dict]:
        """Episode pictures and the contact sheet; returns the episode list the judge gets."""
        (out / "episodes").mkdir(parents=True, exist_ok=True)
        S = len(scenes)
        grid = [records[r * S:(r + 1) * S] for r in range(len(labels))]
        views = [view_for([grid[r][c] for r in range(len(labels))]) for c in range(S)]
        t_max = time_scale(records)
        episodes = []
        for r, label in enumerate(labels):
            for c, scene in enumerate(scenes):
                eid = f"{prefix}{label}_{scene.name}"
                image = out / "episodes" / f"{eid}.png"
                render_episode(grid[r][c], image, title=f"{eid}", view=views[c], duration=t_max)
                episodes.append({"id": eid, "candidate": label, "scene": scene.name, "kind": scene.kind,
                                 "image": str(image.relative_to(self.out_dir)), "sim": sim[r * S + c]})
        render_sheet(grid, labels, [s.name for s in scenes], out / "sheet.png", title=title, duration=t_max)
        with open(out / "records.pkl", "wb") as stream:
            pickle.dump({"grid": grid, "labels": labels, "scenes": [s.name for s in scenes]}, stream)
        return episodes

    # ------------------------------------------------------------------ ask / tell

    def ask(self) -> Path:
        if self.state["pending"]:
            raise RuntimeError(f"generation {self.state['pending']['generation']} is waiting for labels: run tell")
        g = self.state["generation"] + 1
        if g > self.j["generations"]:
            raise RuntimeError(f"all {self.j['generations']} generations are done")
        es = self.state["es"]
        solutions = es.ask()
        candidates = [self.decode(x) for x in solutions]
        scenes = self.scenes(g)
        records, sim = self.simulate(candidates, scenes)
        out = self.gen_dir(g)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        labels = [f"c{i}" for i in range(len(candidates))]
        episodes = self.render(out, records, sim, labels, scenes, f"g{g:02d}_",
                               f"Generation {g}: candidates (rows) x scenes (columns)")
        (out / "sim.json").write_text(json.dumps({e["id"]: e["sim"] for e in episodes}, indent=1))
        (out / "pending.json").write_text(json.dumps({
            "generation": g, "labels": {k: {"fitness": v[0], "meaning": v[1]} for k, v in LABELS.items()},
            "candidates": [{"candidate": labels[i], "parameters": c} for i, c in enumerate(candidates)],
            "episodes": [{k: e[k] for k in ("id", "candidate", "scene", "kind", "image")} for e in episodes]},
            indent=1))
        (self.out_dir / "rubric.md").write_text(RUBRIC + "\n")
        self.state["pending"] = {"generation": g, "solutions": [np.asarray(x) for x in solutions],
                                 "candidates": candidates}
        self._save()
        return out

    def pending_episodes(self) -> tuple[int, Path, dict, dict, list[dict]]:
        pending = self.state["pending"]
        if not pending:
            raise RuntimeError("nothing to score: run ask first")
        g, out = pending["generation"], self.gen_dir(pending["generation"])
        meta, sim = json.loads((out / "pending.json").read_text()), json.loads((out / "sim.json").read_text())
        episodes = [{**e, "sim": sim[e["id"]], "image": str(self.out_dir / e["image"])} for e in meta["episodes"]]
        return g, out, meta, sim, episodes

    def judge(self, judge) -> Path:
        """Label the pending generation and write its labels.json (``tell --judge file`` then reads it)."""
        _, out, _, _, episodes = self.pending_episodes()
        verdicts = judge.label(episodes, out)
        (out / "labels.json").write_text(json.dumps({k: {"label": v.label, "reason": v.reason}
                                                     for k, v in verdicts.items()}, indent=1))
        return out / "labels.json"

    def tell(self, judge) -> dict:
        pending = self.state["pending"]
        g, out, meta, sim, episodes = self.pending_episodes()
        verdicts = judge.label(episodes, out)
        names = [c["candidate"] for c in meta["candidates"]]
        fitness = np.array([np.mean([verdicts[e["id"]].fitness for e in episodes if e["candidate"] == n])
                            for n in names])
        hand = np.array([np.mean([sim[e["id"]]["hand_objective"] for e in episodes if e["candidate"] == n])
                         for n in names])
        es = self.state["es"]
        es.tell(pending["solutions"], (-fitness).tolist())
        with open(self.out_dir / "labels.jsonl", "a") as stream:
            for e in episodes:
                v, i = verdicts[e["id"]], names.index(e["candidate"])
                stream.write(json.dumps({"generation": g, "episode": e["id"], "candidate": e["candidate"],
                                         "scene": e["scene"], "kind": e["kind"], "judge": judge.name,
                                         "parameters": pending["candidates"][i], "label": v.label,
                                         "fitness": v.fitness, "reason": v.reason, "sim": e["sim"]}) + "\n")
        (out / "labels.json").write_text(json.dumps({e["id"]: {"label": verdicts[e["id"]].label,
                                                               "reason": verdicts[e["id"]].reason}
                                                     for e in episodes}, indent=1))
        best = int(fitness.argmax())
        mean_values = self.decode(es.mean)
        record = {"generation": g, "judge": judge.name, "fitness": fitness.tolist(), "fitness_best": float(fitness.max()),
                  "fitness_mean": float(fitness.mean()), "hand_objective": hand.tolist(),
                  "hand_objective_mean": float(hand.mean()), "sigma": float(es.sigma),
                  "best_candidate": names[best], "best_parameters": pending["candidates"][best],
                  "mean_parameters": mean_values}
        with open(self.out_dir / "generations.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.state["best"] is None or fitness[best] >= self.state["best"]["fitness"]:
            self.state["best"] = {"generation": g, "candidate": names[best], "fitness": float(fitness[best]),
                                  "values": pending["candidates"][best]}
            save_policy(self.out_dir / "best_policy.yaml", self.policy, pending["candidates"][best],
                        judged_fitness=float(fitness[best]), generation=g, candidate=names[best])
        save_policy(self.out_dir / "mean_policy.yaml", self.policy, mean_values, generation=g)
        self._scored_sheet(out, verdicts, fitness, names)
        self.state["generation"], self.state["pending"] = g, None
        self._save()
        return record

    def _scored_sheet(self, out: Path, verdicts, fitness, names) -> None:
        """The generation's sheet again, every cell titled with its verdict (what the judge decided)."""
        data = pickle.loads((out / "records.pkl").read_bytes())
        g = int(out.name.split("_")[1])
        colours = {"excellent": "#1a7f37", "acceptable": "#7a6a00", "stalled": "#b35900", "unsafe": "#c0392b"}
        titles, tints = [], []
        for r, n in enumerate(names):
            v = [verdicts[f"g{g:02d}_{n}_{scene}"] for scene in data["scenes"]]
            titles.append([f"{n} (fitness {fitness[r]:+.2f}) | {scene}\n{x.label} ({x.fitness:+.1f})"
                           for scene, x in zip(data["scenes"], v)])
            tints.append([colours[x.label] for x in v])
        render_sheet(data["grid"], names, data["scenes"], out / "scored_sheet.png", cell_titles=titles,
                     cell_colours=tints, title=f"Generation {g}: the judge's labels (row fitness = mean over its scenes)")

    def run(self, judge, generations: int | None = None) -> None:
        target = min(generations or self.j["generations"], self.j["generations"])
        while self.state["generation"] < target:
            if not self.state["pending"]:
                out = self.ask()
                print(f"asked generation {self.state['pending']['generation']}: {out / 'sheet.png'}")
            record = self.tell(judge)
            print(f"gen {record['generation']:2d}  fitness best {record['fitness_best']:+.2f} "
                  f"mean {record['fitness_mean']:+.2f}  (hand J mean {record['hand_objective_mean']:+.3f})  "
                  f"sigma {record['sigma']:.3f}")

    # ------------------------------------------------------------------ test scenes (report)

    def test(self, policies: dict[str, dict]) -> Path:
        """The given weight sets on one held-out test scene per ``test_kinds``: pictures, sheet, GIF."""
        from simplenavsim.visualization.plots import animate_grid

        out = self.out_dir / "test"
        out.mkdir(exist_ok=True)
        scenes = [make_scenario(k, SPLIT_SEED["test"], dt=self.j["dt"], duration=self.j["duration"], pool=self.pool(),
                                split="test") for k in self.kinds("test_kinds")]
        labels = list(policies)
        records, sim = self.simulate(list(policies.values()), scenes)
        episodes = self.render(out, records, sim, labels, scenes, "test_",
                               "Held-out test scenes: " + ", ".join(labels) + " (rows)")
        (out / "sim.json").write_text(json.dumps({e["id"]: e["sim"] for e in episodes}, indent=1))
        (out / "pending.json").write_text(json.dumps({"episodes": [{k: e[k] for k in ("id", "candidate", "scene", "image")}
                                                                    for e in episodes],
                                                      "policies": policies}, indent=1))
        n = lambda t: t.detach().float().cpu().numpy() if t.dtype.is_floating_point else t.detach().cpu().numpy()
        S = len(scenes)
        snaps = []
        for run, batch, count in self._last:
            for r in range(count):
                rows = slice(r * S, (r + 1) * S)
                snaps.append({"names": batch.names[rows], "dt": run.dt, "robot_pose": n(run.robot_pose[rows]),
                              "robot_v": n(run.robot_v[rows]), "ped_pos": n(run.ped_pos[rows]),
                              "ped_active": n(run.ped_active[rows]), "ped_mask": n(batch.ped_mask[rows]),
                              "ped_radius": n(batch.ped_radius[rows]), "segments": n(batch.segments[rows]),
                              "segment_mask": n(batch.segment_mask[rows]), "goal": n(batch.robot_goal[rows]),
                              "start": n(batch.robot_start[rows]), "end_step": n(run.end_step[rows]),
                              "success": n(run.success[rows]),
                              "collision": n((run.ped_collision | run.obstacle_collision)[rows]), "plans": []})
        animate_grid(snaps, labels, [s.kind for s in scenes], out / "test_rollouts.gif",
                     title="Before vs after judged CMA-ES (held-out test scenes)", every=4, fps=10)
        return out


# ---------------------------------------------------------------------------------------------- report

def spearman(a, b) -> float:
    """Spearman rank correlation with average ranks for ties."""
    a, b = np.asarray(a, float), np.asarray(b, float)

    def avg_rank(x):
        order, ranks = np.argsort(x, kind="stable"), np.empty(len(x))
        xs = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return ranks
    ra, rb = avg_rank(a), avg_rank(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")


def report(search: JudgedSearch) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = search.out_dir
    gens = [json.loads(l) for l in (out / "generations.jsonl").read_text().splitlines() if l.strip()]
    rows = [json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines() if l.strip()]
    g = [r["generation"] for r in gens]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for r in gens:
        ax.scatter([r["generation"]] * len(r["fitness"]), r["fitness"], color="#9aa7b8", s=14, zorder=2)
    ax.plot(g, [r["fitness_best"] for r in gens], "-o", color="#1f4e9c", label="generation best")
    ax.plot(g, [r["fitness_mean"] for r in gens], "-s", color="#d9822b", label="generation mean")
    ax.set(xlabel="generation", ylabel="judged fitness (mean label score)", ylim=(-1.05, 1.05),
           title="Fitness = the judge's labels (dots: each candidate)")
    twin = ax.twinx()
    twin.plot(g, [r["sigma"] for r in gens], ":", color="gray")
    twin.set_ylabel("CMA-ES step size", color="gray")
    ax.legend(loc="lower right", fontsize=8), ax.grid(alpha=0.3)
    ax = axes[1]
    order = list(LABELS)
    colours = {"excellent": "#1a7f37", "acceptable": "#c9b400", "stalled": "#e08a2c", "unsafe": "#c0392b"}
    bottom = np.zeros(len(g))
    for label in order:
        counts = np.array([sum(1 for r in rows if r["generation"] == gg and r["label"] == label) for gg in g], float)
        ax.bar(g, counts, bottom=bottom, color=colours[label], label=label)
        bottom += counts
    ax.set(xlabel="generation", ylabel="episodes", title="Labels per generation")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=130)
    plt.close(fig)

    fit, hand = [r["fitness"] for r in rows], [r["sim"]["hand_objective"] for r in rows]
    outcome = lambda s: "collision" if s["collision"] else "reached goal" if s["success"] else "timed out"
    table = {label: {o: sum(1 for r in rows if r["label"] == label and outcome(r["sim"]) == o)
                     for o in ("reached goal", "timed out", "collision")} for label in order}
    per_candidate = [(np.mean(r["fitness"][i:i + 1]), r["hand_objective"][i]) for r in gens for i in range(len(r["fitness"]))]
    summary = {
        "generations": len(gens), "episodes_labelled": len(rows),
        "spearman_episode_label_vs_hand_J": spearman(fit, hand),
        "spearman_candidate_fitness_vs_hand_J": spearman([a for a, _ in per_candidate], [b for _, b in per_candidate]),
        "label_vs_simulator_outcome": table,
        "first_generation": {k: gens[0][k] for k in ("fitness_mean", "fitness_best", "hand_objective_mean")},
        "last_generation": {k: gens[-1][k] for k in ("fitness_mean", "fitness_best", "hand_objective_mean")},
        "best": search.state["best"], "final_mean_parameters": search.decode(search.state["es"].mean),
        "initial_parameters": search.state["base"],
    }
    policies = {"initial": search.state["base"], "final_mean": summary["final_mean_parameters"],
                "best_candidate": search.state["best"]["values"]}
    test_dir = search.test(policies)
    test_sim = json.loads((test_dir / "sim.json").read_text())
    summary["test_hand_J"] = {name: float(np.mean([v["hand_objective"] for k, v in test_sim.items()
                                                    if k.startswith(f"test_{name}_")])) for name in policies}
    labels_file = test_dir / "labels.json"
    if labels_file.exists():
        tl = json.loads(labels_file.read_text())
        summary["test_judged_fitness"] = {name: float(np.mean([FITNESS[v["label"]] for k, v in tl.items()
                                                                if k.startswith(f"test_{name}_")])) for name in policies}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("ask", "judge", "tell", "run", "report", "status"))
    parser.add_argument("--config", type=Path, help="New runs only; a run keeps the config it started with")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/simplenavsim/judged_cmaes"))
    parser.add_argument("--judge", choices=("file", "claude-code", "claude", "rule"), default=None,
                        help="judge / run: who labels (default claude-code); tell: default file (reads labels.json)")
    parser.add_argument("--model", help="Model for the claude / claude-code judge")
    parser.add_argument("--judge-timeout", type=int, default=1800, help="Seconds per generation for claude-code")
    parser.add_argument("--generations", type=int, help="run: stop after this generation")
    parser.add_argument("--device", help="cuda / cpu / auto (simulation only)")
    parser.add_argument("--policy", help="New runs only: start CMA-ES at a saved policy YAML, e.g. the v4 weights")
    parser.add_argument("--random-init", type=int, metavar="SEED", help="New runs only: start from random weights")
    args = parser.parse_args()
    search = JudgedSearch(args.out_dir, args.config, random_init=args.random_init, initial_policy=args.policy,
                          device=args.device)
    name = args.judge or ("file" if args.command == "tell" else "claude-code")
    options = {}
    if name in ("claude", "claude-code") and args.model:
        options["model"] = args.model
    if name == "claude-code":
        options["timeout"] = args.judge_timeout
    if args.command == "status":  # "<generations told> <pending generation or 0> <generations planned>"
        pending = search.state["pending"]
        print(search.state["generation"], pending["generation"] if pending else 0, search.j["generations"])
        return
    if args.command == "ask":
        out = search.ask()
        print(f"generation {search.state['pending']['generation']}: {out / 'sheet.png'}; label it with "
              f"`judge` or write {out / 'labels.json'} yourself, then `tell`")
    elif args.command == "judge":
        print(f"wrote {search.judge(make_judge(name, **options))}")
    elif args.command == "tell":
        r = search.tell(make_judge(name, **options))
        print(json.dumps({k: r[k] for k in ("generation", "fitness", "fitness_mean", "hand_objective_mean", "sigma")}))
    elif args.command == "run":
        search.run(make_judge(name, **options), args.generations)
    else:
        print(json.dumps(report(search), indent=1))


if __name__ == "__main__":
    main()
