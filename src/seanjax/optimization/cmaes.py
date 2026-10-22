"""CMA-ES tuning of a parameterised navigation policy (``cma`` / pycma).

The search runs in the unit hypercube over the selected parameters (``optimize:``), mapped to
each parameter's bounds and scale (log or linear, ``ParameterSpec``); parameters not optimised
keep their (possibly overridden) defaults.

Per generation, every candidate is evaluated on the *same* training scenarios (``train_per_kind``
per kind). They are redrawn every ``train_refresh_every`` generations: scenario randomisation
across training, comparability within a generation, and with a refresh period > 1 less noise
for CMA-ES's step-size adaptation. Every ``validate_every`` generations the distribution mean
and the generation's best candidate are scored on a fixed validation set; the best validation
score selects ``best_policy.yaml``. The untouched test set is only used by ``evaluate.py``.
State (the CMA-ES object, best solution, history) is checkpointed every generation and a run
resumes from it.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cma
import numpy as np

from seanjax.core.scenarios import scenario_set
from seanjax.optimization.evaluator import Evaluator
from seanjax.policies.base import save_policy

log = logging.getLogger("seanjax.cmaes")


@dataclass
class CMAESConfig:
    optimize: list[str]
    population_size: int = 24
    num_generations: int = 100
    sigma0: float = 0.25
    seed: int = 42
    initial: dict[str, float] = field(default_factory=dict)
    kinds: list[str] = field(default_factory=lambda: ["crossing", "head_on", "overtaking", "following"])
    train_per_kind: int = 4
    train_refresh_every: int = 1  # generations sharing one training scenario draw (common random numbers)
    val_per_kind: int = 6
    validate_every: int = 5
    duration: float = 30.0
    dt: float = 0.1


class CMAESTrainer:
    def __init__(self, config: CMAESConfig, evaluator: Evaluator, out_dir, *, pool=None) -> None:
        self.config, self.evaluator, self.out_dir, self.pool = config, evaluator, Path(out_dir), pool
        self.policy = evaluator.policy
        specs = {p.name: p for p in self.policy.parameters}
        unknown = [name for name in config.optimize if name not in specs]
        if unknown:
            raise ValueError(f"Cannot optimise unknown parameters {unknown}")
        self.specs = [specs[name] for name in config.optimize]
        self.base = {**self.policy.default_parameters(), **config.initial}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.val_scenarios = self._scenarios("val", config.val_per_kind, 0)

    def _scenarios(self, split: str, per_kind: int, start: int):
        c = self.config
        return scenario_set(split, c.kinds, per_kind, start=start, dt=c.dt, duration=c.duration, pool=self.pool)

    def decode(self, unit) -> dict[str, float]:
        values = dict(self.base)
        for spec, u in zip(self.specs, np.asarray(unit, dtype=float)):
            values[spec.name] = spec.from_unit(u)
        return values

    def encode(self, values: dict[str, float]) -> list[float]:
        return [spec.to_unit(values[spec.name]) for spec in self.specs]

    # ------------------------------------------------------------------ checkpointing

    @property
    def checkpoint_path(self) -> Path:
        return self.out_dir / "checkpoint.pkl"

    def _save(self, state: dict) -> None:
        temporary = self.checkpoint_path.with_suffix(".tmp")
        with open(temporary, "wb") as stream:
            pickle.dump(state, stream)
        temporary.replace(self.checkpoint_path)

    def _log(self, record: dict) -> None:
        with open(self.out_dir / "generations.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")

    def validate(self, values: dict[str, float]) -> tuple[float, dict]:
        result = self.evaluator.evaluate([values], self.val_scenarios)[0]
        return result.objective, result.metrics

    # ------------------------------------------------------------------ main loop

    def run(self, *, resume: bool = False) -> dict:
        c = self.config
        if resume and self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as stream:
                state = pickle.load(stream)
            es = state["es"]
            log.info("resumed at generation %d", state["generation"])
        else:
            es = cma.CMAEvolutionStrategy(self.encode(self.base), c.sigma0,
                                          {"popsize": c.population_size, "bounds": [0.0, 1.0], "seed": c.seed,
                                           "verbose": -9})
            default_score, default_metrics = self.validate(self.base)
            save_policy(self.out_dir / "default_policy.yaml", self.policy, self.base, validation_objective=default_score)
            state = {"es": es, "generation": 0, "best": {"values": self.base, "val_objective": default_score,
                                                           "generation": 0, "metrics": default_metrics}}
            self._log({"generation": 0, "kind": "validation", "candidate": "default", "val_objective": default_score,
                       "val_metrics": default_metrics})
            save_policy(self.out_dir / "best_policy.yaml", self.policy, self.base, validation_objective=default_score,
                        generation=0)
        while state["generation"] < c.num_generations and not es.stop().get("maxfevals"):
            g = state["generation"] + 1
            started = time.perf_counter()
            solutions = es.ask()
            candidates = [self.decode(x) for x in solutions]
            draw = (g - 1) // max(1, c.train_refresh_every)
            scenarios = self._scenarios("train", c.train_per_kind, draw * c.train_per_kind)
            results = self.evaluator.evaluate(candidates, scenarios)
            objectives = np.array([r.objective for r in results])
            es.tell(solutions, (-objectives).tolist())
            best_index = int(objectives.argmax())
            record = {"generation": g, "kind": "train", "seconds": time.perf_counter() - started,
                      "objective_best": float(objectives.max()), "objective_mean": float(objectives.mean()),
                      "objective_median": float(np.median(objectives)), "sigma": float(es.sigma),
                      "best_parameters": candidates[best_index], "best_metrics": results[best_index].metrics,
                      "mean_parameters": self.decode(es.mean)}
            self._log(record)
            log.info("gen %3d  J best %.4f mean %.4f  sigma %.3f  (%.1fs)", g, record["objective_best"],
                     record["objective_mean"], es.sigma, record["seconds"])
            if g % c.validate_every == 0 or g == c.num_generations:
                for label, values in (("mean", self.decode(es.mean)), ("generation_best", candidates[best_index])):
                    score, metrics = self.validate(values)
                    self._log({"generation": g, "kind": "validation", "candidate": label, "val_objective": score,
                               "parameters": values, "val_metrics": metrics})
                    log.info("gen %3d  validation %-15s J %.4f", g, label, score)
                    if score > state["best"]["val_objective"]:
                        state["best"] = {"values": values, "val_objective": score, "generation": g, "metrics": metrics}
                        save_policy(self.out_dir / "best_policy.yaml", self.policy, values, validation_objective=score,
                                    generation=g, source=label)
            state["generation"] = g
            self._save(state)
        summary = {"generations": state["generation"], "best": state["best"],
                   "optimized": [s.name for s in self.specs]}
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
