"""Batched policy evaluation and the configurable optimisation objective.

``Evaluator.evaluate`` runs every (parameter set, scenario) pair as one row of a single batched
simulation, so a whole CMA-ES population is evaluated in parallel (chunked by
``max_envs`` to bound memory). Robot-absent counterfactual rollouts do not depend on the policy,
so they are computed once per scenario set and cached.

Objective per episode (weights from YAML, ``objective:``):

    J = w_success * success - w_collision * collision - w_disruption * <disruption_metric>
        - w_time * normalized_time - w_comfort * <comfort_metric>
        - w_detour * (1 - path_efficiency) - w_smoothness * command_smoothness

A candidate's J is the mean over its scenarios; CMA-ES minimises -J.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from seanjax.core.metrics import all_metrics, summarize
from seanjax.core.rollout import Rollout, rollout
from seanjax.core.scenarios import Scenario, collate


@dataclass
class Objective:
    success: float = 1.0
    collision: float = 2.0
    disruption: float = 1.0
    time: float = 0.3
    comfort: float = 0.5
    detour: float = 0.0          # weight on (1 - path_efficiency)
    smoothness: float = 0.0      # weight on command_smoothness
    disruption_metric: str = "disruption_displacement"
    comfort_metric: str = "personal_violation_rate"

    def __call__(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        zero = torch.zeros_like(metrics["success"])
        disruption = metrics.get(self.disruption_metric, zero)
        return (self.success * metrics["success"] - self.collision * metrics["collision"]
                - self.disruption * torch.nan_to_num(disruption) - self.time * metrics["normalized_time"]
                - self.comfort * torch.nan_to_num(metrics[self.comfort_metric])
                - self.detour * (1.0 - metrics["path_efficiency"])
                - self.smoothness * torch.nan_to_num(metrics["command_smoothness"]))


@dataclass
class CandidateResult:
    parameters: dict[str, float]
    objective: float
    metrics: dict[str, float]
    per_scenario_objective: list[float]


def _repeat_rollout(run: Rollout, times: int) -> Rollout:
    rep = lambda t: t.repeat(times, *([1] * (t.dim() - 1)))
    return Rollout(rep(run.robot_pose), rep(run.robot_v), rep(run.robot_w), rep(run.command), rep(run.emergency),
                   rep(run.ped_pos), rep(run.ped_active), rep(run.success), rep(run.ped_collision),
                   rep(run.obstacle_collision), rep(run.end_step), run.robot_present, run.dt)


class Evaluator:
    def __init__(self, env, policy, objective: Objective | None = None, *, device="cpu", max_envs: int = 4096):
        self.env, self.policy, self.objective = env, policy, objective or Objective()
        self.device, self.max_envs = torch.device(device), int(max_envs)
        self._counterfactual: dict[tuple, Rollout] = {}

    def counterfactual(self, scenarios: list[Scenario], batch) -> Rollout:
        key = tuple(s.name for s in scenarios)
        if key not in self._counterfactual:
            if len(self._counterfactual) > 64:
                self._counterfactual.clear()
            self._counterfactual[key] = rollout(self.env, batch, robot_present=False)
        return self._counterfactual[key]

    def evaluate(self, parameter_sets: list[dict[str, float]], scenarios: list[Scenario]) -> list[CandidateResult]:
        base = collate(scenarios, self.device)
        cf = self.counterfactual(scenarios, base)
        S = len(scenarios)
        per_chunk = max(1, self.max_envs // S)
        results = []
        for start in range(0, len(parameter_sets), per_chunk):
            chunk = parameter_sets[start:start + per_chunk]
            n = len(chunk)
            batch = base.repeat(n)
            rows = [values for values in chunk for _ in range(S)]
            params = self.policy.parameter_tensor(rows, batch.size, self.device)
            run = rollout(self.env, batch, self.policy, params)
            metrics = all_metrics(run, _repeat_rollout(cf, n), batch, self.env.robot.radius)
            score = self.objective(metrics)
            for c, values in enumerate(chunk):
                rows_c = slice(c * S, (c + 1) * S)
                metrics_c = {key: value[rows_c] for key, value in metrics.items()}
                results.append(CandidateResult(dict(values), float(score[rows_c].mean()), summarize(metrics_c),
                                               score[rows_c].tolist()))
        return results

    def evaluate_detailed(self, values: dict[str, float], scenarios: list[Scenario]):
        """One parameter set: per-scenario metrics tensors, the rollout and its counterfactual."""
        batch = collate(scenarios, self.device)
        cf = self.counterfactual(scenarios, batch)
        params = self.policy.parameter_tensor(values, batch.size, self.device)
        run = rollout(self.env, batch, self.policy, params)
        metrics = all_metrics(run, cf, batch, self.env.robot.radius)
        metrics["objective"] = self.objective(metrics)
        return metrics, run, cf, batch
