"""Forecasting, social-interaction and simulation metrics for multi-agent predictions.

All functions take scene-frame predictions ``[K, N, P, 2]`` for a PyG batch and return sums and
counts, so a ``MetricAccumulator`` can average over a split. Distances are in metres (the scene
frame is a rigid transform of the world frame, so they equal world-frame distances).

* ``target_*``: the benchmark target node of each graph, min over K -- the same population and
  definition as the single-target baselines' best-of-K ADE/FDE.
* ``agent_*``: every node with a fully observed future, min over K per agent (marginal).
* ``joint_*``: per graph, the single sample k minimising the mean ADE over its complete agents.
* ``interacting_*`` / ``isolated_*``: complete agents with / without another complete agent
  within ``interaction_radius`` at some future ground-truth step.
* ``density_<bin>_*``: complete agents grouped by the graph's node count.
* ``collision_rate_pred`` / ``collision_rate_gt``: fraction of complete-agent pairs closer than
  ``collision_distance`` at some future step (prediction averaged over K, and ground truth).
* ``min_distance_*``: mean closest approach between complete-agent pairs.
* ``diversity_final``: mean pairwise distance between the K final positions (K > 1).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

DENSITY_BINS = ((0, 10, "sparse"), (10, 30, "medium"), (30, 10**9, "dense"))


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, float] = defaultdict(float)

    def add(self, name: str, total, count) -> None:
        self.sums[name] += float(total)
        self.counts[name] += float(count)

    def result(self) -> dict[str, float]:
        out = {name: self.sums[name] / self.counts[name] for name in self.sums if self.counts[name] > 0}
        out.update({f"n_{name}": self.counts[name] for name in ("target_ade", "agent_ade", "joint_ade") if name in self.counts})
        return out


def _graph_slices(batch):
    pointer = batch.ptr.tolist()
    return list(zip(pointer[:-1], pointer[1:]))


def pairwise_stats(xy: torch.Tensor, collision_distance: float) -> tuple[float, float, int]:
    """Collided pairs, summed closest approach and number of pairs for ``[N, T, 2]`` agents."""
    n = xy.shape[0]
    if n < 2:
        return 0.0, 0.0, 0
    distance = torch.linalg.norm(xy[:, None] - xy[None], dim=-1)  # [N, N, T]
    pairs = torch.triu_indices(n, n, offset=1, device=xy.device)
    closest = distance[pairs[0], pairs[1]].min(dim=-1).values
    return float((closest < collision_distance).sum()), float(closest.sum()), int(closest.numel())


def forecasting_metrics(prediction: torch.Tensor, batch, accumulator: MetricAccumulator, *,
                        collision_distance: float = 0.2, interaction_radius: float = 2.0) -> None:
    """Add one batch's metrics; ``prediction`` is scene-frame ``[K, N, P, 2]``."""
    future, valid = batch.future, batch.future_valid
    error = torch.linalg.norm(prediction - torch.nan_to_num(future)[None], dim=-1)  # [K, N, P]
    complete = valid.all(dim=-1)
    ade, fde = error.mean(dim=-1), error[..., -1]  # [K, N]; only read for complete agents
    min_ade, min_fde = ade.min(dim=0).values, fde.min(dim=0).values

    target = batch.is_target & complete
    accumulator.add("target_ade", min_ade[target].sum(), target.sum())
    accumulator.add("target_fde", min_fde[target].sum(), target.sum())
    accumulator.add("agent_ade", min_ade[complete].sum(), complete.sum())
    accumulator.add("agent_fde", min_fde[complete].sum(), complete.sum())

    for start, stop in _graph_slices(batch):
        nodes = torch.arange(start, stop, device=prediction.device)[complete[start:stop]]
        if not len(nodes):
            continue
        best = ade[:, nodes].mean(dim=-1).argmin()
        accumulator.add("joint_ade", ade[best, nodes].mean(), 1)
        accumulator.add("joint_fde", fde[best, nodes].mean(), 1)

        truth = future[nodes]
        if len(nodes) > 1:
            gap = torch.linalg.norm(truth[:, None] - truth[None], dim=-1).min(dim=-1).values
            gap.fill_diagonal_(float("inf"))
            interacting = (gap < interaction_radius).any(dim=-1)
        else:
            interacting = torch.zeros(1, dtype=torch.bool, device=prediction.device)
        for name, mask in (("interacting", interacting), ("isolated", ~interacting)):
            accumulator.add(f"{name}_ade", min_ade[nodes][mask].sum(), mask.sum())
            accumulator.add(f"{name}_fde", min_fde[nodes][mask].sum(), mask.sum())
        size = stop - start
        label = next(name for low, high, name in DENSITY_BINS if low <= size < high)
        accumulator.add(f"density_{label}_ade", min_ade[nodes].sum(), len(nodes))

        collided, closest, pairs = pairwise_stats(truth, collision_distance)
        accumulator.add("collision_rate_gt", collided, pairs)
        accumulator.add("min_distance_gt", closest, pairs)
        for sample in prediction[:, nodes]:
            collided, closest, pairs = pairwise_stats(sample, collision_distance)
            accumulator.add("collision_rate_pred", collided, pairs)
            accumulator.add("min_distance_pred", closest, pairs)

    if prediction.shape[0] > 1:
        final = prediction[:, complete, -1]  # [K, M, 2]
        k = final.shape[0]
        spread = torch.linalg.norm(final[:, None] - final[None], dim=-1).sum(dim=(0, 1)) / (k * (k - 1))
        accumulator.add("diversity_final", spread.sum(), spread.numel())


def motion_statistics(xy: torch.Tensor, dt: float) -> dict[str, np.ndarray]:
    """Per-step speed and acceleration magnitudes of ``[N, T, 2]`` trajectories (for distributions)."""
    velocity = (xy[:, 1:] - xy[:, :-1]) / dt
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
    return {"speed": torch.linalg.norm(velocity, dim=-1).flatten().cpu().numpy(),
            "acceleration": torch.linalg.norm(acceleration, dim=-1).flatten().cpu().numpy()}


def wasserstein_1d(a: np.ndarray, b: np.ndarray, points: int = 512) -> float:
    """1-D earth mover's distance between two empirical samples (quantile approximation)."""
    if not len(a) or not len(b):
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, points)
    return float(np.mean(np.abs(np.quantile(a, quantiles) - np.quantile(b, quantiles))))
