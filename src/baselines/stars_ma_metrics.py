"""Forecasting, social-interaction and simulation metrics for multi-agent predictions.

``batch_metrics`` takes scene-frame predictions ``[K, N, P, 2]`` for a ``stars_ma_batch.GraphBatch``
and adds sums and counts to a ``MetricAccumulator``, which averages over a split. Distances are in metres (the scene
frame is a rigid transform of the world frame, so they equal world-frame distances).

* ``target_*``: the benchmark target node of each graph, min over K -- the same population and
  definition as the single-target baselines' best-of-K ADE/FDE.
* ``agent_*``: every node with a fully observed future, min over K per agent (marginal).
* ``joint_*``: per graph, the single sample k minimising the mean ADE over its complete agents.
* ``interacting_*`` / ``isolated_*``: complete agents with / without another complete agent
  within ``interaction_radius`` at some future ground-truth step.
* ``density_<bin>_*``: complete agents grouped by the graph's node count.
* ``moving_*`` / ``stationary_*``: complete agents whose ground-truth future mean speed is above /
  below ``moving_speed`` (most STEPS pedestrians stand, so pooled ADE mostly measures them).
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
    """Running sums and counts; values may be Python numbers or (device) tensors, read with ``result``."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, float] = defaultdict(float)

    def add(self, name: str, total, count) -> None:
        self.sums[name] = self.sums[name] + (total.detach().double() if torch.is_tensor(total) else float(total))
        self.counts[name] = self.counts[name] + (count.detach().double() if torch.is_tensor(count) else float(count))

    def totals(self) -> dict[str, tuple[float, float]]:
        return {name: (float(self.sums[name]), float(self.counts[name])) for name in self.sums}

    def all_reduce(self) -> None:
        """Sum every rank's totals (torch.distributed); afterwards every rank holds the global result."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, self.totals())
        self.sums, self.counts = defaultdict(float), defaultdict(float)
        for totals in gathered:
            for name, (total, count) in totals.items():
                self.sums[name] += total
                self.counts[name] += count

    def result(self) -> dict[str, float]:
        totals = self.totals()
        out = {name: total / count for name, (total, count) in totals.items() if count > 0}
        out.update({f"n_{name}": totals[name][1] for name in ("target_ade", "agent_ade", "joint_ade") if name in totals})
        return out


def batch_metrics(prediction: torch.Tensor, batch, accumulator: MetricAccumulator, *,
                  collision_distance: float = 0.2, interaction_radius: float = 2.0,
                  dataset_names: list[str] | None = None, moving_speed: float = 0.2) -> None:
    """Add one batch's metrics (module docstring), vectorised over graphs without host syncs.

    With ``dataset_names`` the headline metrics are also reported per source dataset as
    ``<dataset>/<metric>``.
    """
    K = prediction.shape[0]
    mask = batch.node_mask                                            # [B, N]
    pred = batch.to_padded(prediction, node_dim=1)                    # [K, B, N, P, 2]
    future = batch.to_padded(torch.nan_to_num(batch.future))          # [B, N, P, 2]
    complete = batch.to_padded(batch.future_valid.all(dim=-1), False) & mask
    error = torch.linalg.norm(pred - future[None], dim=-1)            # [K, B, N, P]
    ade, fde = error.mean(dim=-1), error[..., -1]
    min_ade, min_fde = ade.min(dim=0).values, fde.min(dim=0).values
    cf = complete.to(ade.dtype)
    count = cf.sum(dim=-1)                                            # [B]
    per_graph: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    target = cf[:, 0]
    per_graph["target_ade"] = (min_ade[:, 0] * target, target)
    per_graph["target_fde"] = (min_fde[:, 0] * target, target)
    per_graph["agent_ade"] = ((min_ade * cf).sum(-1), count)
    per_graph["agent_fde"] = ((min_fde * cf).sum(-1), count)
    has = (count > 0).to(ade.dtype)
    mean_ade = (ade * cf).sum(-1) / count.clamp(min=1)               # [K, B]
    best = mean_ade.argmin(dim=0)
    joint_fde = (fde * cf).sum(-1) / count.clamp(min=1)
    per_graph["joint_ade"] = (mean_ade.gather(0, best[None])[0] * has, has)
    per_graph["joint_fde"] = (joint_fde.gather(0, best[None])[0] * has, has)

    N = mask.shape[1]
    pair = complete[:, :, None] & complete[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=mask.device)
    upper = pair & torch.ones(N, N, dtype=torch.bool, device=mask.device).triu(1)
    pairs = upper.to(ade.dtype).sum(dim=(1, 2))

    def closest(xy):                                                  # [B, N, P, 2] -> [B, N, N]
        return torch.linalg.norm(xy[:, :, None] - xy[:, None], dim=-1).min(dim=-1).values

    gap = closest(future)
    interacting = ((gap < interaction_radius) & pair).any(dim=-1) & complete
    for name, chosen in (("interacting", interacting), ("isolated", complete & ~interacting)):
        cm = chosen.to(ade.dtype)
        per_graph[f"{name}_ade"] = ((min_ade * cm).sum(-1), cm.sum(-1))
        per_graph[f"{name}_fde"] = ((min_fde * cm).sum(-1), cm.sum(-1))
    start = batch.to_padded(batch.obs_xy[:, -1])                       # [B, N, 2]
    path = torch.cat([start[:, :, None], future], dim=2)
    mean_speed = torch.linalg.norm(path[:, :, 1:] - path[:, :, :-1], dim=-1).mean(-1) / batch.dt
    moving = (mean_speed > moving_speed) & complete
    for name, chosen in (("moving", moving), ("stationary", complete & ~moving)):
        cm = chosen.to(ade.dtype)
        per_graph[f"{name}_ade"] = ((min_ade * cm).sum(-1), cm.sum(-1))
        per_graph[f"{name}_fde"] = ((min_fde * cm).sum(-1), cm.sum(-1))
    sizes = (batch.ptr[1:] - batch.ptr[:-1]).to(ade.dtype)
    for low, high, label in DENSITY_BINS:
        in_bin = ((sizes >= low) & (sizes < high)).to(ade.dtype)
        per_graph[f"density_{label}_ade"] = ((min_ade * cf).sum(-1) * in_bin, count * in_bin)
    uf = upper.to(ade.dtype)
    per_graph["collision_rate_gt"] = (((gap < collision_distance) * uf).sum(dim=(1, 2)), pairs)
    per_graph["min_distance_gt"] = ((gap * uf).sum(dim=(1, 2)), pairs)
    collided, distance = torch.zeros_like(pairs), torch.zeros_like(pairs)
    for sample in pred:
        predicted = closest(sample)
        collided = collided + ((predicted < collision_distance) * uf).sum(dim=(1, 2))
        distance = distance + (predicted * uf).sum(dim=(1, 2))
    per_graph["collision_rate_pred"] = (collided, pairs * K)
    per_graph["min_distance_pred"] = (distance, pairs * K)
    if K > 1:
        final = pred[..., -1, :]                                      # [K, B, N, 2]
        spread = torch.zeros_like(min_ade)
        for k in range(K):
            spread = spread + torch.linalg.norm(final - final[k:k + 1], dim=-1).sum(dim=0)
        spread = spread / (K * (K - 1))
        per_graph["diversity_final"] = ((spread * cf).sum(-1), count)

    for name, (total, number) in per_graph.items():
        accumulator.add(name, total.sum(), number.sum())
    if dataset_names:
        headline = ("target_ade", "target_fde", "agent_ade", "agent_fde", "joint_ade", "joint_fde",
                    "moving_ade", "moving_fde")
        for index in torch.unique(batch.dataset).tolist():
            chosen = batch.dataset == index
            for name in headline:
                total, number = per_graph[name]
                accumulator.add(f"{dataset_names[index]}/{name}", total[chosen].sum(), number[chosen].sum())


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
