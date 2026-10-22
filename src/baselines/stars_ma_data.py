"""Shared forecasting scenes as multi-agent STARS graphs, one graph per scene window.

The benchmark population is one sample per (window, target pedestrian), so a window with 26
forecastable pedestrians appears 26 times, each in its own target's frame. Multi-agent training
predicts every pedestrian at once, so ``windows="unique"`` keeps the first sample of each
``(dataset, recording, start_step)`` window; ``windows="all"`` keeps every benchmark sample, so that
the ``is_target`` node of each graph is evaluated on exactly the population the single-target
baselines report.

Every node is forecast. Its future is supervised wherever it is observed (``future_valid``); the
standard ADE/FDE use nodes whose whole future is observed. Positions stay in the sample's scene
frame (``steps_default``: target-centric SE(2)); the model works in each node's local frame and
``origin``/``heading`` map between the two.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from baselines.stars_ma_graph import build_graph, fill_observed, select_agents


def _leaf_rows(dataset, offset: int = 0):
    """Yield ``(global_index, window_key)`` for every sample of a (possibly wrapped) SceneDataset."""
    if isinstance(dataset, ConcatDataset):
        start = offset
        for child in dataset.datasets:
            yield from _leaf_rows(child, start)
            start += len(child)
    elif isinstance(dataset, Subset):
        parent = {index: key for index, key in _leaf_rows(dataset.dataset)}
        for position, index in enumerate(dataset.indices):
            yield offset + position, parent[int(index)]
    elif hasattr(dataset, "rows"):
        name = getattr(getattr(dataset, "spec", None), "name", "")
        for index, row in enumerate(dataset.rows):
            yield offset + index, (name, row["recording"], int(row["start_step"]))
    else:
        raise TypeError(f"Cannot read window keys from {type(dataset).__name__}")


def unique_window_indices(dataset) -> list[int]:
    """Index of the first sample of every distinct scene window, in dataset order."""
    seen, keep = set(), []
    for index, key in _leaf_rows(dataset):
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return keep


class StarsMultiAgentDataset(Dataset):
    """Adapt shared ``Scene`` samples to multi-agent STARS graphs with every node's future."""

    def __init__(self, source, observation_steps: int, prediction_steps: int, *, dt: float,
                 knn: int | None, radius: float | None, windows: str = "unique", limit: int | None = None) -> None:
        if windows not in {"unique", "all"}:
            raise ValueError("windows must be 'unique' or 'all'")
        self.source = source
        self.observation_steps, self.prediction_steps = int(observation_steps), int(prediction_steps)
        self.dt, self.knn, self.radius = float(dt), knn, radius
        self.indices = unique_window_indices(source) if windows == "unique" else list(range(len(source)))
        if limit is not None:
            self.indices = self.indices[:limit]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        from torch_geometric.data import Data

        scene = self.source[self.indices[index]]
        obs, pred = self.observation_steps, self.prediction_steps
        xy = torch.from_numpy(np.asarray(scene.xy, dtype=np.float32))
        valid = torch.from_numpy(np.asarray(scene.valid, dtype=bool))
        if xy.shape[0] != obs + pred:
            raise ValueError(f"Scene has {xy.shape[0]} steps, expected {obs + pred}")
        agents = select_agents(valid, obs)
        if not len(agents) or int(agents[0]) != 0:
            raise ValueError("The benchmark target must be a node (visible at the last observed step)")
        observed_xy, observed_valid = xy[:obs, agents], valid[:obs, agents]
        graph = build_graph(observed_xy, observed_valid, dt=self.dt, knn=self.knn, radius=self.radius)
        future_valid = valid[obs:, agents].T
        is_target = torch.zeros(len(agents), dtype=torch.bool)
        is_target[0] = True
        return Data(
            num_nodes=len(agents),
            **graph,
            obs_xy=fill_observed(torch.nan_to_num(observed_xy), observed_valid).transpose(0, 1).contiguous(),
            future=xy[obs:, agents].transpose(0, 1).contiguous(),
            future_valid=future_valid.contiguous(),
            is_target=is_target,
            agent_column=agents,
            rotation=torch.tensor([float(scene.rotation)], dtype=torch.float64),
            center=torch.from_numpy(np.asarray(scene.center, dtype=np.float64))[None],
            sample_index=torch.tensor([int(self.indices[index])]),
        )
