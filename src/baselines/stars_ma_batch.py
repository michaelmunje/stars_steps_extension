"""Scaled multi-agent STARS data path: cached windows, batched GPU graph construction, node budgets.

``stars_ma_data.StarsMultiAgentDataset`` builds one PyG graph per window in Python, which caps
training at a few thousand graphs per second. This module is the training path instead:

* ``WindowStore`` holds every window of a split as flat node arrays (``ptr`` marks windows). It is
  extracted from the shared ``SceneDataset`` once (``build_store``, parallel) and cached on disk;
  a whole split is small enough (a few GB at most) to live on the GPU.
* ``build_batch`` gathers a set of windows into padded ``[B, N, T]`` tensors and builds all their
  graphs at once on the device -- the same quantities as ``stars_ma_graph.build_graph`` (checked in
  the tests), vectorised over graphs. ``GraphBatch.rebuild`` does the same for closed-loop
  rollouts, so multi-step training no longer rebuilds graph by graph.
* ``node_budget_batches`` groups windows of similar size into batches of at most ``max_nodes``
  nodes, so one setting works for STEPS (~20 people per window) and MADRAS (up to ~550).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from baselines.stars_data import MIN_HISTORY_POINTS, NODE_TYPE_HUMAN
from baselines.stars_ma_graph import MOTION_EPSILON, NODE_FEATURE_DIM, rotate

CACHE_VERSION = 1


@dataclass
class WindowStore:
    """Every forecast window of one or more splits as flat node arrays.

    Node ``ptr[w]`` is window ``w``'s benchmark target. Invalid positions are stored as 0.
    """

    obs_xy: torch.Tensor        # [M, T, 2] float32
    obs_valid: torch.Tensor     # [M, T] bool
    future: torch.Tensor        # [M, P, 2] float32
    future_valid: torch.Tensor  # [M, P] bool
    ptr: torch.Tensor           # [W + 1] long
    dataset: torch.Tensor       # [W] long, index into ``names``
    sample_index: torch.Tensor  # [W] long, index of the window's sample in its source split
    names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.ptr.numel() - 1)

    @property
    def sizes(self) -> torch.Tensor:
        return self.ptr[1:] - self.ptr[:-1]

    @property
    def device(self) -> torch.device:
        return self.obs_xy.device

    def to(self, device) -> "WindowStore":
        return WindowStore(*(getattr(self, name).to(device) for name in _TENSORS), names=list(self.names))

    def select(self, windows: torch.Tensor) -> "WindowStore":
        """A store of only ``windows`` (in that order)."""
        windows = torch.as_tensor(windows, dtype=torch.long, device=self.ptr.device)
        sizes = self.sizes[windows]
        nodes = _ranges(self.ptr[windows], sizes)
        ptr = torch.zeros(len(windows) + 1, dtype=torch.long, device=self.ptr.device)
        ptr[1:] = sizes.cumsum(0)
        return WindowStore(self.obs_xy[nodes], self.obs_valid[nodes], self.future[nodes], self.future_valid[nodes],
                           ptr, self.dataset[windows], self.sample_index[windows], list(self.names))

    def state(self) -> dict:
        return {**{name: getattr(self, name).cpu() for name in _TENSORS}, "names": list(self.names)}

    @classmethod
    def from_state(cls, state: dict) -> "WindowStore":
        return cls(*(state[name] for name in _TENSORS), names=list(state["names"]))


_TENSORS = ("obs_xy", "obs_valid", "future", "future_valid", "ptr", "dataset", "sample_index")


def concat_stores(stores: list[WindowStore]) -> WindowStore:
    """One store holding every window of ``stores``; dataset ids are remapped onto the union of names."""
    names: list[str] = []
    for store in stores:
        names += [name for name in store.names if name not in names]
    ptr, offset, datasets = [torch.zeros(1, dtype=torch.long)], 0, []
    for store in stores:
        ptr.append(store.ptr[1:].cpu() + offset)
        offset += int(store.ptr[-1])
        remap = torch.tensor([names.index(name) for name in store.names], dtype=torch.long)
        datasets.append(remap[store.dataset.cpu()])
    cat = lambda name: torch.cat([getattr(store, name).cpu() for store in stores])
    return WindowStore(cat("obs_xy"), cat("obs_valid"), cat("future"), cat("future_valid"), torch.cat(ptr),
                       torch.cat(datasets), cat("sample_index"), names)


def _ranges(starts: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """Concatenated ``arange(start, start + size)`` for every pair."""
    total = int(sizes.sum())
    if not total:
        return torch.zeros(0, dtype=torch.long, device=starts.device)
    offsets = torch.repeat_interleave(starts - torch.cumsum(sizes, 0) + sizes, sizes)
    return offsets + torch.arange(total, device=starts.device)


# ----------------------------------------------------------------------------- extraction and cache


def extract_window(scene, observation_steps: int, prediction_steps: int):
    """One sample's forecast nodes (target first) as numpy arrays, or None if the target is not a node."""
    xy = np.asarray(scene.xy, dtype=np.float32)
    valid = np.asarray(scene.valid, dtype=bool)
    obs = observation_steps
    if xy.shape[0] != obs + prediction_steps:
        raise ValueError(f"Scene has {xy.shape[0]} steps, expected {obs + prediction_steps}")
    observed = valid[:obs]
    agents = np.flatnonzero(observed[-1] & (observed.sum(axis=0) >= MIN_HISTORY_POINTS))
    if not len(agents) or agents[0] != 0:
        return None
    xy = np.where(valid[..., None], np.nan_to_num(xy), 0.0).astype(np.float32)
    return (xy[:obs, agents].transpose(1, 0, 2), observed[:, agents].T,
            xy[obs:, agents].transpose(1, 0, 2), valid[obs:, agents].T)


class _Extract(torch.utils.data.Dataset):
    def __init__(self, source, indices, obs, pred):
        self.source, self.indices, self.obs, self.pred = source, indices, obs, pred

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return index, extract_window(self.source[index], self.obs, self.pred)


def build_store(source, name: str, observation_steps: int, prediction_steps: int, *, windows: str = "unique",
                workers: int = 0, limit: int | None = None, progress=None) -> WindowStore:
    """Extract a split's windows from a ``SceneDataset`` (``windows`` as in ``StarsMultiAgentDataset``)."""
    from baselines.stars_ma_data import unique_window_indices

    if windows not in {"unique", "all"}:
        raise ValueError("windows must be 'unique' or 'all'")
    indices = unique_window_indices(source) if windows == "unique" else list(range(len(source)))
    if limit is not None:
        indices = indices[:limit]
    loader = torch.utils.data.DataLoader(_Extract(source, indices, observation_steps, prediction_steps),
                                         batch_size=64, num_workers=workers, collate_fn=list,
                                         prefetch_factor=4 if workers else None)
    parts = {key: [] for key in ("obs_xy", "obs_valid", "future", "future_valid")}
    sizes, sample_index = [], []
    for chunk in (progress(loader) if progress else loader):
        for index, window in chunk:
            if window is None:
                continue
            for key, value in zip(parts, window):
                parts[key].append(value)
            sizes.append(len(window[0]))
            sample_index.append(index)
    if not sizes:
        raise RuntimeError(f"{name}: no forecastable windows")
    ptr = torch.zeros(len(sizes) + 1, dtype=torch.long)
    ptr[1:] = torch.tensor(sizes).cumsum(0)
    cat = lambda key, dtype: torch.from_numpy(np.concatenate(parts[key]).astype(dtype))
    return WindowStore(cat("obs_xy", np.float32), cat("obs_valid", bool), cat("future", np.float32),
                       cat("future_valid", bool), ptr, torch.zeros(len(sizes), dtype=torch.long),
                       torch.tensor(sample_index, dtype=torch.long), [name])


def cache_key(source, name: str, split: str, observation_steps: int, prediction_steps: int, windows: str,
              forecasting: dict) -> str:
    """Hash of everything a cached store depends on: source index, split, window and preset config."""
    from data import _index_path

    spec, cfg = source.spec, source.cfg
    index = _index_path(spec, cfg.seeds.data_seed, cfg.split_strategy, cfg.held_out_scene)
    stat = index.stat()
    # The index file's identity (name, size, mtime; kept by ``rsync -a``) stands for the data, so a
    # copied dataset reuses copied caches; the absolute path is deliberately not part of the key.
    payload = {"version": CACHE_VERSION, "name": name, "split": split,
               "fraction": spec.fraction, "split_strategy": cfg.split_strategy, "data_seed": cfg.seeds.data_seed,
               "index": [index.name, stat.st_size, stat.st_mtime_ns], "windows": windows,
               "obs": observation_steps, "pred": prediction_steps, "forecasting": forecasting, "rows": len(source)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_store(source, name: str, split: str, observation_steps: int, prediction_steps: int, *, windows: str,
                 forecasting: dict, cache_dir: Path, workers: int = 0, log=print) -> WindowStore:
    """Load a split's store from ``cache_dir`` or build and save it (atomically)."""
    key = cache_key(source, name, split, observation_steps, prediction_steps, windows, forecasting)
    path = Path(cache_dir) / f"{name}-{split}-{windows}-{key}.pt"
    if path.exists():
        return WindowStore.from_state(torch.load(path, map_location="cpu", weights_only=False))
    log(f"building window cache {path.name} ({len(source)} samples, {workers} workers)")
    store = build_store(source, name, observation_steps, prediction_steps, windows=windows, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(store.state(), temporary)
    os.replace(temporary, path)
    return store


# ----------------------------------------------------------------------------- batched graphs


def fill_observed_padded(xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``stars_ma_graph.fill_observed`` for ``[..., T, 2]`` / ``[..., T]``: interpolate gaps, hold the ends."""
    steps = valid.shape[-1]
    step = torch.arange(steps, device=xy.device).expand_as(valid)
    before = torch.cummax(torch.where(valid, step, torch.full_like(step, -1)), dim=-1).values
    after = torch.flip(torch.cummin(torch.flip(torch.where(valid, step, torch.full_like(step, steps)), [-1]),
                                    dim=-1).values, [-1])
    before, after = torch.where(before >= 0, before, after), torch.where(after < steps, after, before)
    before, after = before.clamp(0, steps - 1), after.clamp(0, steps - 1)
    t0, t1 = before.to(xy.dtype), after.to(xy.dtype)
    weight = torch.where(t1 > t0, (step.to(xy.dtype) - t0) / (t1 - t0).clamp(min=1.0), torch.zeros_like(t0))
    gather = lambda index: torch.gather(xy, -2, index[..., None].expand(*index.shape, 2))
    value = (1.0 - weight[..., None]) * gather(before) + weight[..., None] * gather(after)
    return torch.where(valid[..., None], xy, value)


def headings_padded(xy: torch.Tensor) -> torch.Tensor:
    """``headings_from_motion(xy, backfill=True)`` for ``[..., T, 2]`` positions: ``[..., T]``."""
    displacement = xy[..., 1:, :] - xy[..., :-1, :]
    moving = displacement.abs().sum(dim=-1) > MOTION_EPSILON
    angle = torch.atan2(displacement[..., 1], displacement[..., 0])
    if not displacement.shape[-2]:
        return torch.zeros_like(xy[..., 0])
    step = torch.arange(moving.shape[-1], device=xy.device).expand_as(moving)
    last = torch.cummax(torch.where(moving, step, torch.full_like(step, -1)), dim=-1).values
    held = torch.where(last >= 0, torch.gather(angle, -1, last.clamp(min=0)), torch.zeros_like(angle))
    theta = torch.cat([torch.zeros_like(held[..., :1]), held], dim=-1)
    has_moved = moving.any(dim=-1, keepdim=True)
    first = moving.float().argmax(dim=-1, keepdim=True)
    initial = torch.gather(angle, -1, first)
    position = torch.arange(theta.shape[-1], device=xy.device)
    return torch.where(has_moved & (position <= first), initial, theta)


@dataclass
class GraphBatch:
    """A batch of window graphs, flattened like a PyG ``Batch`` and remembering its padded layout."""

    node_type: torch.Tensor
    node_attr: torch.Tensor     # [Nt, T, 7]
    valid_mask: torch.Tensor    # [Nt, T] float
    edge_index: torch.Tensor    # [2, E]
    edge_attr: torch.Tensor     # [E, T, 8]
    origin: torch.Tensor        # [Nt, 2]
    heading: torch.Tensor       # [Nt]
    obs_xy: torch.Tensor        # [Nt, T, 2], gaps filled
    future: torch.Tensor        # [Nt, P, 2]
    future_valid: torch.Tensor  # [Nt, P] bool
    is_target: torch.Tensor     # [Nt] bool
    batch: torch.Tensor         # [Nt] graph of each node
    ptr: torch.Tensor           # [B + 1]
    node_mask: torch.Tensor     # [B, N] padded layout
    dataset: torch.Tensor       # [B]
    windows: torch.Tensor       # [B] store window ids
    knn: int | None = None
    radius: float | None = None
    dt: float = 0.4

    @property
    def num_graphs(self) -> int:
        return int(self.ptr.numel() - 1)

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.numel())

    def to_padded(self, values: torch.Tensor, fill: float = 0.0, *, node_dim: int = 0) -> torch.Tensor:
        """Flat per-node ``values`` (nodes along ``node_dim``) in the padded ``[B, N]`` layout."""
        B, N = self.node_mask.shape
        values = values.movedim(node_dim, 0)
        out = values.new_full((B * N, *values.shape[1:]), fill)
        out[self.node_mask.flatten()] = values
        return out.view(B, N, *values.shape[1:]).movedim((0, 1), (node_dim, node_dim + 1))

    def rebuild(self, window: torch.Tensor, valid: torch.Tensor) -> "GraphBatch":
        """The same windows' graphs from new per-node scene-frame states ``[Nt, T, 2]`` / ``[Nt, T]``."""
        graph = batched_graph(self.to_padded(window), self.to_padded(valid.to(torch.bool), False), self.node_mask,
                              dt=self.dt, knn=self.knn, radius=self.radius)
        return GraphBatch(**graph, future=self.future, future_valid=self.future_valid, is_target=self.is_target,
                          batch=self.batch, ptr=self.ptr, node_mask=self.node_mask, dataset=self.dataset,
                          windows=self.windows, knn=self.knn, radius=self.radius, dt=self.dt)


def batched_graph(xy: torch.Tensor, valid: torch.Tensor, node_mask: torch.Tensor, *, dt: float,
                  knn: int | None, radius: float | None) -> dict[str, torch.Tensor]:
    """``build_graph`` for padded ``xy [B, N, T, 2]``, ``valid [B, N, T]``, ``node_mask [B, N]``, flattened.

    Every real node must be valid at the last step. Edges follow ``neighbourhood_edges``: no
    self-loops, endpoints share a valid step, within ``radius`` of the destination at the last step,
    and each destination keeps its ``knn`` nearest sources (``None``/``None``: fully connected).
    """
    with torch.autocast(device_type=xy.device.type, enabled=False):
        return _batched_graph(xy.float(), valid, node_mask, dt=dt, knn=knn, radius=radius)


def _batched_graph(xy, valid, node_mask, *, dt, knn, radius):
    B, N, T, _ = xy.shape
    valid = valid & node_mask[..., None]
    filled = fill_observed_padded(xy, valid)
    theta = headings_padded(filled)
    origin, heading = filled[:, :, -1], theta[:, :, -1]
    local = rotate(filled - origin[:, :, None], -heading[:, :, None])
    velocity = torch.cat([torch.zeros_like(local[:, :, :1]), (local[:, :, 1:] - local[:, :, :-1]) / dt], dim=2)
    relative = theta - heading[..., None]
    node_attr = torch.cat([local, velocity, torch.linalg.norm(velocity, dim=-1, keepdim=True),
                           torch.cos(relative)[..., None], torch.sin(relative)[..., None]], dim=-1)

    with torch.no_grad():
        vf = valid.to(torch.float32)
        allowed = (vf @ vf.transpose(1, 2)) > 0
        allowed &= node_mask[:, :, None] & node_mask[:, None, :]
        allowed &= ~torch.eye(N, dtype=torch.bool, device=xy.device)
        if radius is not None or knn is not None:
            last = filled[:, :, -1].detach()
            distance = torch.cdist(last, last)  # [B, source, destination]
            if radius is not None:
                allowed &= distance <= radius
            if knn is not None and knn < N - 1:
                ranked = distance.masked_fill(~allowed, math.inf)
                nearest = ranked.topk(knn, dim=1, largest=False).indices
                chosen = torch.zeros_like(allowed)
                chosen.scatter_(1, nearest, True)
                allowed &= chosen
        graph_id, source, destination = torch.nonzero(allowed, as_tuple=True)
        flat = torch.cumsum(node_mask.flatten().to(torch.long), 0) - 1
        edge_index = torch.stack([flat[graph_id * N + source], flat[graph_id * N + destination]])

    mask = node_mask.flatten()
    flatten = lambda value: value.reshape(B * N, *value.shape[2:])[mask]
    theta_i, theta_j = theta[graph_id, source].transpose(0, 1), theta[graph_id, destination].transpose(0, 1)
    pos_i, pos_j = filled[graph_id, source].transpose(0, 1), filled[graph_id, destination].transpose(0, 1)
    cos_i, sin_i, cos_j, sin_j = torch.cos(theta_i), torch.sin(theta_i), torch.cos(theta_j), torch.sin(theta_j)
    dx, dy = pos_j[..., 0] - pos_i[..., 0], pos_j[..., 1] - pos_i[..., 1]
    pose = torch.stack([cos_i * dx + sin_i * dy, -sin_i * dx + cos_i * dy,
                        cos_j * cos_i + sin_j * sin_i, sin_j * cos_i - cos_j * sin_i], dim=-1)  # [T, E, 4]
    delta = torch.cat([torch.zeros_like(pose[:1]), pose[1:] - pose[:-1]], dim=0)
    nodes = int(mask.sum())
    return {
        "node_type": torch.full((nodes,), NODE_TYPE_HUMAN, dtype=torch.long, device=xy.device),
        "node_attr": flatten(node_attr),
        "valid_mask": flatten(valid).to(xy.dtype),
        "edge_index": edge_index,
        "edge_attr": torch.cat([pose, delta], dim=-1).transpose(0, 1),
        "origin": flatten(origin),
        "heading": flatten(heading),
        "obs_xy": flatten(filled),
    }


def build_batch(store: WindowStore, windows: torch.Tensor, *, dt: float, knn: int | None,
                radius: float | None) -> GraphBatch:
    """Gather ``windows`` of ``store`` (on the store's device) and build their graphs at once."""
    windows = torch.as_tensor(windows, dtype=torch.long, device=store.ptr.device)
    sizes = store.sizes[windows]
    B, N = len(windows), int(sizes.max())
    node_mask = torch.arange(N, device=store.ptr.device)[None] < sizes[:, None]
    nodes = _ranges(store.ptr[windows], sizes)
    pad = lambda value, fill: _scatter_padded(value[nodes], node_mask, fill)
    graph = batched_graph(pad(store.obs_xy, 0.0), pad(store.obs_valid, False), node_mask, dt=dt, knn=knn, radius=radius)
    ptr = torch.zeros(B + 1, dtype=torch.long, device=store.ptr.device)
    ptr[1:] = sizes.cumsum(0)
    is_target = torch.zeros(len(nodes), dtype=torch.bool, device=store.ptr.device)
    is_target[ptr[:-1]] = True
    return GraphBatch(**graph, future=store.future[nodes], future_valid=store.future_valid[nodes], is_target=is_target,
                      batch=torch.repeat_interleave(torch.arange(B, device=store.ptr.device), sizes), ptr=ptr,
                      node_mask=node_mask, dataset=store.dataset[windows], windows=windows, knn=knn, radius=radius,
                      dt=dt)


def graph_batch_from_padded(xy: torch.Tensor, valid: torch.Tensor, node_mask: torch.Tensor, *, dt: float,
                            knn: int | None, radius: float | None, prediction_steps: int = 12) -> GraphBatch:
    """A ``GraphBatch`` from padded observed windows ``xy [B, N, T, 2]`` / ``valid [B, N, T]``.

    Used for inference outside the datasets (e.g. pedestrians in a simulator); futures are empty.
    Every masked-in node must be valid at the last step.
    """
    B, N = node_mask.shape
    graph = batched_graph(xy, valid, node_mask, dt=dt, knn=knn, radius=radius)
    sizes = node_mask.sum(dim=1)
    ptr = torch.zeros(B + 1, dtype=torch.long, device=xy.device)
    ptr[1:] = sizes.cumsum(0)
    nodes = int(ptr[-1])
    is_target = torch.zeros(nodes, dtype=torch.bool, device=xy.device)
    return GraphBatch(**graph, future=xy.new_zeros(nodes, prediction_steps, 2),
                      future_valid=torch.zeros(nodes, prediction_steps, dtype=torch.bool, device=xy.device),
                      is_target=is_target, batch=torch.repeat_interleave(torch.arange(B, device=xy.device), sizes),
                      ptr=ptr, node_mask=node_mask, dataset=torch.zeros(B, dtype=torch.long, device=xy.device),
                      windows=torch.arange(B, device=xy.device), knn=knn, radius=radius, dt=dt)


def _scatter_padded(values: torch.Tensor, node_mask: torch.Tensor, fill) -> torch.Tensor:
    B, N = node_mask.shape
    out = values.new_full((B * N, *values.shape[1:]), fill)
    out[node_mask.flatten()] = values
    return out.view(B, N, *values.shape[1:])


# ----------------------------------------------------------------------------- batching


def window_weights(store: WindowStore, dataset_weights: dict[str, float] | None) -> torch.Tensor | None:
    """Per-window sampling weights: each window of a dataset counts ``weight`` times (None = all 1).

    A dataset's share of an epoch is therefore proportional to ``weight x its window count``; equal
    weights give the natural mixture.
    """
    if not dataset_weights or all(dataset_weights.get(name, 1.0) == 1.0 for name in store.names):
        return None
    per_dataset = torch.tensor([dataset_weights.get(name, 1.0) for name in store.names], dtype=torch.float64)
    return per_dataset[store.dataset.cpu()]


def node_budget_batches(sizes: torch.Tensor, max_nodes: int, *, generator: torch.Generator | None = None,
                        shuffle: bool = True, weights: torch.Tensor | None = None, max_graphs: int | None = None,
                        pool: int = 64) -> list[torch.Tensor]:
    """Window-id batches of at most ``max_nodes`` nodes (a larger window gets its own batch).

    Windows are drawn in random order (or with replacement, proportionally to ``weights``, for one
    epoch's worth), then sorted by size inside pools of ~``pool`` batches so a batch holds windows
    of similar size (little padding); the batches themselves are shuffled.
    """
    sizes = sizes.cpu()
    count = len(sizes)
    if weights is not None:
        order = torch.multinomial(weights, count, replacement=True, generator=generator)
    elif shuffle:
        order = torch.randperm(count, generator=generator)
    else:
        order = torch.arange(count)
    mean = float(sizes.float().mean()) if count else 1.0
    chunk = max(1, int(pool * max_nodes / max(mean, 1.0)))
    batches = []
    for start in range(0, count, chunk):
        part = order[start:start + chunk]
        part = part[torch.argsort(sizes[part], stable=True)] if shuffle or weights is not None else part
        current, total = [], 0
        for window, size in zip(part.tolist(), sizes[part].tolist()):
            if current and (total + size > max_nodes or (max_graphs and len(current) >= max_graphs)):
                batches.append(torch.tensor(current))
                current, total = [], 0
            current.append(window)
            total += size
        if current:
            batches.append(torch.tensor(current))
    if shuffle or weights is not None:
        batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
    return batches
