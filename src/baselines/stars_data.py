"""STARS graph adapter over the shared forecasting scene.

One graph per benchmark sample: the target pedestrian takes STARS's robot slot (node 0,
``node_type`` 0) and every neighbour observed at least ``MIN_HISTORY_POINTS`` times in the
observation window is a human node. Edges connect every ordered pair of nodes that share at
least one valid observed step. Only the observed steps build the graph, so no future
information reaches the encoder.

Edge features follow STARS's SWM forecasting dataset
(``references/stars/src/dataloaders/swm_segment_forecasting_dataset.py``): the SE(2) pose of
the destination node in the source node's frame, ``[rel_x, rel_y, cos_rel, sin_rel]``, plus
its per-step change, with headings taken from direction of travel, then linearly resampled
(``align_corners=True``) onto STARS's 40-step grid. The upstream helpers work one edge and one
step at a time; these are vectorised equivalents, checked against them in
``tests/test_stars_integration.py``.
"""
from __future__ import annotations

import numpy as np


STARS_TIMESTEPS = 40
EDGE_FEATURE_DIM = 8
MIN_HISTORY_POINTS = 2
NODE_TYPE_TARGET = 0  # STARS's robot slot
NODE_TYPE_HUMAN = 1


def fill_observed(xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill missing observed steps per agent by linear interpolation, clamped at the ends.

    Matches STARS's ``np.interp(..., left=first, right=last)`` treatment of gaps. ``xy`` is
    ``[T, A, 2]`` and ``valid`` is ``[T, A]``; every agent needs at least one valid step.
    """
    filled = np.array(xy, dtype=np.float32, copy=True)
    steps = np.arange(xy.shape[0], dtype=np.float32)
    for agent in range(xy.shape[1]):
        known = np.flatnonzero(valid[:, agent])
        if len(known) == len(steps):
            continue
        for axis in range(2):
            filled[:, agent, axis] = np.interp(steps, steps[known], xy[known, agent, axis])
    return filled


def headings_from_motion(xy: np.ndarray) -> np.ndarray:
    """Direction of travel per step, ``[T, A]``: STARS's ``_xy_to_theta`` for every agent.

    Step 0 is 0; later steps take the angle of the last non-zero displacement (0 until the
    agent first moves).
    """
    displacement = np.diff(xy, axis=0)  # [T-1, A, 2]
    moving = (np.abs(displacement[..., 0]) + np.abs(displacement[..., 1])) > 1e-8
    angle = np.arctan2(displacement[..., 1], displacement[..., 0]).astype(np.float32)
    theta = np.zeros(xy.shape[:2], dtype=np.float32)
    for step in range(1, xy.shape[0]):
        theta[step] = np.where(moving[step - 1], angle[step - 1], theta[step - 1])
    return theta


def fully_connected_edges(n_nodes: int) -> np.ndarray:
    """Directed edges without self-loops, in STARS's ``(i, j) for i for j`` order: ``[2, E]``."""
    source, destination = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes), indexing="ij")
    keep = source != destination
    return np.stack([source[keep], destination[keep]]).astype(np.int64)


def edge_features(xy: np.ndarray, theta: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """SE(2) relative pose plus per-step delta for every edge and step: ``[E, T, 8]``."""
    source, destination = edge_index
    cos_i, sin_i = np.cos(theta[:, source]), np.sin(theta[:, source])  # [T, E]
    cos_j, sin_j = np.cos(theta[:, destination]), np.sin(theta[:, destination])
    dx = xy[:, destination, 0] - xy[:, source, 0]
    dy = xy[:, destination, 1] - xy[:, source, 1]
    relative = np.stack(
        [
            cos_i * dx + sin_i * dy,
            -sin_i * dx + cos_i * dy,
            cos_j * cos_i + sin_j * sin_i,
            sin_j * cos_i - cos_j * sin_i,
        ],
        axis=-1,
    ).astype(np.float32)  # [T, E, 4]
    delta = np.zeros_like(relative)
    delta[1:] = relative[1:] - relative[:-1]
    return np.concatenate([relative, delta], axis=-1).transpose(1, 0, 2)


def resample_linear(values: np.ndarray, size: int = STARS_TIMESTEPS) -> np.ndarray:
    """Resample axis 1 of ``[E, T, F]`` to ``size`` steps, as ``F.interpolate(linear, align_corners=True)``."""
    steps = values.shape[1]
    if steps == size:
        return values.astype(np.float32)
    position = np.linspace(0.0, steps - 1, size)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, steps - 1)
    weight = (position - low).astype(np.float32)[None, :, None]
    return ((1.0 - weight) * values[:, low] + weight * values[:, high]).astype(np.float32)


def resample_valid(valid: np.ndarray, size: int = STARS_TIMESTEPS) -> np.ndarray:
    """Per-node validity on the resampled grid, ``[N, size]``: both bracketing steps must be valid."""
    steps = valid.shape[1]
    position = np.linspace(0.0, steps - 1, size)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, steps - 1)
    exact = np.isclose(position, low)
    return (valid[:, low] & (exact | valid[:, high])).astype(np.float32)


def stars_graph(xy: np.ndarray, valid: np.ndarray, observation_steps: int) -> dict[str, np.ndarray]:
    """Build one STARS graph from a scene's ``[T, A, 2]`` positions (target first) and ``[T, A]`` validity.

    Positions should already be in the model frame (``steps_default``: target-centric SE(2)).
    Returns ``node_type [N]``, ``edge_index [2, E]``, ``edge_attr [E, 40, 8]``,
    ``valid_mask [N, 40]``, ``history [obs, 2]`` (the target's observed positions) and
    ``node_agent_index [N]`` (each node's column in the scene).
    """
    observed_xy = np.asarray(xy[:observation_steps], dtype=np.float32)
    observed_valid = np.asarray(valid[:observation_steps], dtype=bool)
    if not observed_valid[:, 0].all():
        raise ValueError("STARS target pedestrian must be observed at every history step")
    counts = observed_valid.sum(axis=0)
    agents = np.concatenate([[0], 1 + np.flatnonzero(counts[1:] >= MIN_HISTORY_POINTS)])
    node_xy = fill_observed(observed_xy[:, agents], observed_valid[:, agents])
    node_valid = observed_valid[:, agents]
    theta = headings_from_motion(node_xy)
    valid_mask = resample_valid(node_valid.T)
    edge_index = fully_connected_edges(len(agents))
    # An edge whose endpoints never share a valid step is fully masked in STARS's temporal edge
    # transformer, which then returns NaN in eval mode. It carries no information, so drop it.
    overlapping = (valid_mask[edge_index[0]] * valid_mask[edge_index[1]]).any(axis=1)
    edge_index = edge_index[:, overlapping]
    attributes = resample_linear(edge_features(node_xy, theta, edge_index))
    node_type = np.full(len(agents), NODE_TYPE_HUMAN, dtype=np.int64)
    node_type[0] = NODE_TYPE_TARGET
    return {
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_attr": attributes,
        "valid_mask": valid_mask,
        "history": np.array(observed_xy[:, 0], copy=True),
        "node_agent_index": agents.astype(np.int64),
    }
