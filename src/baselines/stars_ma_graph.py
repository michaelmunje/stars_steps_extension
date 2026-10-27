"""Multi-agent STARS graphs at the native STEPS rate, built in torch.

One graph per scene window. Every pedestrian that is visible at the last observed step and was
seen at least ``MIN_HISTORY_POINTS`` times in the observation window becomes a node
(``stars_ma_batch.extract_window``); all nodes are forecast. Unlike ``stars_data.py`` (single target in STARS's robot slot, 8 steps resampled to
40), there is no temporal interpolation onto STARS's 40-step grid: node and edge sequences have
the observation length itself. Node type 0 stays reserved for a robot, so every pedestrian is
``NODE_TYPE_HUMAN``.

Node features describe each pedestrian's own motion in its *local frame* (origin at its last
observed position, x axis along its last heading), so they are invariant to the scene frame:
``[x, y, vx, vy, speed, cos(dtheta), sin(dtheta)]`` per observed step, velocities in m/s.

Edge features are STARS's SE(2) features (``stars_data.edge_features``): the destination's pose in
the source's frame plus its per-step change. Headings come from the direction of travel as in
STARS, except that steps before an agent's first movement take that movement's direction instead
of the scene x axis (``headings_from_motion``), so the features do not depend on the scene frame. Edges follow STARS's ``(source, destination)``
convention; the topology is fully connected (as STARS) or, for the large STEPS graphs, the ``knn``
nearest sources within ``radius`` metres of each destination at the last observed step.

This per-window builder is the reference for the batched training path (``stars_ma_batch``),
which computes the same quantities for many graphs at once (checked in the tests).
"""
from __future__ import annotations

import math

import torch

from baselines.stars_data import NODE_TYPE_HUMAN

NODE_FEATURE_DIM = 7  # x, y, vx, vy, speed, cos(dtheta), sin(dtheta)
MOTION_EPSILON = 1e-8  # STARS's _xy_to_theta "has moved" threshold on |dx| + |dy|


def fill_observed(xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate missing steps per agent, clamped at the ends: ``[T, N, 2]``.

    Torch counterpart of ``stars_data.fill_observed`` (STARS's ``np.interp`` treatment); every agent
    needs at least one valid step. Observed values pass through unchanged, so gradients reach them.
    """
    steps, agents = valid.shape
    if bool(valid.all()):
        return xy
    step = torch.arange(steps, device=xy.device)
    filled = []
    for agent in range(agents):
        known = torch.nonzero(valid[:, agent], as_tuple=False).flatten()
        if len(known) == steps:
            filled.append(xy[:, agent])
            continue
        # Last known step at or before t, first known step at or after t; both clamp to the
        # nearest known step outside the known range, giving STARS's left/right hold.
        before = (torch.searchsorted(known, step, right=True) - 1).clamp(0, len(known) - 1)
        after = torch.searchsorted(known, step).clamp(0, len(known) - 1)
        t0, t1 = known[before].to(xy.dtype), known[after].to(xy.dtype)
        weight = torch.where(t1 > t0, (step.to(xy.dtype) - t0) / (t1 - t0).clamp(min=1.0), torch.zeros_like(t0))
        value = (1.0 - weight[:, None]) * xy[known[before], agent] + weight[:, None] * xy[known[after], agent]
        filled.append(torch.where(valid[:, agent, None], xy[:, agent], value))
    return torch.stack(filled, dim=1)


def headings_from_motion(xy: torch.Tensor) -> torch.Tensor:
    """Direction of travel per step ``[T, N]``.

    STARS's ``_xy_to_theta`` takes the angle of the last non-zero displacement, and 0 (the scene
    frame's x axis) until an agent first moves, which makes early-step features depend on the scene
    frame; here the steps before an agent's first movement take that first movement's direction.
    An agent that never moves keeps 0.
    """
    displacement = xy[1:] - xy[:-1]
    moving = displacement.abs().sum(dim=-1) > MOTION_EPSILON
    angle = torch.atan2(displacement[..., 1], displacement[..., 0])
    theta = [torch.zeros_like(xy[0, :, 0])]
    for step in range(displacement.shape[0]):
        theta.append(torch.where(moving[step], angle[step], theta[-1]))
    theta = torch.stack(theta)
    if displacement.shape[0]:
        has_moved = moving.any(dim=0)
        first = torch.where(has_moved, moving.float().argmax(dim=0), torch.zeros_like(has_moved, dtype=torch.long))
        initial = angle.gather(0, first[None]).squeeze(0)
        step = torch.arange(xy.shape[0], device=xy.device)[:, None]
        before_first = has_moved[None] & (step <= first[None])
        theta = torch.where(before_first, initial[None], theta)
    return theta


def rotate(xy: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate ``[..., 2]`` vectors counter-clockwise by ``angle`` (broadcast over leading dims)."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([cos * xy[..., 0] - sin * xy[..., 1], sin * xy[..., 0] + cos * xy[..., 1]], dim=-1)


def node_features(xy: torch.Tensor, theta: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-node local-frame motion features ``[N, T, 7]`` and each node's frame ``(origin [N, 2], heading [N])``."""
    origin, heading = xy[-1], theta[-1]
    local = rotate(xy - origin, -heading)  # [T, N, 2]
    velocity = torch.zeros_like(local)
    velocity[1:] = (local[1:] - local[:-1]) / dt
    speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
    relative = theta - heading
    features = torch.cat([local, velocity, speed, torch.cos(relative)[..., None], torch.sin(relative)[..., None]], dim=-1)
    return features.transpose(0, 1), origin, heading


def edge_features(xy: torch.Tensor, theta: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """STARS SE(2) relative pose plus per-step delta for every edge and step: ``[E, T, 8]``.

    Same quantities as ``stars_data.edge_features`` (checked against it in the tests).
    """
    source, destination = edge_index
    cos_i, sin_i = torch.cos(theta[:, source]), torch.sin(theta[:, source])
    cos_j, sin_j = torch.cos(theta[:, destination]), torch.sin(theta[:, destination])
    dx = xy[:, destination, 0] - xy[:, source, 0]
    dy = xy[:, destination, 1] - xy[:, source, 1]
    relative = torch.stack(
        [cos_i * dx + sin_i * dy, -sin_i * dx + cos_i * dy,
         cos_j * cos_i + sin_j * sin_i, sin_j * cos_i - cos_j * sin_i],
        dim=-1,
    )  # [T, E, 4]
    delta = torch.cat([torch.zeros_like(relative[:1]), relative[1:] - relative[:-1]], dim=0)
    return torch.cat([relative, delta], dim=-1).transpose(0, 1)


def neighbourhood_edges(
    position: torch.Tensor, node_valid: torch.Tensor, *, knn: int | None, radius: float | None
) -> torch.Tensor:
    """Directed ``[2, E]`` edges, no self-loops, between nodes that share a valid step.

    ``knn=None`` and ``radius=None`` give STARS's fully connected graph. Otherwise each destination
    keeps its ``knn`` nearest sources within ``radius`` metres at the last observed step.
    """
    n_nodes = position.shape[0]
    device = position.device
    with torch.no_grad():
        overlap = (node_valid.float() @ node_valid.float().T) > 0
        allowed = overlap & ~torch.eye(n_nodes, dtype=torch.bool, device=device)
        if radius is not None or knn is not None:
            distance = torch.cdist(position.detach(), position.detach())  # [source, destination]
            if radius is not None:
                allowed &= distance <= radius
            if knn is not None and knn < n_nodes - 1:
                ranked = distance.masked_fill(~allowed, math.inf)
                nearest = ranked.topk(min(knn, n_nodes), dim=0, largest=False).indices  # per destination
                chosen = torch.zeros_like(allowed)
                chosen.scatter_(0, nearest, True)
                allowed &= chosen
        source, destination = torch.nonzero(allowed, as_tuple=True)
    order = torch.argsort(source * n_nodes + destination)  # STARS's (i, j) for i for j order
    return torch.stack([source[order], destination[order]])


def build_graph(
    xy: torch.Tensor,
    valid: torch.Tensor,
    *,
    dt: float,
    knn: int | None = None,
    radius: float | None = None,
) -> dict[str, torch.Tensor]:
    """One graph from the observed window of already-selected nodes.

    ``xy`` is ``[T_obs, N, 2]`` (NaN allowed where invalid), ``valid`` is ``[T_obs, N]`` and every node
    must be valid at the last step. Returns ``node_type [N]``, ``node_attr [N, T, 7]``,
    ``valid_mask [N, T]``, ``edge_index [2, E]``, ``edge_attr [E, T, 8]`` and each node's local frame
    ``origin [N, 2]`` / ``heading [N]`` in the input frame.
    """
    if not bool(valid[-1].all()):
        raise ValueError("Every node must be visible at the last observed step")
    filled = fill_observed(torch.nan_to_num(xy), valid)
    theta = headings_from_motion(filled)
    nodes, origin, heading = node_features(filled, theta, dt)
    edge_index = neighbourhood_edges(filled[-1], valid.T, knn=knn, radius=radius)
    return {
        "node_type": torch.full((xy.shape[1],), NODE_TYPE_HUMAN, dtype=torch.long, device=xy.device),
        "node_attr": nodes,
        "valid_mask": valid.T.to(xy.dtype),
        "edge_index": edge_index,
        "edge_attr": edge_features(filled, theta, edge_index),
        "origin": origin,
        "heading": heading,
    }


def to_local(xy: torch.Tensor, origin: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Scene-frame ``[N, T, 2]`` positions into each node's local frame."""
    return rotate(xy - origin[:, None], -heading[:, None])


def to_scene(local: torch.Tensor, origin: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Local-frame ``[..., N, T, 2]`` positions back into the scene frame."""
    return rotate(local, heading[:, None]) + origin[:, None]
