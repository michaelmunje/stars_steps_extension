"""Batched planar geometry: distances and closest points to circles and line segments."""
from __future__ import annotations

import torch

FAR = 1e6


def segment_closest(points: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """Closest point on each segment for each point: ``points [..., Q, 2]``, ``segments [..., S, 4]`` ->
    ``[..., Q, S, 2]`` (leading dims broadcast)."""
    a, b = segments[..., None, :, :2], segments[..., None, :, 2:]
    p = points[..., :, None, :]
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)
    return a + t.clamp(0.0, 1.0)[..., None] * ab


def obstacle_clearance(points: torch.Tensor, circles: torch.Tensor, circle_mask: torch.Tensor,
                       segments: torch.Tensor, segment_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance from each point to the nearest obstacle surface and the unit direction *away*
    from it: ``points [B, Q, 2]`` -> ``clearance [B, Q]``, ``away [B, Q, 2]`` (FAR when no obstacle)."""
    B, Q = points.shape[:2]
    best = points.new_full((B, Q), FAR)
    away = points.new_zeros((B, Q, 2))
    if circles.shape[1]:
        rel = points[:, :, None] - circles[:, None, :, :2]                       # [B, Q, C, 2]
        dist = torch.linalg.norm(rel, dim=-1)
        clear = torch.where(circle_mask[:, None], dist - circles[:, None, :, 2], FAR)
        value, index = clear.min(dim=-1)
        direction = rel / dist.clamp(min=1e-9)[..., None]
        picked = torch.gather(direction, 2, index[..., None, None].expand(B, Q, 1, 2))[:, :, 0]
        better = value < best
        best, away = torch.where(better, value, best), torch.where(better[..., None], picked, away)
    if segments.shape[1]:
        closest = segment_closest(points, segments)                              # [B, Q, S, 2]
        rel = points[:, :, None] - closest
        dist = torch.linalg.norm(rel, dim=-1)
        clear = torch.where(segment_mask[:, None], dist, FAR)
        value, index = clear.min(dim=-1)
        direction = rel / dist.clamp(min=1e-9)[..., None]
        picked = torch.gather(direction, 2, index[..., None, None].expand(B, Q, 1, 2))[:, :, 0]
        better = value < best
        best, away = torch.where(better, value, best), torch.where(better[..., None], picked, away)
    return best, away


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rectangle_segments(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Four wall segments of an axis-aligned rectangle."""
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]


def polygon_segments(vertices) -> list[list[float]]:
    """Edges of a closed polygon given as ``[(x, y), ...]``."""
    return [[*vertices[i], *vertices[(i + 1) % len(vertices)]] for i in range(len(vertices))]
