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


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def crosses_segments(old: torch.Tensor, new: torch.Tensor, segments: torch.Tensor,
                     segment_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Whether each move ``old -> new`` (``[B, Q, 2]``) crosses a wall (``segments [B, S, 4]``):
    ``crossed [B, Q]`` and the unit normal of the first wall crossed ``[B, Q, 2]``."""
    a, b = segments[:, None, :, :2], segments[:, None, :, 2:]                  # [B, 1, S, 2]
    p, r = old[:, :, None], (new - old)[:, :, None]                            # [B, Q, 1, 2]
    s = b - a
    denom = _cross(r, s)
    ok = denom.abs() > 1e-12
    safe = torch.where(ok, denom, torch.ones_like(denom))
    t, u = _cross(a - p, s) / safe, _cross(a - p, r) / safe
    hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & segment_mask[:, None]
    first = torch.where(hit, t, torch.full_like(t, 2.0)).argmin(-1)             # [B, Q]
    tangent = torch.gather(s.expand(*hit.shape, 2), 2, first[..., None, None].expand(*first.shape, 1, 2))[:, :, 0]
    tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-12)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], -1)
    return hit.any(-1), normal


def block_wall_crossings(old: torch.Tensor, new: torch.Tensor, segments: torch.Tensor,
                         segment_mask: torch.Tensor) -> torch.Tensor:
    """Moves that would pass through a wall slide along it instead (or stop at a corner)."""
    if not segments.shape[1]:
        return new
    crossed, normal = crosses_segments(old, new, segments, segment_mask)
    move = new - old
    slide = old + move - (move * normal).sum(-1, keepdim=True) * normal
    out = torch.where(crossed[..., None], slide, new)
    again, _ = crosses_segments(old, out, segments, segment_mask)
    return torch.where((crossed & again)[..., None], old, out)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rectangle_segments(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Four wall segments of an axis-aligned rectangle."""
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]

