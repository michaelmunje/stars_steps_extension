"""Model C: Optimal Reciprocal Collision Avoidance (van den Berg et al. 2011), batched over environments
and pedestrians. A port of RVO2's ``Agent::computeNewVelocity`` and its linear programs.

Every step each pedestrian picks the velocity closest to its preferred velocity inside its
max-speed disc that satisfies one half-plane ("ORCA line") per nearby interaction:

* other pedestrians (the ``max_neighbors`` nearest within ``neighbor_dist``): reciprocal, each
  takes half of the avoidance (``u / 2``), with truncation horizon ``time_horizon``; when two
  already overlap, the cut-off is the ``1 / dt`` circle, as in RVO2;
* the robot: a *non-reciprocal* moving disc (``robot_radius``, its current velocity); pedestrians
  take full responsibility (``u``), horizon ``robot_time_horizon``; ignored where ``robot_present``
  is False (the counterfactual);
* static obstacles, horizon ``time_horizon_obst``: wall segments use RVO2's obstacle lines for
  a two-vertex (isolated, convex) obstacle; circles are static discs (full responsibility).

The half-planes are solved with RVO2's incremental 2-D linear program (``linearProgram2`` /
``linearProgram1``). When they are infeasible (dense crowds) ``linearProgram3`` minimises the
maximum penetration of the pedestrian and robot lines while keeping obstacle lines hard.
Python loops run only over constraint slots, never over environments or pedestrians.

Preferred velocities: ``mode="reactive_replay"`` tracks the reference trajectory exactly like
``ReactiveReplayModel`` (``v_ref + tracking_gain * (x_ref - x)``, then walk to the reference's end
and leave); ``mode="goal"`` walks to the reference's end point at the preferred speed like
``SocialForceModel``. Spawn / finish / done bookkeeping matches those models. After the velocity
update, ``social_force.resolve_robot_contact`` guarantees no pedestrian overlaps the robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from simplenavsim.core.geometry import segment_closest
from simplenavsim.core.state import PedestrianState
from simplenavsim.pedestrians.base import PedestrianModel, clip_norm, reference_velocity
from simplenavsim.pedestrians.social_force import resolve_robot_contact

EPSILON = 1e-5  # RVO_EPSILON


@dataclass
class OrcaParams:
    neighbor_dist: float = 5.0
    max_neighbors: int = 10
    time_horizon: float = 2.0
    time_horizon_obst: float = 1.0
    robot_time_horizon: float = 2.0
    max_speed_factor: float = 1.3     # max speed = factor * preferred speed
    mode: str = "reactive_replay"     # "reactive_replay" (track the reference) or "goal"
    tracking_gain: float = 1.0        # reactive_replay mode (1/s)
    max_overrun: float = 10.0         # reactive_replay mode: s a pedestrian may run past its reference
    arrival_distance: float = 0.3
    obstacles: bool = True            # ORCA lines for static segments and circles


# ----------------------------------------------------------------------------- vector helpers


def _det(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)


def _perp(v: torch.Tensor) -> torch.Tensor:
    """``(-y, x)``: the direction 90 degrees to the left."""
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def _where(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[..., None], a, b)


# ----------------------------------------------------------------------------- ORCA lines


def velocity_obstacle_lines(rel_pos, rel_vel, radius, inv_horizon: float, inv_dt: float, vel, share: float):
    """ORCA lines for discs (RVO2's agent case). ``rel_pos`` other minus self, ``rel_vel`` self minus
    other, ``radius`` combined; ``share`` 0.5 for reciprocal pedestrians, 1 for the robot and static
    discs. Returns ``point, direction [..., 2]``; the permitted side is to the left of ``direction``."""
    dist_sq = _dot(rel_pos, rel_pos)
    radius_sq = radius * radius
    w = rel_vel - inv_horizon * rel_pos
    w_sq = _dot(w, w)
    w_dot = _dot(w, rel_pos)
    cutoff = (w_dot < 0) & (w_dot * w_dot > radius_sq * w_sq)
    w_len = torch.sqrt(w_sq.clamp(min=1e-24))
    unit_w = w / w_len[..., None]
    cut_dir = torch.stack([unit_w[..., 1], -unit_w[..., 0]], -1)
    cut_u = (radius * inv_horizon - w_len)[..., None] * unit_w

    leg = torch.sqrt((dist_sq - radius_sq).clamp(min=0.0))
    x, y = rel_pos[..., 0], rel_pos[..., 1]
    left = torch.stack([x * leg - y * radius, x * radius + y * leg], -1) / dist_sq.clamp(min=1e-12)[..., None]
    right = -torch.stack([x * leg + y * radius, -x * radius + y * leg], -1) / dist_sq.clamp(min=1e-12)[..., None]
    leg_dir = _where(_det(rel_pos, w) > 0, left, right)
    leg_u = _dot(rel_vel, leg_dir)[..., None] * leg_dir - rel_vel

    w_c = rel_vel - inv_dt * rel_pos                                     # already overlapping
    w_c_len = torch.linalg.norm(w_c, dim=-1).clamp(min=1e-12)
    unit_c = w_c / w_c_len[..., None]
    col_dir = torch.stack([unit_c[..., 1], -unit_c[..., 0]], -1)
    col_u = (radius * inv_dt - w_c_len)[..., None] * unit_c

    apart = dist_sq > radius_sq
    direction = _where(apart, _where(cutoff, cut_dir, leg_dir), col_dir)
    u = _where(apart, _where(cutoff, cut_u, leg_u), col_u)
    return vel + share * u, direction


def segment_lines(pos, vel, radius, max_speed, segments, segment_mask, inv_horizon: float):
    """RVO2 obstacle lines for isolated wall segments (two-vertex convex obstacles), sorted by
    distance with RVO2's "already covered" test. ``pos, vel [N, 2]``, ``radius, max_speed [N]``,
    ``segments [N, S, 4]`` -> ``point, direction [N, S, 2]``, ``valid [N, S]``."""
    a, b = segments[..., :2], segments[..., 2:]
    p = pos[:, None]
    # Use the directed edge that has the agent on its right (RVO2 only sees that side).
    left_of = _det(a - p, b - a)
    flip = left_of > 0
    a, b = _where(flip, b, a), _where(flip, a, b)
    closest = segment_closest(pos[:, None], torch.cat([a, b], -1))[:, 0]   # [N, S, 2]
    dist_sq = _dot(closest - p, closest - p)
    reach = max_speed / inv_horizon + radius
    seen = segment_mask & (left_of != 0) & (dist_sq < (reach * reach)[:, None])
    order = torch.argsort(torch.where(seen, dist_sq, torch.inf), dim=1)
    take = lambda t: torch.gather(t, 1, order[..., None].expand_as(t)) if t.dim() == 3 else torch.gather(t, 1, order)
    a, b, seen = take(a), take(b), take(seen)

    r = radius[:, None]
    r_sq = r * r
    v = vel[:, None]
    rp1, rp2 = a - p, b - p
    d_a = _normalize(b - a)                                             # direction of vertex a; b's is -d_a
    edge = b - a
    s = _dot(-rp1, edge) / _dot(edge, edge).clamp(min=1e-12)
    d1, d2 = _dot(rp1, rp1), _dot(rp2, rp2)
    d_line = _dot(-rp1 - s[..., None] * edge, -rp1 - s[..., None] * edge)

    zero = torch.zeros_like(rp1)
    hit_left = (s < 0) & (d1 <= r_sq)
    hit_right = ~hit_left & (s > 1) & (d2 <= r_sq)
    hit_edge = ~hit_left & ~hit_right & (s >= 0) & (s <= 1) & (d_line <= r_sq)
    hit = hit_left | hit_right | hit_edge
    hit_dir = _where(hit_left, _normalize(_perp(rp1)), _where(hit_right, _normalize(_perp(rp2)), -d_a))
    hit_valid = hit_left | (hit_right & (_det(rp2, -d_a) >= 0)) | hit_edge

    oblique_left = (s < 0) & (d_line <= r_sq)
    oblique_right = (s > 1) & (d_line <= r_sq)
    same = oblique_left | oblique_right
    q1 = _where(oblique_right, rp2, rp1)
    q2 = _where(oblique_left, rp1, rp2)
    dir1 = _where(oblique_right, -d_a, d_a)                              # obstacle1's own direction
    dir2 = _where(oblique_left, d_a, -d_a)                               # obstacle2's own direction

    def legs(q, sign):
        q_sq = _dot(q, q).clamp(min=1e-12)
        leg = torch.sqrt((q_sq - r_sq).clamp(min=0.0))
        x, y = q[..., 0], q[..., 1]
        return torch.stack([x * leg - sign * y * r, sign * x * r + y * leg], -1) / q_sq[..., None]

    left_leg, right_leg = legs(q1, 1.0), legs(q2, -1.0)
    left_foreign = _det(left_leg, dir1) >= 0                              # leg points into the edge
    right_foreign = _det(right_leg, dir2) <= 0
    left_leg = _where(left_foreign, dir1, left_leg)
    right_leg = _where(right_foreign, dir2, right_leg)

    ih = inv_horizon
    left_cut, right_cut = ih * q1, ih * q2
    cut_vec = right_cut - left_cut
    t = torch.where(same, 0.5, _dot(v - left_cut, cut_vec) / _dot(cut_vec, cut_vec).clamp(min=1e-12))
    t_left = _dot(v - left_cut, left_leg)
    t_right = _dot(v - right_cut, right_leg)

    on_left_circle = ((t < 0) & (t_left < 0)) | (same & (t_left < 0) & (t_right < 0))
    on_right_circle = ~on_left_circle & (t > 1) & (t_right < 0)
    unit_l, unit_r = _normalize(v - left_cut), _normalize(v - right_cut)
    circle_dir = _where(on_left_circle, torch.stack([unit_l[..., 1], -unit_l[..., 0]], -1),
                        torch.stack([unit_r[..., 1], -unit_r[..., 0]], -1))
    circle_point = _where(on_left_circle, left_cut + r[..., None] * ih * unit_l, right_cut + r[..., None] * ih * unit_r)

    inf = torch.full_like(t, torch.inf)
    sq = lambda e: _dot(e, e)
    dist_cut = torch.where((t < 0) | (t > 1) | same, inf, sq(v - (left_cut + t[..., None] * cut_vec)))
    dist_left = torch.where(t_left < 0, inf, sq(v - (left_cut + t_left[..., None] * left_leg)))
    dist_right = torch.where(t_right < 0, inf, sq(v - (right_cut + t_right[..., None] * right_leg)))
    on_cut = (dist_cut <= dist_left) & (dist_cut <= dist_right)
    on_left = ~on_cut & (dist_left <= dist_right)
    leg_dir = _where(on_cut, -dir1, _where(on_left, left_leg, -right_leg))
    leg_point = _where(on_cut | on_left, left_cut, right_cut) + r[..., None] * ih * _perp(leg_dir)
    leg_valid = ~(on_left & left_foreign) & ~(~on_cut & ~on_left & right_foreign)

    circle = on_left_circle | on_right_circle
    point = _where(hit, zero, _where(circle, circle_point, leg_point))
    direction = _where(hit, hit_dir, _where(circle, circle_dir, leg_dir))
    valid = seen & torch.where(hit, hit_valid, circle | leg_valid)

    # Skip an edge whose velocity obstacle earlier (nearer) lines already cover.
    S = segments.shape[1]
    for k in range(1, S):
        earlier = valid[:, :k]
        rad = (ih * radius)[:, None]
        c1 = _det(ih * rp1[:, k:k + 1] - point[:, :k], direction[:, :k]) - rad >= -EPSILON
        c2 = _det(ih * rp2[:, k:k + 1] - point[:, :k], direction[:, :k]) - rad >= -EPSILON
        covered = (earlier & c1 & c2).any(dim=1)
        valid[:, k] &= ~covered
    return point, direction, valid


# ----------------------------------------------------------------------------- linear programs


def linear_program1(point, direction, valid, i: int, radius, opt, direction_opt: bool):
    """Optimum on line ``i`` subject to lines ``< i`` and the speed disc; ``ok [N]``, ``result [N, 2]``."""
    p_i, d_i = point[:, i], direction[:, i]
    dot = _dot(p_i, d_i)
    disc = dot * dot + radius * radius - _dot(p_i, p_i)
    ok = disc >= 0
    root = torch.sqrt(disc.clamp(min=0.0))
    t_left, t_right = -dot - root, -dot + root
    if i:
        p_j, d_j, v_j = point[:, :i], direction[:, :i], valid[:, :i]
        den = _det(d_i[:, None], d_j)
        num = _det(d_j, p_i[:, None] - p_j)
        parallel = den.abs() <= EPSILON
        ok &= ~(v_j & parallel & (num < 0)).any(dim=1)
        t = num / torch.where(parallel, 1.0, den)
        use = v_j & ~parallel
        t_right = torch.minimum(t_right, torch.where(use & (den >= 0), t, torch.inf).min(dim=1).values)
        t_left = torch.maximum(t_left, torch.where(use & (den < 0), t, -torch.inf).max(dim=1).values)
        ok &= t_left <= t_right
    if direction_opt:
        t = torch.where(_dot(opt, d_i) > 0, t_right, t_left)
    else:
        t = torch.minimum(torch.maximum(_dot(d_i, opt - p_i), t_left), t_right)
    return ok, p_i + t[:, None] * d_i


def linear_program2(point, direction, valid, radius, opt, direction_opt: bool = False):
    """RVO2's incremental 2-D LP over ``L`` line slots (invalid slots are ignored). Returns the index of
    the first infeasible line (``L`` on success) and the result (the last feasible one on failure)."""
    if direction_opt:
        result = opt * radius[:, None]
    else:
        norm = torch.linalg.norm(opt, dim=-1, keepdim=True)
        result = torch.where(norm > radius[:, None], opt / norm.clamp(min=1e-12) * radius[:, None], opt)
    L = point.shape[1]
    fail = torch.full(radius.shape, L, dtype=torch.long, device=radius.device)
    for i in range(L):
        violated = valid[:, i] & (fail == L) & (_det(direction[:, i], point[:, i] - result) > 0)
        if not bool(violated.any()):
            continue
        ok, candidate = linear_program1(point, direction, valid, i, radius, opt, direction_opt)
        result = _where(violated & ok, candidate, result)
        fail = torch.where(violated & ~ok, i, fail)
    return fail, result


def linear_program3(point, direction, valid, n_obstacle: int, begin, radius, result):
    """RVO2's fallback for infeasible programs: minimise the largest violation of the soft lines
    (slots ``>= n_obstacle``) while the obstacle slots stay hard constraints."""
    L = point.shape[1]
    slot = torch.arange(L, device=point.device)
    soft = slot >= n_obstacle
    distance = torch.zeros_like(radius)
    result = result.clone()
    for i in range(L):
        need = (i >= begin) & valid[:, i] & (_det(direction[:, i], point[:, i] - result) > distance)
        if not bool(need.any()):
            continue
        rows = need.nonzero().squeeze(1)
        p_i, d_i = point[rows, i][:, None], direction[rows, i][:, None]
        p_j, d_j = point[rows], direction[rows]
        determinant = _det(d_i, d_j)
        parallel = determinant.abs() <= EPSILON
        proj_point = torch.where(parallel[..., None], 0.5 * (p_i + p_j),
                                 p_i + (_det(d_j, p_i - p_j) / torch.where(parallel, 1.0, determinant))[..., None] * d_i)
        proj_dir = _normalize(d_j - d_i)
        keep = soft & (slot < i) & valid[rows] & ~(parallel & (_dot(d_i, d_j) > 0))
        proj_point = _where(soft.expand_as(keep), proj_point, p_j)
        proj_dir = _where(soft.expand_as(keep), proj_dir, d_j)
        proj_valid = torch.where(soft, keep, valid[rows])
        fail, candidate = linear_program2(proj_point, proj_dir, proj_valid, radius[rows], _perp(d_i[:, 0]), True)
        candidate = _where(fail < L, result[rows], candidate)  # numerical failure: keep the current result
        result[rows] = candidate
        distance[rows] = _det(d_i[:, 0], p_i[:, 0] - candidate)
    return result


# ----------------------------------------------------------------------------- velocities


def orca_velocities(pos, vel, pref, active, radius, max_speed, dt: float, p: OrcaParams, *,
                    robot_pos=None, robot_vel=None, robot_radius: float = 0.0, robot_present=None,
                    circles=None, circle_mask=None, segments=None, segment_mask=None) -> torch.Tensor:
    """New velocities ``[B, P, 2]`` for pedestrians at ``pos, vel [B, P, 2]`` wanting ``pref``. Only
    ``active`` pedestrians are neighbours; inactive ones get zero velocity."""
    B, P = pos.shape[:2]
    N = B * P
    dev, dtype = pos.device, pos.dtype
    flat = lambda t: t.reshape(N, *t.shape[2:])
    points, directions, valids = [], [], []

    # Static obstacles (hard constraints, first as in RVO2).
    inv_obst = 1.0 / p.time_horizon_obst
    if p.obstacles and segments is not None and segments.shape[1]:
        S = segments.shape[1]
        seg = segments[:, None].expand(B, P, S, 4).reshape(N, S, 4)
        mask = segment_mask[:, None].expand(B, P, S).reshape(N, S)
        pt, dr, vd = segment_lines(flat(pos), flat(vel), flat(radius), flat(max_speed), seg, mask, inv_obst)
        points.append(pt), directions.append(dr), valids.append(vd)
    if p.obstacles and circles is not None and circles.shape[1]:
        rel = circles[:, None, :, :2] - pos[:, :, None]                                   # [B, P, C, 2]
        combined = radius[:, :, None] + circles[:, None, :, 2]
        pt, dr = velocity_obstacle_lines(rel, vel[:, :, None].expand_as(rel), combined, inv_obst, 1.0 / dt,
                                         vel[:, :, None], 1.0)
        reach = p.time_horizon_obst * max_speed + radius
        near = torch.linalg.norm(rel, dim=-1) - circles[:, None, :, 2] < reach[..., None]
        points.append(flat(pt)), directions.append(flat(dr)), valids.append(flat(circle_mask[:, None] & near))
    n_obstacle = sum(t.shape[1] for t in points)

    # The robot: a moving obstacle the pedestrians fully avoid.
    if robot_pos is not None:
        rel = (robot_pos[:, None] - pos)[:, :, None]
        rel_vel = (vel - robot_vel[:, None])[:, :, None]
        pt, dr = velocity_obstacle_lines(rel, rel_vel, (radius + robot_radius)[..., None], 1.0 / p.robot_time_horizon,
                                         1.0 / dt, vel[:, :, None], 1.0)
        near = (_dot(rel, rel) < p.neighbor_dist ** 2) & robot_present[:, None, None]
        points.append(flat(pt)), directions.append(flat(dr)), valids.append(flat(near))

    # Other pedestrians: the nearest ``max_neighbors`` within ``neighbor_dist``, nearest first.
    M = min(p.max_neighbors, P - 1)
    if M > 0:
        diff = pos[:, None] - pos[:, :, None]                                             # [B, i, j, 2] j - i
        dist_sq = _dot(diff, diff)
        pair = active[:, :, None] & active[:, None] & ~torch.eye(P, dtype=torch.bool, device=dev)
        dist_sq = torch.where(pair & (dist_sq < p.neighbor_dist ** 2), dist_sq, torch.inf)
        nearest, index = torch.topk(dist_sq, M, dim=-1, largest=False, sorted=True)       # [B, P, M]
        gather = lambda t: torch.gather(t[:, None].expand(B, P, P, *t.shape[2:]), 2,
                                        index.reshape(B, P, M, *([1] * (t.dim() - 2))).expand(B, P, M, *t.shape[2:]))
        rel = gather(pos) - pos[:, :, None]
        rel_vel = vel[:, :, None] - gather(vel)
        combined = radius[:, :, None] + gather(radius)
        pt, dr = velocity_obstacle_lines(rel, rel_vel, combined, 1.0 / p.time_horizon, 1.0 / dt, vel[:, :, None], 0.5)
        points.append(flat(pt)), directions.append(flat(dr)), valids.append(flat(torch.isfinite(nearest)))

    speed = flat(max_speed).to(dtype)
    if not points:
        new = clip_norm(flat(pref), speed)
    else:
        point, direction, valid = torch.cat(points, 1), torch.cat(directions, 1), torch.cat(valids, 1)
        valid = valid & flat(active)[:, None]
        fail, new = linear_program2(point, direction, valid, speed, flat(pref))
        failed = fail < point.shape[1]
        if bool(failed.any()):
            new = linear_program3(point, direction, valid, n_obstacle, fail, speed, new)
    new = new.reshape(B, P, 2)
    return torch.where(active[..., None], new, torch.zeros((), dtype=dtype, device=dev))


class OrcaModel(PedestrianModel):
    """ORCA pedestrians; see the module docstring. ``OrcaModel(**kwargs)`` sets ``OrcaParams`` fields."""

    name = "orca"

    def __init__(self, params: OrcaParams | None = None, **kwargs) -> None:
        self.params = params or OrcaParams(**kwargs)
        if self.params.mode not in ("reactive_replay", "goal"):
            raise ValueError(f"ORCA mode must be 'reactive_replay' or 'goal', not {self.params.mode!r}")

    def step(self, state, batch, k, robot, robot_radius, robot_present) -> PedestrianState:
        p = self.params
        K, dt = batch.steps, batch.dt
        kk = min(k, K - 1)
        spawn = (self.first == k) & batch.ped_mask & ~state.active & ~state.done
        final = torch.gather(batch.ped_ref, 2, self.last.clamp(min=0)[:, :, None, None].expand(-1, -1, 1, 2))[:, :, 0]
        to_final = final - state.pos
        distance = torch.linalg.norm(to_final, dim=-1, keepdim=True)
        walk = to_final / distance.clamp(min=1e-9) * torch.minimum(batch.ped_speed[..., None], distance / dt)
        previous = max(0, kk - 1)
        ref_vel = reference_velocity(batch, previous)
        if p.mode == "reactive_replay":
            tracking = state.active & (k <= self.last)
            limit = p.max_speed_factor * torch.maximum(batch.ped_speed, torch.linalg.norm(ref_vel, dim=-1))
            track = clip_norm(ref_vel + p.tracking_gain * (batch.ped_ref[:, :, previous] - state.pos), limit)
            preferred = torch.where(tracking[..., None], track, walk)
        else:
            limit = p.max_speed_factor * batch.ped_speed
            preferred = walk
        vel = orca_velocities(state.pos, state.vel, preferred, state.active, batch.ped_radius, limit, dt, p,
                              robot_pos=robot.pos, robot_vel=robot.velocity, robot_radius=robot_radius,
                              robot_present=robot_present, circles=batch.circles, circle_mask=batch.circle_mask,
                              segments=batch.segments, segment_mask=batch.segment_mask)
        pos = state.pos + vel * dt
        pos, vel = resolve_robot_contact(pos, vel, batch, robot, robot_radius, robot_present)
        if p.mode == "reactive_replay":
            overrun = k - self.last
            finished = state.active & (overrun > 0) & (
                (torch.linalg.norm(final - pos, dim=-1) < p.arrival_distance) | (overrun * dt > p.max_overrun))
            pos = torch.where(spawn[..., None], batch.ped_ref[:, :, kk], pos)
            vel = torch.where(spawn[..., None], ref_vel, vel)
        else:
            start = torch.gather(batch.ped_ref, 2, self.first.clamp(max=K - 1)[:, :, None, None]
                                 .expand(-1, -1, 1, 2))[:, :, 0]
            stationary = torch.linalg.norm(final - start, dim=-1) < p.arrival_distance
            finished = state.active & (torch.linalg.norm(final - pos, dim=-1) < p.arrival_distance) & ~stationary
            pos = torch.where(spawn[..., None], batch.ped_ref[:, :, k], pos)
            vel = torch.where(spawn[..., None], 0.0, vel)
        active = (state.active & ~finished) | spawn
        return PedestrianState(pos, torch.where(active[..., None], vel, 0.0), active, state.done | finished)
