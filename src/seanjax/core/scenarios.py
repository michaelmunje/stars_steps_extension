"""Navigation scenarios: seeded, randomised generators and batching.

Every pedestrian, scripted or recorded, is described by a *reference trajectory* on the
simulation time grid (``ped_ref [P, K, 2]`` with presence ``ped_valid [P, K]``). Pedestrian models
interpret it: replay follows it exactly, the social force model walks to its end point, reactive
replay tracks it while avoiding the robot. The robot drives from ``robot_start`` to
``robot_goal`` within ``duration``.

Scenario kinds (``KINDS``): empty, crossing, head_on, overtaking, following, group_toward,
narrow_passage, stationary_group, dense_flow, recorded. Randomised quantities are drawn from a
``numpy`` generator seeded by ``(kind, seed)``; ``scenario_set`` gives disjoint seed ranges per
split so train / validation / test scenarios never coincide. ``recorded`` scenarios take windows of
real traffic from a ``datasets.trajectory_loader.RecordingPool`` whose recordings are split by
recording, so no trajectory crosses splits.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch

from seanjax.core.geometry import rectangle_segments
from seanjax.core.state import ScenarioBatch

DEFAULT_DT = 0.1
SPLIT_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}
PED_RADIUS = 0.25


@dataclass
class Scenario:
    name: str
    kind: str
    seed: int
    dt: float
    duration: float
    robot_start: np.ndarray            # [3]
    robot_goal: np.ndarray             # [2]
    ped_ref: np.ndarray                # [P, K, 2]
    ped_valid: np.ndarray              # [P, K]
    ped_radius: np.ndarray             # [P]
    ped_speed: np.ndarray              # [P]
    circles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    segments: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    meta: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(round(self.duration / self.dt)) + 1


def _rng(kind: str, seed: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(kind.encode()), seed])


def _walker(start, goal, speed: float, t0: float, dt: float, steps: int, *, stay: bool = False):
    """A pedestrian walking a straight line at ``speed`` from time ``t0``; it leaves at the goal
    (or stays there with ``stay``). Before ``t0`` it is not in the scene."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    t = np.arange(steps) * dt
    length = float(np.linalg.norm(goal - start))
    travel = length / max(speed, 1e-6)
    s = np.clip((t - t0) / max(travel, 1e-9), 0.0, 1.0)
    ref = start + s[:, None] * (goal - start)
    valid = (t >= t0) & ((t <= t0 + travel) | stay)
    return ref.astype(np.float32), valid


def _stationary(position, steps: int):
    return np.repeat(np.asarray(position, np.float32)[None], steps, 0), np.ones(steps, bool)


def _robot(rng, length: float, lateral: float = 0.0):
    start = np.array([0.0, rng.uniform(-lateral, lateral), rng.normal(0.0, 0.1)], np.float32)
    goal = np.array([length, rng.uniform(-lateral, lateral)], np.float32)
    return start, goal


def _finish(name, kind, seed, dt, duration, start, goal, walkers, speeds, *, circles=None, segments=None, **meta):
    steps = int(round(duration / dt)) + 1
    if walkers:
        ref = np.stack([w[0] for w in walkers]).astype(np.float32)
        valid = np.stack([w[1] for w in walkers])
    else:
        ref, valid = np.zeros((0, steps, 2), np.float32), np.zeros((0, steps), bool)
    p = len(walkers)
    return Scenario(name, kind, seed, dt, duration, start, goal, ref, valid,
                    np.full(p, PED_RADIUS, np.float32), np.asarray(speeds, np.float32).reshape(p),
                    np.asarray(circles if circles is not None else np.zeros((0, 3)), np.float32).reshape(-1, 3),
                    np.asarray(segments if segments is not None else np.zeros((0, 4)), np.float32).reshape(-1, 4),
                    meta)


def make_scenario(kind: str, seed: int, *, dt: float = DEFAULT_DT, duration: float = 30.0, pool=None,
                  split: str = "train") -> Scenario:
    """One randomised scenario of ``kind``."""
    rng = _rng(kind, seed)
    steps = int(round(duration / dt)) + 1
    name = f"{kind}-{seed}"
    length = rng.uniform(8.0, 12.0)
    start, goal = _robot(rng, length)
    robot_speed = 0.8  # nominal, only used to time encounters
    if kind == "empty":
        return _finish(name, kind, seed, dt, duration, start, goal, [], [])
    if kind == "crossing":
        x_c = rng.uniform(0.35, 0.65) * length
        side = rng.choice([-1.0, 1.0])
        speed = rng.uniform(0.9, 1.4)
        distance = rng.uniform(3.0, 5.0)
        t_c = x_c / robot_speed + rng.normal(0.0, 1.0)
        a, b = [x_c, side * distance], [x_c, -side * distance]
        return _finish(name, kind, seed, dt, duration, start, goal,
                       [_walker(a, b, speed, max(0.0, t_c - distance / speed), dt, steps)], [speed])
    if kind == "head_on":
        speed = rng.uniform(0.9, 1.4)
        y = rng.normal(0.0, 0.25)
        a, b = [length + rng.uniform(1.0, 3.0), y], [-2.0, y]
        return _finish(name, kind, seed, dt, duration, start, goal, [_walker(a, b, speed, 0.0, dt, steps)], [speed])
    if kind == "overtaking":
        speed = rng.uniform(0.35, 0.6)
        a = [rng.uniform(1.5, 3.0), rng.normal(0.0, 0.2)]
        b = [a[0] + speed * duration, a[1]]
        return _finish(name, kind, seed, dt, duration, start, goal, [_walker(a, b, speed, 0.0, dt, steps)], [speed])
    if kind == "following":
        speed = rng.uniform(0.8, 1.1)
        a = [rng.uniform(1.2, 2.0), rng.normal(0.0, 0.15)]
        b = [length + 3.0, a[1]]
        return _finish(name, kind, seed, dt, duration, start, goal, [_walker(a, b, speed, 0.0, dt, steps)], [speed])
    if kind == "group_toward":
        speed = rng.uniform(0.8, 1.2)
        size = int(rng.integers(2, 5))
        x0 = length + rng.uniform(1.0, 3.0)
        lateral = rng.normal(0.0, 0.3)
        offsets = (np.arange(size) - (size - 1) / 2) * rng.uniform(0.6, 0.8)
        walkers = [_walker([x0 + rng.normal(0, 0.15), lateral + o], [-3.0, lateral + o], speed, 0.0, dt, steps)
                   for o in offsets]
        return _finish(name, kind, seed, dt, duration, start, goal, walkers, [speed] * size)
    if kind == "narrow_passage":
        width = rng.uniform(1.4, 2.2)
        x0, x1 = 0.35 * length, 0.65 * length
        segments = [[x0, width / 2, x1, width / 2], [x0, -width / 2, x1, -width / 2],
                    [x0, width / 2, x0, width / 2 + 3.0], [x0, -width / 2, x0, -width / 2 - 3.0],
                    [x1, width / 2, x1, width / 2 + 3.0], [x1, -width / 2, x1, -width / 2 - 3.0]]
        speed = rng.uniform(0.8, 1.2)
        y = rng.normal(0.0, 0.15)
        t0 = rng.uniform(0.0, 6.0)
        walker = _walker([length + 2.0, y], [-2.0, y], speed, t0, dt, steps)
        start[1], goal[1] = 0.0, 0.0
        return _finish(name, kind, seed, dt, duration, start, goal, [walker], [speed], segments=segments, width=width)
    if kind == "stationary_group":
        size = int(rng.integers(3, 6))
        center = np.array([rng.uniform(0.4, 0.6) * length, rng.uniform(-0.4, 0.4)])
        radius = rng.uniform(0.6, 0.9)
        angles = rng.uniform(0, 2 * math.pi) + np.arange(size) * 2 * math.pi / size
        walkers = [_stationary(center + radius * np.array([math.cos(a), math.sin(a)]), steps) for a in angles]
        return _finish(name, kind, seed, dt, duration, start, goal, walkers, [1.2] * size, group_center=center.tolist())
    if kind == "dense_flow":
        half = rng.uniform(2.0, 3.0)
        count = int(rng.integers(10, 21))
        walkers, speeds = [], []
        for i in range(count):
            direction = 1.0 if i % 2 == 0 else -1.0
            speed = rng.uniform(0.8, 1.4)
            y = rng.uniform(-half + 0.4, half - 0.4)
            x_start = rng.uniform(-4.0, length + 4.0)
            t0 = rng.uniform(0.0, 10.0)
            x_end = length + 6.0 if direction > 0 else -6.0
            walkers.append(_walker([x_start, y], [x_end, y], speed, t0, dt, steps))
            speeds.append(speed)
        segments = [[-6.0, half, length + 6.0, half], [-6.0, -half, length + 6.0, -half]]
        start[1], goal[1] = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        return _finish(name, kind, seed, dt, duration, start, goal, walkers, speeds, segments=segments)
    if kind == "recorded":
        if pool is None:
            raise ValueError("recorded scenarios need a RecordingPool (seanjax.datasets.trajectory_loader)")
        return pool.scenario(rng, name=name, seed=seed, dt=dt, duration=duration, split=split)
    raise ValueError(f"Unknown scenario kind {kind!r}; choose from {KINDS}")


KINDS = ("empty", "crossing", "head_on", "overtaking", "following", "group_toward", "narrow_passage",
         "stationary_group", "dense_flow", "recorded")
SYNTHETIC_KINDS = KINDS[:-1]


def scenario_set(split: str, kinds, per_kind: int, *, start: int = 0, dt: float = DEFAULT_DT,
                 duration: float = 30.0, pool=None) -> list[Scenario]:
    """``per_kind`` scenarios of each kind with seeds unique to ``split`` (``start`` shifts the window)."""
    offset = SPLIT_OFFSETS[split] + start
    return [make_scenario(kind, offset + i, dt=dt, duration=duration, pool=pool, split=split)
            for kind in kinds for i in range(per_kind)]


def collate(scenarios: list[Scenario], device="cpu") -> ScenarioBatch:
    """Pad and stack scenarios (same ``dt``) into a ``ScenarioBatch``."""
    dt = scenarios[0].dt
    if any(abs(s.dt - dt) > 1e-9 for s in scenarios):
        raise ValueError("All scenarios in a batch must share dt")
    B = len(scenarios)
    K = max(s.ped_ref.shape[1] if s.ped_ref.size else s.steps for s in scenarios)
    P = max(1, max(len(s.ped_radius) for s in scenarios))
    C = max(len(s.circles) for s in scenarios)
    S = max(len(s.segments) for s in scenarios)
    ped_ref = np.zeros((B, P, K, 2), np.float32)
    ped_valid = np.zeros((B, P, K), bool)
    ped_radius = np.full((B, P), PED_RADIUS, np.float32)
    ped_speed = np.ones((B, P), np.float32)
    ped_mask = np.zeros((B, P), bool)
    circles, circle_mask = np.zeros((B, C, 3), np.float32), np.zeros((B, C), bool)
    segments, segment_mask = np.zeros((B, S, 4), np.float32), np.zeros((B, S), bool)
    for b, s in enumerate(scenarios):
        p, k = s.ped_ref.shape[:2]
        if p:
            ped_ref[b, :p, :k] = s.ped_ref
            ped_ref[b, :p, k:] = s.ped_ref[:, -1:]
            ped_valid[b, :p, :k] = s.ped_valid
            ped_radius[b, :p], ped_speed[b, :p], ped_mask[b, :p] = s.ped_radius, s.ped_speed, True
        circles[b, :len(s.circles)], circle_mask[b, :len(s.circles)] = s.circles, True
        segments[b, :len(s.segments)], segment_mask[b, :len(s.segments)] = s.segments, True
    t = lambda a: torch.as_tensor(a, device=device)
    return ScenarioBatch(
        robot_start=t(np.stack([s.robot_start for s in scenarios]).astype(np.float32)),
        robot_goal=t(np.stack([s.robot_goal for s in scenarios]).astype(np.float32)),
        duration=t(np.array([s.duration for s in scenarios], np.float32)),
        ped_ref=t(ped_ref), ped_valid=t(ped_valid), ped_radius=t(ped_radius), ped_speed=t(ped_speed),
        ped_mask=t(ped_mask), circles=t(circles), circle_mask=t(circle_mask), segments=t(segments),
        segment_mask=t(segment_mask), dt=float(dt), names=[s.name for s in scenarios])


def bounds(scenario: Scenario, margin: float = 1.0) -> tuple[float, float, float, float]:
    """Plot extent covering the robot task, pedestrians and obstacles."""
    points = [scenario.robot_start[None, :2], scenario.robot_goal[None]]
    if scenario.ped_ref.size:
        points.append(scenario.ped_ref[scenario.ped_valid])
    if len(scenario.segments):
        points += [scenario.segments[:, :2], scenario.segments[:, 2:]]
    if len(scenario.circles):
        points.append(scenario.circles[:, :2])
    allp = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in points])
    lo, hi = allp.min(0) - margin, allp.max(0) + margin
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def arena_walls(scenario: Scenario, margin: float = 3.0) -> np.ndarray:
    """Rectangle around a scenario (used by recorded scenes without a map)."""
    x0, y0, x1, y1 = bounds(scenario, margin)
    return np.asarray(rectangle_segments(x0, y0, x1, y1), np.float32)
