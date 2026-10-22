"""Recorded pedestrian trajectories: a simple CSV schema, converters, and recorded-traffic scenarios.

CSV schema (one row per observation, any order; header required)::

    recording_id,pedestrian_id,t,x,y[,vx,vy,theta]

``t`` in seconds, positions in metres in one world frame per recording; ``vx, vy`` (m/s) and
``theta`` (rad) are optional and, when absent, derived from positions. Any dataset with ids,
times and planar positions converts to it; ``convert_steps`` does so for STEPS's canonical
Parquet, and ``from_steps`` reads STEPS directly.

Tracks are resampled onto the simulation grid by linear interpolation; gaps longer than
``max_gap`` seconds split a track (the pedestrian leaves and re-enters). ``RecordingPool``
assigns whole recordings to train / val / test (from STEPS's grouped split manifest when
available, else a seeded hash), so overlapping windows of one recording never cross splits.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seanjax.core.geometry import rectangle_segments
from seanjax.core.scenarios import PED_RADIUS, Scenario

STEPS_HZ = 2.5


@dataclass
class Track:
    t: np.ndarray    # [N] seconds, increasing
    xy: np.ndarray   # [N, 2]


@dataclass
class Recording:
    id: str
    tracks: dict[str, Track]

    @property
    def span(self) -> tuple[float, float]:
        return (min(float(tr.t[0]) for tr in self.tracks.values()), max(float(tr.t[-1]) for tr in self.tracks.values()))


def load_csv(path) -> list[Recording]:
    rows: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"recording_id", "pedestrian_id", "t", "x", "y"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            rows.setdefault((row["recording_id"], row["pedestrian_id"]), []).append(
                (float(row["t"]), float(row["x"]), float(row["y"])))
    recordings: dict[str, dict[str, Track]] = {}
    for (rec, ped), samples in rows.items():
        samples.sort()
        data = np.asarray(samples, dtype=np.float64)
        recordings.setdefault(rec, {})[ped] = Track(data[:, 0], data[:, 1:3])
    return [Recording(rec, tracks) for rec, tracks in sorted(recordings.items())]


def write_csv(recordings: list[Recording], path) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["recording_id", "pedestrian_id", "t", "x", "y"])
        for rec in recordings:
            for ped, track in rec.tracks.items():
                for t, (x, y) in zip(track.t, track.xy):
                    writer.writerow([rec.id, ped, f"{t:.3f}", f"{x:.4f}", f"{y:.4f}"])


def _steps_recording(path: Path) -> Recording:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["track_id", "step", "x", "y"]).to_pandas()
    tracks = {}
    for track_id, group in table.groupby("track_id"):
        group = group.sort_values("step")
        tracks[str(track_id)] = Track(group["step"].to_numpy(np.float64) / STEPS_HZ,
                                      group[["x", "y"]].to_numpy(np.float64))
    return Recording(path.name.removesuffix(".traj.parquet"), tracks)


def steps_split(dataset_dir) -> dict[str, str]:
    """Recording -> split from STEPS's grouped split manifest."""
    manifest = Path(dataset_dir) / "split_manifests" / "grouped-seed42.json"
    records = json.loads(manifest.read_text())["records"]
    return {r["recording"]: r["split"] for r in records}


def convert_steps(dataset_dir, out_csv, recordings: list[str] | None = None) -> int:
    """Write STEPS canonical recordings (all, or the named ones) to the CSV schema."""
    files = sorted((Path(dataset_dir) / "converted").glob("*.traj.parquet"))
    chosen = [f for f in files if recordings is None or f.name.removesuffix(".traj.parquet") in recordings]
    write_csv([_steps_recording(f) for f in chosen], out_csv)
    return len(chosen)


def resample(track: Track, t0: float, dt: float, steps: int, max_gap: float = 1.0):
    """Positions ``[steps, 2]`` on ``t0 + k dt`` and presence ``[steps]`` (inside the track, no long gap)."""
    t = t0 + np.arange(steps) * dt
    inside = (t >= track.t[0]) & (t <= track.t[-1])
    xy = np.stack([np.interp(t, track.t, track.xy[:, i]) for i in range(2)], -1)
    gaps = np.diff(track.t) > max_gap
    if gaps.any():
        index = np.searchsorted(track.t, t, side="right") - 1
        in_gap = np.zeros_like(inside)
        valid_index = (index >= 0) & (index < len(gaps))
        in_gap[valid_index] = gaps[index[valid_index]]
        inside &= ~in_gap
    return xy.astype(np.float32), inside


def _hash_split(name: str, seed: int, ratios=(0.8, 0.1, 0.1)) -> str:
    u = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < ratios[0] else ("val" if u < ratios[0] + ratios[1] else "test")


class RecordingPool:
    """Recordings grouped by split; draws recorded-traffic scenarios for ``scenarios.make_scenario``."""

    def __init__(self, recordings: list[Recording], splits: dict[str, str] | None = None, *, seed: int = 0,
                 max_pedestrians: int = 40, min_pedestrians: int = 3, wall_margin: float = 2.0,
                 window_candidates: int = 8) -> None:
        self.by_split: dict[str, list[Recording]] = {"train": [], "val": [], "test": []}
        for rec in recordings:
            split = (splits or {}).get(rec.id) or _hash_split(rec.id, seed)
            self.by_split.setdefault(split, []).append(rec)
        self.max_pedestrians, self.min_pedestrians, self.wall_margin = max_pedestrians, min_pedestrians, wall_margin
        self.window_candidates = window_candidates

    @classmethod
    def from_steps(cls, dataset_dir, *, limit_per_split: int | None = None, **kwargs) -> "RecordingPool":
        dataset_dir = Path(dataset_dir)
        splits = steps_split(dataset_dir)
        excluded = set(json.loads((dataset_dir / "source.json").read_text()).get("excluded_recordings", []))
        chosen, counts = [], {}
        for name, split in sorted(splits.items()):
            if name in excluded:
                continue
            if limit_per_split and counts.get(split, 0) >= limit_per_split:
                continue
            counts[split] = counts.get(split, 0) + 1
            chosen.append(_steps_recording(dataset_dir / "converted" / f"{name}.traj.parquet"))
        return cls(chosen, splits, **kwargs)

    @classmethod
    def from_csv(cls, path, **kwargs) -> "RecordingPool":
        return cls(load_csv(path), None, **kwargs)

    def scenario(self, rng: np.random.Generator, *, name: str, seed: int, dt: float, duration: float,
                 split: str, attempts: int = 50) -> Scenario:
        recordings = self.by_split.get(split) or []
        if not recordings:
            raise ValueError(f"No {split} recordings in the pool")
        steps = int(round(duration / dt)) + 1
        for _ in range(attempts):
            # Busiest of a few random windows: recorded scenes should actually involve pedestrians.
            best = None
            for _ in range(self.window_candidates):
                rec = recordings[int(rng.integers(len(recordings)))]
                lo, hi = rec.span
                if hi - lo < duration:
                    continue
                t0 = float(rng.uniform(lo, hi - duration))
                refs, valids = [], []
                for track in rec.tracks.values():
                    if track.t[-1] < t0 or track.t[0] > t0 + duration:
                        continue
                    xy, inside = resample(track, t0, dt, steps)
                    if inside.sum() >= 5:
                        refs.append(xy), valids.append(inside)
                score = sum(int(v.sum()) for v in valids)
                if len(refs) >= self.min_pedestrians and (best is None or score > best[0]):
                    best = (score, rec, t0, refs, valids)
            if best is None:
                continue
            _, rec, t0, refs, valids = best
            ref, valid = np.stack(refs), np.stack(valids)
            occupied = ref[valid]
            low, high = np.percentile(occupied, 5, axis=0), np.percentile(occupied, 95, axis=0)
            # Route through where the crowd is mid-window, across the occupied area.
            middle = valid[:, steps // 3: 2 * steps // 3]
            center = ref[:, steps // 3: 2 * steps // 3][middle].mean(0) if middle.any() else (low + high) / 2
            half = np.maximum((high - low) / 2, 2.0)
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            reach = float(np.linalg.norm(half * direction)) * 0.9
            start, goal = center - reach * direction, center + reach * direction
            if (np.linalg.norm(ref[:, 0] - start, axis=-1)[valid[:, 0]] < 1.0).any():
                continue
            # The goal must be reachable: not next to someone (standing pedestrians in a learned
            # crowd do not yield, and a goal inside a group cannot be reached safely).
            if (np.linalg.norm(occupied - goal, axis=-1) < 1.0).any():
                continue
            # Keep the pedestrians nearest to the robot's straight path.
            seg = goal - start
            cross = lambda d: np.abs(seg[0] * d[:, 1] - seg[1] * d[:, 0]) / np.linalg.norm(seg)
            path_distance = np.array([np.min(cross(r[v] - start)) for r, v in zip(ref, valid)])
            keep = np.argsort(path_distance)[: self.max_pedestrians]
            ref, valid = ref[keep], valid[keep]
            speeds = []
            for r, v in zip(ref, valid):
                step = np.linalg.norm(np.diff(r[v], axis=0), axis=-1) / dt
                speeds.append(float(np.clip(np.median(step) if len(step) else 1.0, 0.5, 1.6)))
            x0, y0 = np.minimum(low, np.minimum(start, goal)) - self.wall_margin
            x1, y1 = np.maximum(high, np.maximum(start, goal)) + self.wall_margin
            heading = float(np.arctan2(direction[1], direction[0]))
            return Scenario(name, "recorded", seed, dt, duration,
                            np.array([*start, heading], np.float32), goal.astype(np.float32), ref, valid,
                            np.full(len(ref), PED_RADIUS, np.float32), np.asarray(speeds, np.float32),
                            np.zeros((0, 3), np.float32), np.asarray(rectangle_segments(x0, y0, x1, y1), np.float32),
                            # The box bounds the robot's arena; recorded people cross it.
                            {"recording": rec.id, "t0": t0, "pedestrian_walls": False})
        raise RuntimeError(f"Could not draw a recorded scenario from {split} recordings after {attempts} attempts")
