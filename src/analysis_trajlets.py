from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pyarrow as pa


OPENTRAJ_TRAJLET_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("scene_id", pa.string()),
        ("scene_sequence", pa.string()),
        ("recording", pa.string()),
        ("split", pa.string()),
        ("agent_id", pa.int64()),
        ("source_start_step", pa.int32()),
        ("source_returned_end_step", pa.int32()),
        ("source_static_endpoint_step", pa.int32()),
        ("start_time", pa.float64()),
        ("returned_end_time", pa.float64()),
        ("static_endpoint_time", pa.float64()),
        ("n_returned_samples", pa.int16()),
        ("static_displacement_m", pa.float32()),
        ("opentraj_static", pa.bool_()),
        ("retained_after_static_filter", pa.bool_()),
        ("trajnet_category", pa.int8()),
    ],
    metadata={
        b"representation":b"opentraj_analysis_trajlet_index",
        b"returned_sample_convention":b"returns s..s+11; s+12 is required only for the static endpoint test at 2.5 Hz",
        b"retention_criterion":b"norm(x[s+12]-x[s]) > 1.0 metre",
        b"opentraj_reference":b"crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a",
    },
)


@dataclass(frozen=True)
class OpenTrajTrajletConvention:
    length_s: float = 4.8
    overlap_s: float = 2.0
    static_threshold_m: float = 1.0
    epsilon_s: float = 1e-2

    def sample_counts(self, dt: float) -> tuple[int, int]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        returned = int(math.ceil((self.length_s - self.epsilon_s) / dt))
        step = int(math.ceil((self.length_s - self.overlap_s - self.epsilon_s) / dt))
        return returned, step


@dataclass(frozen=True)
class AnalysisTrajlet:
    dataset: str
    scene_id: str
    scene_sequence: str
    recording: str
    split: str | None
    agent_id: int
    source_start_step: int
    source_returned_end_step: int
    source_static_endpoint_step: int
    start_time: float
    returned_end_time: float
    static_endpoint_time: float
    n_returned_samples: int
    static_displacement_m: float
    opentraj_static: bool
    retained_after_static_filter: bool
    trajnet_category: int | None = None
    source_indices: tuple[int, ...] = ()

    def as_index_row(self) -> dict:
        row = dict(self.__dict__)
        row.pop("source_indices")
        return row


def continuous_segments(steps: np.ndarray) -> Iterator[np.ndarray]:
    """Yield positional indices for each uninterrupted canonical-step run."""
    steps = np.asarray(steps)
    if steps.ndim != 1:
        raise ValueError("steps must be one-dimensional")
    if len(steps) and np.any(np.diff(steps) <= 0):
        raise ValueError("steps must be strictly increasing within one agent/sequence")
    for segment in np.split(np.arange(len(steps)), np.flatnonzero(np.diff(steps) != 1) + 1):
        if len(segment):
            yield segment


def extract_opentraj_trajlets(
    *,
    dataset: str,
    scene_id: str,
    scene_sequence: str,
    recording: str | None = None,
    agent_id: int,
    steps: np.ndarray,
    xy: np.ndarray,
    sample_hz: float = 2.5,
    convention: OpenTrajTrajletConvention | None = None,
) -> list[AnalysisTrajlet]:
    """Replicate OpenTraj's 12-returned + 13th-static-endpoint behavior, with one
    deliberate deviation: candidate start indices are restricted to a single
    uninterrupted canonical-step run rather than the whole agent track (see the
    comment above the ``range(...)`` below for the rationale)."""
    convention = convention or OpenTrajTrajletConvention()
    steps = np.asarray(steps, dtype=np.int64)
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (len(steps), 2):
        raise ValueError("xy must have shape (len(steps), 2)")
    dt = 1.0 / float(sample_hz)
    returned_count, start_stride = convention.sample_counts(dt)
    output: list[AnalysisTrajlet] = []
    for segment in continuous_segments(steps):
        segment_steps = steps[segment]
        segment_xy = xy[segment]
        # Upstream (crowdbotp/OpenTraj@97ec1d5e.../core/trajlet.py:25-36) uses
        # range(0, n_frames - f_per_traj, f_step) where n_frames = len(tr) and tr is
        # the *whole* (scene_id, agent_id) group, sliced positionally: it never splits
        # on temporal gaps, so upstream will emit a trajlet that spans a gap. Here
        # `segment` is one uninterrupted canonical-step run from continuous_segments()
        # rather than the whole group, so `len(segment)` deliberately restricts
        # candidate starts to gap-free runs (research-plan.md:26-29: identities are
        # never re-identified across gaps). The range formula itself is unchanged from
        # upstream; only its operand (whole group vs. one run) differs. The two rules
        # coincide whenever a track has no gaps.
        # It requires source[start + f_per_traj] but returns the preceding slice.
        for start in range(0, len(segment) - returned_count, start_stride):
            endpoint = start + returned_count
            returned_end = endpoint - 1
            displacement = float(np.linalg.norm(segment_xy[endpoint] - segment_xy[start]))
            retained = displacement > convention.static_threshold_m
            source_start = int(segment_steps[start])
            source_end = int(segment_steps[returned_end])
            source_endpoint = int(segment_steps[endpoint])
            output.append(
                AnalysisTrajlet(
                    dataset=dataset,
                    scene_id=scene_id,
                    scene_sequence=scene_sequence,
                    recording=recording or scene_sequence,
                    split=None,
                    agent_id=int(agent_id),
                    source_start_step=source_start,
                    source_returned_end_step=source_end,
                    source_static_endpoint_step=source_endpoint,
                    start_time=source_start / sample_hz,
                    returned_end_time=source_end / sample_hz,
                    static_endpoint_time=source_endpoint / sample_hz,
                    n_returned_samples=returned_count,
                    static_displacement_m=displacement,
                    opentraj_static=not retained,
                    retained_after_static_filter=retained,
                    trajnet_category=None,
                    source_indices=tuple(segment[start:endpoint].tolist()),
                )
            )
    return output


def index_rows_to_table(index_rows: list[dict]) -> pa.Table:
    """Build the trajlet index table from ``AnalysisTrajlet.as_index_row()`` dicts.

    Callers pass the already-materialized rows so the same list can be reused
    for the index table and the manifest sample counts.
    """
    return pa.Table.from_pylist(index_rows, schema=OPENTRAJ_TRAJLET_SCHEMA)
