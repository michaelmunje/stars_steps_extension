"""Shared test environment and fixture helpers for the repository's ``src`` layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
existing = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    f"{SOURCE_ROOT}{os.pathsep}{existing}" if existing else str(SOURCE_ROOT)
)

# ``pythonpath = src`` in pytest.ini already puts ``src`` on sys.path for this
# process, so the canonical-schema helpers below can be imported directly.
from canonical import canonical_metadata, schema_with_metadata  # noqa: E402


def write_canonical_fixture(
    path: Path,
    tracks: Mapping[int, Sequence[tuple[int, float, float, float | None]]],
    *,
    dataset: str,
    scene_id: str,
    scene_sequence: str,
    recording: str,
) -> Path:
    """Write a small canonical-schema (track_id, step, x, y, heading) parquet fixture.

    ``tracks`` maps a ``track_id`` to a sequence of ``(step, x, y, heading)``
    rows for that track; rows across all tracks are sorted by ``(step,
    track_id)`` before writing. ``heading`` is used exactly as supplied by the
    caller: pass ``None`` for an Arrow validity-null (``is_valid=False``) or a
    float such as ``float("nan")`` for a valid-but-NaN value
    (``is_valid=True``) -- these are NOT equivalent at the Arrow level, so the
    helper never substitutes one for the other.
    """
    rows = [
        (track_id, step, x, y, heading)
        for track_id, track_rows in tracks.items()
        for step, x, y, heading in track_rows
    ]
    rows.sort(key=lambda row: (row[1], row[0]))
    schema = schema_with_metadata(canonical_metadata(
        dataset=dataset, scene_id=scene_id, scene_sequence=scene_sequence, recording=recording,
    ))
    table = pa.Table.from_arrays([
        pa.array([row[0] for row in rows], pa.int32()),
        pa.array([row[1] for row in rows], pa.int32()),
        pa.array([row[2] for row in rows], pa.float32()),
        pa.array([row[3] for row in rows], pa.float32()),
        pa.array([row[4] for row in rows], pa.float32()),
    ], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_run_result(directory: Path, *, model: str = "moflow_teacher", **evaluation_fields) -> Path:
    """Write a baseline run directory's ``config.json`` + ``evaluation.json`` fixture pair.

    ``evaluation_fields`` becomes the entire contents of ``evaluation.json``;
    each caller supplies its own full, scenario-specific metric dict.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model": model}))
    (directory / "evaluation.json").write_text(json.dumps(evaluation_fields))
    return directory
