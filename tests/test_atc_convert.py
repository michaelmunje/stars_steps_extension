"""Tests for the ATC converter (src/datasets_preprocessing/atc/convert.py).

Real ATC raw rows have the layout:
  timestamp, track_id, raw_x_mm, raw_y_mm, height_mm, body_velocity, heading_rad, angular_velocity
`_rows()` reads columns 0, 1, 2, 3, 6 (timestamp, track_id, raw_x, raw_y, heading).
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from datasets_preprocessing.atc import convert as MODULE


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("".join(rows))
    return path


def test_end_to_end_conversion_aligns_steps_and_matches_canonical_schema(tmp_path):
    # Two tracks interleaved at 2.5 Hz-aligned timestamps (0.4 s stride).
    # track 1 has rows at t=0.0, 0.4, 0.8 -> steps 0, 1, 2.
    # track 2 has rows at t=0.0, 0.4 -> steps 0, 1.
    csv_path = _write_csv(
        tmp_path / "atc-test.csv",
        [
            "0.0,1,1000,2000,0,0,0.1,0\n",
            "0.0,2,500,500,0,0,-0.1,0\n",
            "0.4,1,1100,2000,0,0,0.2,0\n",
            "0.4,2,600,500,0,0,-0.2,0\n",
            "0.8,1,1200,2000,0,0,0.3,0\n",
        ],
    )
    source = MODULE.RawSource(csv_path)
    output = tmp_path / "atc-test.traj.parquet"

    MODULE.convert_file(source, output, center=(0.0, 0.0))

    table = pq.read_table(output)
    assert table.schema.remove_metadata() == MODULE.OUTPUT_SCHEMA

    frame = table.to_pandas().sort_values(["track_id", "step"]).reset_index(drop=True)
    assert list(zip(frame.track_id, frame.step)) == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    # No duplicate (track_id, step) pairs in the canonical output.
    assert not frame.duplicated(["track_id", "step"]).any()

    track1 = frame[frame.track_id == 1].sort_values("step")
    assert track1.x.tolist() == pytest.approx([1.0, 1.1, 1.2], abs=1e-5)
    assert track1.y.tolist() == pytest.approx([2.0, 2.0, 2.0], abs=1e-5)
    assert track1.heading.tolist() == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)

    track2 = frame[frame.track_id == 2].sort_values("step")
    assert track2.x.tolist() == pytest.approx([0.5, 0.6], abs=1e-5)
    assert track2.y.tolist() == pytest.approx([0.5, 0.5], abs=1e-5)

    diagnostics_path = output.with_suffix(".diagnostics.json")
    assert diagnostics_path.exists()


def test_non_finite_raw_coordinate_is_rejected(tmp_path):
    csv_path = _write_csv(
        tmp_path / "atc-bad.csv",
        [
            "0.0,1,1000,2000,0,0,0.1,0\n",
            "0.4,1,nan,2000,0,0,0.2,0\n",
        ],
    )
    source = MODULE.RawSource(csv_path)

    with pytest.raises(ValueError, match="must be finite"):
        list(MODULE._rows(source))


def test_non_finite_heading_is_rejected(tmp_path):
    csv_path = _write_csv(
        tmp_path / "atc-bad-heading.csv",
        ["0.0,1,1000,2000,0,0,inf,0\n"],
    )
    source = MODULE.RawSource(csv_path)

    with pytest.raises(ValueError, match="must be finite"):
        list(MODULE._rows(source))


def test_per_track_nearest_alignment_error_dedup_keeps_the_closer_raw_row(tmp_path):
    # Two raw rows for the same track both round to canonical step 1
    # (ideal elapsed time 0.4 s at 2.5 Hz). The row at t=0.42 s is closer
    # to the ideal alignment (|0.42 - 0.4| = 0.02) than the row at
    # t=0.35 s (|0.35 - 0.4| = 0.05), so the converter must keep the
    # t=0.42 s sample (raw_x=3000) and discard the t=0.35 s one
    # (raw_x=2000), even though it arrived first.
    csv_path = _write_csv(
        tmp_path / "atc-dedup.csv",
        [
            "0.0,1,1000,2000,0,0,0.0,0\n",   # step 0 (anchor)
            "0.35,1,2000,2000,0,0,0.0,0\n",  # step 1, alignment error 0.05
            "0.42,1,3000,2000,0,0,0.0,0\n",  # step 1, alignment error 0.02 (closer, wins)
            "0.8,1,4000,2000,0,0,0.0,0\n",   # step 2 (forces the step-1 flush)
        ],
    )
    source = MODULE.RawSource(csv_path)
    output = tmp_path / "atc-dedup.traj.parquet"

    MODULE.convert_file(source, output, center=(0.0, 0.0))

    frame = pq.read_table(output).to_pandas().sort_values("step").reset_index(drop=True)
    assert not frame.duplicated(["track_id", "step"]).any()
    step1 = frame[frame.step == 1]
    assert len(step1) == 1
    # 3000 mm * 0.001 = 3.0 m; the discarded row would have produced 2.0 m.
    assert step1.x.iloc[0] == pytest.approx(3.0, abs=1e-5)


def _flush_then_unordered_csv_rows() -> list[str]:
    """Rows for a single track that force an intermediate flush() (3 completed
    steps, exceeding a monkeypatched FLUSH_ROW_THRESHOLD of 3) followed by a
    5th row whose timestamp jumps backward, violating the strict step
    ordering invariant and raising ValueError from convert_file's main loop.
    """
    return [
        "0.0,1,1000,2000,0,0,0.0,0\n",  # step 0
        "0.4,1,1000,2000,0,0,0.0,0\n",  # step 1 -> step-0 row queued
        "0.8,1,1000,2000,0,0,0.0,0\n",  # step 2 -> step-1 row queued
        "1.2,1,1000,2000,0,0,0.0,0\n",  # step 3 -> step-2 row queued; pending=3 >= threshold -> flush()
        "0.4,1,1000,2000,0,0,0.0,0\n",  # step 1 again: 1 < current_step(3) -> ValueError
    ]


def test_mid_conversion_failure_after_a_flush_leaves_no_output_file(tmp_path, monkeypatch):
    """Regression test for D-004: a flush() to the temp Parquet file followed
    by a later ValueError (ordering violation) must not leave a truncated,
    schema-valid file at the canonical `output` path, and must not leave a
    stray `.tmp.parquet` behind either.
    """
    monkeypatch.setattr(MODULE, "FLUSH_ROW_THRESHOLD", 3)
    csv_path = _write_csv(tmp_path / "atc-truncated.csv", _flush_then_unordered_csv_rows())
    source = MODULE.RawSource(csv_path)
    output = tmp_path / "atc-truncated.traj.parquet"

    with pytest.raises(ValueError, match="timestamps are not ordered"):
        MODULE.convert_file(source, output, center=(0.0, 0.0))

    assert not output.exists(), "a failed conversion must not leave a file at the canonical output path"
    assert not output.with_suffix(".tmp.parquet").exists(), "the temp Parquet file must be cleaned up on failure"
    assert not output.with_suffix(".diagnostics.json").exists()


def test_mid_conversion_failure_preserves_a_prior_valid_output(tmp_path, monkeypatch):
    """More severe D-004 variant: retrying a failed conversion over an existing
    valid output must not destroy the prior good file.
    """
    output = tmp_path / "atc-retry.traj.parquet"

    good_csv = _write_csv(
        tmp_path / "atc-good.csv",
        [
            "0.0,1,1000,2000,0,0,0.1,0\n",
            "0.4,1,1100,2000,0,0,0.2,0\n",
        ],
    )
    MODULE.convert_file(MODULE.RawSource(good_csv), output, center=(0.0, 0.0))
    assert output.exists()
    original_bytes = output.read_bytes()

    monkeypatch.setattr(MODULE, "FLUSH_ROW_THRESHOLD", 3)
    bad_csv = _write_csv(tmp_path / "atc-truncated-retry.csv", _flush_then_unordered_csv_rows())
    with pytest.raises(ValueError, match="timestamps are not ordered"):
        MODULE.convert_file(MODULE.RawSource(bad_csv), output, center=(0.0, 0.0))

    assert output.exists(), "the previously valid output must survive a failed retry"
    assert output.read_bytes() == original_bytes, "the previously valid output must be byte-for-byte unchanged"
    assert not output.with_suffix(".tmp.parquet").exists()


def test_parallel_convert_dataset_surfaces_the_same_exception_type_as_sequential(tmp_path):
    """Regression test for D-007: convert_dataset's workers>1 path (executor.submit
    / as_completed) must not wrap a worker's original exception in a RuntimeError;
    it must surface the same exception type (ValueError, from a bad row) that the
    workers=1 sequential path raises.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(
        raw_dir / "atc-bad.csv",
        [
            "0.0,1,1000,2000,0,0,0.1,0\n",
            "0.4,1,nan,2000,0,0,0.2,0\n",
        ],
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(ValueError, match="must be finite") as sequential_excinfo:
        MODULE.convert_dataset(dataset_dir, raw_dir=raw_dir, workers=1)

    with pytest.raises(ValueError, match="must be finite") as parallel_excinfo:
        MODULE.convert_dataset(dataset_dir, raw_dir=raw_dir, workers=2)

    assert type(sequential_excinfo.value) is type(parallel_excinfo.value) is ValueError
    assert not isinstance(parallel_excinfo.value, RuntimeError)


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
