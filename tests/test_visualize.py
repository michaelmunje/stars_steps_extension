from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from matplotlib.figure import Figure

from conftest import write_canonical_fixture
from data import Scene
from visualize import load_track_diagnostic_window, render_storyboard


def _synthetic_scene(dataset: str, history_steps: int, future_steps: int) -> Scene:
    total = history_steps + future_steps
    agents = 2
    xy = np.zeros((total, agents, 2), dtype=np.float32)
    for step in range(total):
        xy[step, 0] = [step * 0.1, 0.0]
        xy[step, 1] = [step * 0.1, 1.0]
    valid = np.ones((total, agents), dtype=bool)
    return Scene(
        xy=xy,
        primary_track_id="1",
        neighbor_track_ids=["2"],
        dataset=dataset,
        recording="rec",
        start_step=0,
        scene_tag=0,
        sub_tags=[],
        valid=valid,
        rotation=0.0,
        center=np.zeros(2, dtype=np.float32),
    )


def _rendered_legend_labels(scene: Scene, output: Path, task: dict) -> list[str]:
    """Render ``scene`` and capture the legend labels ``render_storyboard`` builds.

    Monkeypatches ``Figure.legend`` to intercept the handles passed to it,
    mirroring how the issue was independently verified by execution rather
    than by re-deriving the expected string from the source.
    """
    captured: dict[str, list[str]] = {}
    original_legend = Figure.legend

    def fake_legend(self, handles=None, **kwargs):
        captured["labels"] = [handle.get_label() for handle in handles]
        return original_legend(self, handles=handles, **kwargs)

    Figure.legend = fake_legend
    try:
        render_storyboard(scene, output, task)
    finally:
        Figure.legend = original_legend
    return captured["labels"]


def test_storyboard_legend_boundary_matches_default_sample_hz(tmp_path: Path) -> None:
    history_steps, future_steps = 8, 12
    scene = _synthetic_scene("synthetic", history_steps, future_steps)
    task = {"history_steps": history_steps, "future_steps": future_steps, "sample_hz": 2.5}

    labels = _rendered_legend_labels(scene, tmp_path / "storyboard.pdf", task)

    boundary = history_steps * (1.0 / task["sample_hz"])
    assert boundary == 3.2
    assert any(f"{boundary:g}" in label for label in labels if "Past observation" in label)
    assert any(f"{boundary:g}" in label for label in labels if "Future ground truth" in label)


def test_storyboard_legend_boundary_follows_per_dataset_sample_hz_override(tmp_path: Path) -> None:
    """The legend boundary must track ``sample_hz``, including a per-dataset override.

    ``render_storyboard`` already computes ``sample_dt`` (and thus the true
    observation boundary) from a per-dataset override under
    ``task["datasets"][scene.dataset]["sample_hz"]``. This asserts the
    rendered legend text reflects the *computed* boundary rather than any
    fixed string, so a reintroduced hardcoded literal fails this test.
    """
    history_steps, future_steps = 8, 12
    dataset = "synthetic_override"
    scene = _synthetic_scene(dataset, history_steps, future_steps)
    task = {
        "history_steps": history_steps,
        "future_steps": future_steps,
        "sample_hz": 2.5,
        "datasets": {dataset: {"sample_hz": 5.0}},
    }

    labels = _rendered_legend_labels(scene, tmp_path / "storyboard.pdf", task)

    boundary = history_steps * (1.0 / task["datasets"][dataset]["sample_hz"])
    assert boundary == 1.6

    past_labels = [label for label in labels if "Past observation" in label]
    future_labels = [label for label in labels if "Future ground truth" in label]
    assert past_labels and future_labels
    assert all(f"{boundary:g}" in label for label in past_labels)
    assert all(f"{boundary:g}" in label for label in future_labels)
    # The stale, hardcoded boundary must not leak into the rendered legend.
    assert not any("3.2" in label for label in past_labels + future_labels)


def test_track_diagnostic_uses_requested_time_as_observation_boundary(tmp_path: Path) -> None:
    converted = tmp_path / "converted"
    converted.mkdir()
    schema = pa.schema(
        [
            pa.field("track_id", pa.int32(), nullable=False),
            pa.field("step", pa.int32(), nullable=False),
            pa.field("x", pa.float32(), nullable=False),
            pa.field("y", pa.float32(), nullable=False),
        ],
        metadata={b"canonical_hz": b"2.5"},
    )
    rows = {
        "track_id": [1] * 20 + [2] * 5,
        "step": list(range(1343, 1363)) + list(range(1348, 1353)),
        "x": [float(i) for i in range(20)] + [float(i) for i in range(5)],
        "y": [0.0] * 25,
    }
    pq.write_table(pa.Table.from_pydict(rows, schema=schema), converted / "example.traj.parquet")

    steps, xy, anchor_step, hz = load_track_diagnostic_window(
        tmp_path, "example", [1, 2], 540.0, history_steps=8, future_steps=12
    )

    assert hz == 2.5
    assert anchor_step == 1350
    assert steps.tolist() == list(range(1343, 1363))
    assert steps[7] == anchor_step
    assert xy.shape == (20, 2, 2)
    assert np.isfinite(xy[:, 0]).all()
    assert np.isnan(xy[:5, 1]).all()
    assert np.isfinite(xy[5:10, 1]).all()


def _write_recording(
    dataset_dir: Path,
    recording: str,
    tracks: dict[int, list[tuple[int, float, float, float | None]]],
) -> Path:
    """Write ``recording`` as a canonical-schema parquet under ``dataset_dir/converted``.

    Canonical recordings are stored at ``canonical.CANONICAL_HZ`` (2.5 Hz), which
    is the rate the arithmetic in these tests assumes.
    """
    return write_canonical_fixture(
        dataset_dir / "converted" / f"{recording}.traj.parquet",
        tracks,
        dataset="fixture",
        scene_id="fixture",
        scene_sequence="fixture-000",
        recording=recording,
    )


def _straight_track(steps: range) -> list[tuple[int, float, float, float | None]]:
    return [(step, float(step), 0.0, None) for step in steps]


def test_track_diagnostic_rejects_duplicate_track_step_rows(tmp_path: Path) -> None:
    """A canonical recording holds at most one sample per ``(track_id, step)``.

    Without the guard the second row for the same cell would silently overwrite
    the first (last write wins), so the diagnostic figure would plot a position
    that depends on row order rather than failing.
    """
    _write_recording(
        tmp_path,
        "example",
        {1: _straight_track(range(0, 20)) + [(10, 99.0, 99.0, None)]},
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_track_diagnostic_window(
            tmp_path, "example", [1], 4.0, history_steps=8, future_steps=12
        )


def test_track_diagnostic_accepts_earliest_anchor_with_full_history(tmp_path: Path) -> None:
    """8 observations ending at the anchor need 7 earlier steps, so step 7 (2.8 s) is the earliest.

    At 2.5 Hz, step 7 is 7 / 2.5 = 2.8 s and the window starts exactly at step 0.
    """
    _write_recording(tmp_path, "example", {1: _straight_track(range(0, 20))})

    steps, xy, anchor_step, hz = load_track_diagnostic_window(
        tmp_path, "example", [1], 2.8, history_steps=8, future_steps=12
    )

    assert hz == 2.5
    assert anchor_step == 7
    assert steps[0] == 0
    assert steps[7] == anchor_step
    assert xy.shape == (20, 1, 2)
    assert np.isfinite(xy).all()


def test_track_diagnostic_rejects_anchor_before_full_history(tmp_path: Path) -> None:
    """2.0 s is step 5 at 2.5 Hz; 8 observations would start two steps before the recording."""
    _write_recording(tmp_path, "example", {1: _straight_track(range(0, 20))})

    with pytest.raises(ValueError, match="too early"):
        load_track_diagnostic_window(
            tmp_path, "example", [1], 2.0, history_steps=8, future_steps=12
        )


def test_track_diagnostic_rejects_track_absent_from_window(tmp_path: Path) -> None:
    """Every requested track must contribute samples to the returned window.

    The function promises a ``(len(steps), len(track_ids), 2)`` array; an
    all-NaN column would otherwise reach the caller's bounds computation.
    """
    _write_recording(
        tmp_path,
        "example",
        {1: _straight_track(range(0, 20)), 2: _straight_track(range(40, 50))},
    )

    with pytest.raises(ValueError, match=r"Tracks \[2\]"):
        load_track_diagnostic_window(
            tmp_path, "example", [1, 2], 4.0, history_steps=8, future_steps=12
        )


def test_track_diagnostic_rejects_missing_recording(tmp_path: Path) -> None:
    """An unknown recording fails with the function's own message, not an incidental reader error.

    Matching the guard's wording is deliberate: opening a missing parquet would
    raise ``FileNotFoundError`` from pyarrow anyway, so a looser match would not
    show that the argument is validated up front.
    """
    _write_recording(tmp_path, "example", {1: _straight_track(range(0, 20))})

    with pytest.raises(FileNotFoundError, match="Canonical recording not found"):
        load_track_diagnostic_window(
            tmp_path, "absent", [1], 4.0, history_steps=8, future_steps=12
        )


def test_track_diagnostic_rejects_empty_track_ids(tmp_path: Path) -> None:
    _write_recording(tmp_path, "example", {1: _straight_track(range(0, 20))})

    with pytest.raises(ValueError, match="At least one track ID"):
        load_track_diagnostic_window(
            tmp_path, "example", [], 4.0, history_steps=8, future_steps=12
        )


def test_track_diagnostic_rejects_repeated_track_ids(tmp_path: Path) -> None:
    """Repeated IDs would map two output columns onto one track, so they are rejected."""
    _write_recording(tmp_path, "example", {1: _straight_track(range(0, 20))})

    with pytest.raises(ValueError, match="unique"):
        load_track_diagnostic_window(
            tmp_path, "example", [1, 1], 4.0, history_steps=8, future_steps=12
        )
