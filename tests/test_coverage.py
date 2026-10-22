"""Regression test for R-002: ``coverage.coverage_curves``'s "random" curve must be
plotted at the hours it actually retains, not at ``hours * fraction`` (which, for short
recordings where ``bins_to_keep = round(n_bins * fraction)`` rounds several requested
fractions onto the same integer bin count, overstates how much data was drawn -- up to
3.3x at the 10% point for an ETH/UCY-scale 0.2149h recording). The "contiguous" branch
already used its exact ``block_hours = hours * fraction`` and must not move.

This also covers the round-half-up fix for ``bins_to_keep`` (previously Python/NumPy
round-half-to-even via ``int(round(...))`` gave ``n_bins=5, fraction=0.5`` -> 2/5 instead
of the expected 3/5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coverage import coverage_curves

STORAGE_HZ = 1.0  # steps are plain seconds, chosen for readable hour arithmetic.


def _write_recording(path: Path, track_steps: list[list[int]]) -> None:
    """Write a synthetic ``*.traj.parquet`` recording. Each inner list of ``track_steps``
    is one track's step values; x advances linearly with step, y is constant, so no
    track is degenerate (zero displacement) but the exact motion is otherwise
    irrelevant to this test."""
    track_id, step, x, y = [], [], [], []
    for tid, steps in enumerate(track_steps):
        for s in steps:
            track_id.append(tid)
            step.append(s)
            x.append(0.01 * s)
            y.append(0.0)
    table = pa.table({"track_id": track_id, "step": step, "x": x, "y": y})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _make_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "synth"
    converted = dataset_dir / "converted"

    # Recording A: span 0..900s -> hours = 0.25, n_bins = ceil(0.25*10) = 3.
    # Three tracks placed early/middle/late so their median times land in bins 0, 1, 2.
    _write_recording(
        converted / "recA.traj.parquet",
        [[0, 10, 20, 30], [440, 450, 460, 470], [870, 880, 890, 900]],
    )

    # Recording B: span 0..1800s -> hours = 0.5, n_bins = ceil(0.5*10) = 5.
    # At fraction=0.5, n_bins*fraction=2.5 is the round-half-up/round-half-to-even tie.
    _write_recording(converted / "recB.traj.parquet", [[0, 600, 1200, 1800]])

    return dataset_dir


def test_random_curve_hours_match_realized_retention(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    fractions = np.linspace(0.1, 1, 10)
    result = coverage_curves(
        [dataset_dir], STORAGE_HZ, seeds=range(2), fractions=fractions, k=1
    )

    curves_a = result["recordings"]["synth/recA.traj"]
    hours_a = 0.25
    n_bins_a = 3  # ceil(0.25 * 10)

    plotted_random_hours = [v["hours"] for v in curves_a["random"]]
    plotted_contiguous_hours = [v["hours"] for v in curves_a["contiguous"]]

    expected_random_hours = []
    for fraction in fractions:
        bins_to_keep = max(1, int(np.floor(n_bins_a * fraction + 0.5)))
        expected_random_hours.append(hours_a * bins_to_keep / n_bins_a)
    expected_contiguous_hours = [hours_a * fraction for fraction in fractions]

    assert plotted_random_hours == pytest.approx(expected_random_hours)
    # Acceptance criterion 3: the contiguous branch's exact, unrounded definition is
    # untouched by this fix.
    assert plotted_contiguous_hours == pytest.approx(expected_contiguous_hours)

    # This is the concrete regression this issue reports: at the labelled 10% point the
    # random series must be plotted at the hours it truly retains (1/3 of the recording,
    # because n_bins=3 cannot resolve a 10% step), not at 0.10 * hours.
    assert plotted_random_hours[0] == pytest.approx(hours_a / n_bins_a)
    assert plotted_random_hours[0] != pytest.approx(hours_a * fractions[0])


def test_bins_to_keep_uses_round_half_up_not_banker_rounding(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    fractions = np.linspace(0.1, 1, 10)
    result = coverage_curves(
        [dataset_dir], STORAGE_HZ, seeds=range(2), fractions=fractions, k=1
    )

    curves_b = result["recordings"]["synth/recB.traj"]
    hours_b = 0.5
    n_bins_b = 5  # ceil(0.5 * 10)

    fraction_half_index = int(np.flatnonzero(np.isclose(fractions, 0.5))[0])
    retained_hours_at_half = curves_b["random"][fraction_half_index]["hours"]

    # n_bins_b * 0.5 == 2.5 is an exact tie. Python/NumPy round-half-to-even rounds this
    # down to bins_to_keep=2 (0.2h retained); round-half-up must instead keep 3 bins
    # (0.3h retained), matching the plain-English "round" a reader expects.
    assert retained_hours_at_half == pytest.approx(hours_b * 3 / n_bins_b)
    assert retained_hours_at_half != pytest.approx(hours_b * 2 / n_bins_b)
