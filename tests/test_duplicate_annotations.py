"""Duplicate-annotation detection and its effect on the local-density indicator.

Grand Central's and MADRAS's published local density was dominated by upstream
annotation duplicates rather than by crowding. These tests pin the detector's
behaviour, including a real-data regression on the documented Grand Central
pairs, so the artifact cannot silently return.
"""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from duplicate_annotations import (
    DUPLICATE_MIN_FRAMES,
    DUPLICATE_SEPARATION_M,
    DuplicatePair,
    duplicate_pairs,
    duplicate_summary,
    duplicate_tracks,
)
from opentraj_indicators import frame_local_density
from opentraj_report import _filtered_density


FIXTURE = Path(__file__).parent / "fixtures" / "grand_central_duplicate_tracks.csv"


def _walk(track_id: int, count: int, offset: tuple[float, float]) -> list[tuple[int, int, float, float]]:
    """A track walking along +x at 0.4 m per canonical frame from `offset`."""
    return [(track_id, step, offset[0] + 0.4 * step, offset[1]) for step in range(count)]


def _columns(rows: list[tuple[int, int, float, float]]):
    values = np.asarray(rows, dtype=float)
    return values[:, 0].astype(int), values[:, 1].astype(int), values[:, 2], values[:, 3]


class DuplicateDetectionTests(unittest.TestCase):
    def test_detects_persistently_coincident_identity(self):
        rows = _walk(1, 8, (0.0, 0.0)) + _walk(2, 8, (0.02, 0.0))
        pairs = duplicate_pairs(*_columns(rows))
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0].track_a, pairs[0].track_b), (1, 2))
        self.assertEqual(pairs[0].co_occurring_frames, 8)
        self.assertEqual(pairs[0].frames_within_threshold, 8)
        self.assertAlmostEqual(pairs[0].median_separation_m, 0.02, places=6)
        # The lower identity survives, so the pedestrian is still counted once.
        self.assertEqual(sorted(duplicate_tracks(pairs)), [2])

    def test_ignores_genuine_close_companions(self):
        # 0.4 m apart for the whole overlap: tight, but two real people. The
        # threshold exists precisely so this is never removed.
        rows = _walk(1, 8, (0.0, 0.0)) + _walk(2, 8, (0.0, 0.4))
        self.assertEqual(duplicate_pairs(*_columns(rows)), ())

    def test_min_frames_is_the_reporting_boundary(self):
        # One identity walks away after `close_steps` coincident frames: below
        # DUPLICATE_MIN_FRAMES that is a crossing or a tracker glitch, at it the
        # pair is a duplicated identity.
        for close_steps, expected_pairs in (
            (DUPLICATE_MIN_FRAMES - 1, 0),
            (DUPLICATE_MIN_FRAMES, 1),
        ):
            with self.subTest(close_steps=close_steps):
                rows = _walk(1, 8, (0.0, 0.0))
                rows += [(2, step, 0.4 * step + 0.01, 0.0) for step in range(close_steps)]
                rows += [(2, step, 0.4 * step + 3.0, 0.0) for step in range(close_steps, 8)]
                pairs = duplicate_pairs(*_columns(rows))
                self.assertEqual(len(pairs), expected_pairs)
                if expected_pairs:
                    self.assertEqual(pairs[0].frames_within_threshold, DUPLICATE_MIN_FRAMES)

    def test_triplicated_identity_keeps_one_survivor(self):
        # One pedestrian annotated three times is one connected component, so
        # two identities are dropped -- not one per pair.
        rows = _walk(7, 8, (0.0, 0.0)) + _walk(3, 8, (0.01, 0.0)) + _walk(9, 8, (0.02, 0.0))
        pairs = duplicate_pairs(*_columns(rows))
        self.assertEqual(len(pairs), 3)
        self.assertEqual(sorted(duplicate_tracks(pairs)), [7, 9])

    def test_non_finite_positions_are_skipped(self):
        rows = _walk(1, 8, (0.0, 0.0)) + _walk(2, 8, (0.02, 0.0))
        track_id, step, x, y = _columns(rows)
        x[0] = np.nan
        pairs = duplicate_pairs(track_id, step, x, y)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].frames_within_threshold, 7)
        # The pair is a constant 0.02 m apart at every step other than the
        # poisoned step 0, so the reported min/median are that constant: one
        # non-finite position must not decide them.
        self.assertFalse(np.isnan(pairs[0].min_separation_m))
        self.assertFalse(np.isnan(pairs[0].median_separation_m))
        self.assertAlmostEqual(pairs[0].min_separation_m, 0.02, places=6)
        self.assertAlmostEqual(pairs[0].median_separation_m, 0.02, places=6)

    def test_duplicate_summary_min_is_order_independent_with_a_nan_pair(self):
        # A pair with no finite co-occurring separation reports NaN for its own
        # min/median. The published `duplicate_min_separation_m` must ignore it
        # rather than depend on where that pair sits in the tuple.
        finite_pair = DuplicatePair(
            track_a=1,
            track_b=2,
            co_occurring_frames=10,
            frames_within_threshold=10,
            min_separation_m=0.02,
            median_separation_m=0.02,
        )
        nan_pair = DuplicatePair(
            track_a=3,
            track_b=4,
            co_occurring_frames=8,
            frames_within_threshold=8,
            min_separation_m=float("nan"),
            median_separation_m=float("nan"),
        )
        excluded = frozenset({2, 4})
        nan_first = duplicate_summary((nan_pair, finite_pair), excluded)
        nan_second = duplicate_summary((finite_pair, nan_pair), excluded)
        self.assertEqual(nan_first["duplicate_min_separation_m"], 0.02)
        self.assertEqual(nan_second["duplicate_min_separation_m"], 0.02)

    def test_summary_records_the_applied_criterion(self):
        rows = _walk(1, 8, (0.0, 0.0)) + _walk(2, 8, (0.02, 0.0))
        pairs = duplicate_pairs(*_columns(rows))
        summary = duplicate_summary(pairs, duplicate_tracks(pairs))
        self.assertEqual(summary["duplicate_separation_m"], DUPLICATE_SEPARATION_M)
        self.assertEqual(summary["duplicate_min_frames"], DUPLICATE_MIN_FRAMES)
        self.assertEqual(summary["duplicate_pairs"], 1)
        self.assertEqual(summary["duplicate_tracks_excluded"], 1)


class GrandCentralRegressionTests(unittest.TestCase):
    """Real canonical rows for the two documented Grand Central duplicate pairs."""

    @classmethod
    def setUpClass(cls):
        with FIXTURE.open() as stream:
            rows = [
                (int(row["track_id"]), int(row["step"]), float(row["x"]), float(row["y"]))
                for row in csv.DictReader(stream)
            ]
        cls.columns = _columns(rows)

    def test_documented_duplicate_pairs_are_detected(self):
        pairs = duplicate_pairs(*self.columns)
        self.assertEqual(
            [(pair.track_a, pair.track_b) for pair in pairs],
            [(14739, 14748), (14740, 14747)],
        )
        # Coordinator-measured in C-002: 136 co-occurring frames for 14740/14747
        # with a median separation of 0.063 m -- two people cannot do that.
        by_pair = {(pair.track_a, pair.track_b): pair for pair in pairs}
        documented = by_pair[(14740, 14747)]
        self.assertEqual(documented.co_occurring_frames, 136)
        self.assertAlmostEqual(documented.median_separation_m, 0.0627, places=3)
        self.assertEqual(documented.min_separation_m, 0.0)
        self.assertEqual(sorted(duplicate_tracks(pairs)), [14747, 14748])

    def test_filtering_removes_the_density_inflation(self):
        track_id, step, x, y = self.columns
        excluded = duplicate_tracks(duplicate_pairs(track_id, step, x, y))
        # The most crowded frame of the fixture, as the report sees it.
        busiest = max(set(step), key=lambda value: int((step == value).sum()))
        rows = step == busiest
        state = np.c_[x[rows], y[rows], np.zeros(rows.sum()), np.zeros(rows.sum())]
        retained = ~np.isin(track_id[rows], sorted(excluded))

        published = frame_local_density(state[:, :2])
        filtered = _filtered_density(state, retained)

        self.assertTrue(np.isnan(filtered[~retained]).all())
        self.assertLess(np.nanmax(filtered), np.nanmax(published) / 10)


if __name__ == "__main__":
    unittest.main()
