import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from artifact_provenance import sha256_file
from canonical import canonical_metadata, schema_with_metadata
from paper_dataset_metrics import (
    EXPECTED_INDICATOR_IMPLEMENTATION_HASH,
    INDICATOR_CONTRACT,
)
from opentraj_dataset_stats import (
    DatasetGroup,
    _analysis_stats,
    _write_markdown,
    _write_tex,
    canonical_general_stats,
    main,
    summarize_group,
)


# M-001: `_analysis_stats` now refuses an indicator artifact whose provenance
# metadata does not match the contract and implementation hash the report writer
# stamps in (`src/opentraj_report.py` `provenance`), because it publishes that
# artifact's de-duplicated density column. Fixtures must therefore carry the same
# metadata a real `trajlet_indicators.parquet` carries. The two thresholds below
# are fixture values read back into the row, not the live source constants.
FIXTURE_DUPLICATE_SEPARATION_M = 0.05
FIXTURE_DUPLICATE_MIN_FRAMES = 3


def indicator_table(columns: dict) -> pa.Table:
    return pa.table(columns).replace_schema_metadata(
        {
            b"indicator_contract": INDICATOR_CONTRACT.encode(),
            b"indicator_implementation_hash": EXPECTED_INDICATOR_IMPLEMENTATION_HASH.encode(),
            b"duplicate_separation_m": str(FIXTURE_DUPLICATE_SEPARATION_M).encode(),
            b"duplicate_min_frames": str(FIXTURE_DUPLICATE_MIN_FRAMES).encode(),
        }
    )


class OpenTrajDatasetStatsTests(unittest.TestCase):
    def _complete_row_with_none_density(self) -> dict:
        # BC-001: `_analysis_stats` returns `average_local_density_ped_m2=None`
        # whenever no retained trajlet had a finite `local_density_max` (e.g.
        # scenes where at most one pedestrian is ever present at once), while
        # `summarize_group` still marks such rows "complete". The writers must
        # not crash formatting `None` with `:.3f`.
        return {
            "name": "SPARSE",
            "dataset": "sparse",
            "status": "complete",
            "recording_duration_hours": 1.0,
            "total_trajectory_duration_hours": 0.5,
            "raw_trajectories": 3,
            "trajlets_4_8s": 2,
            "non_static_trajlets_4_8s": 2,
            "average_local_density_ped_m2": None,
            "median_local_density_ped_m2": None,
            "local_density_trajlets_with_finite_value": 0,
            "average_local_density_deduplicated_ped_m2": None,
            "median_local_density_deduplicated_ped_m2": None,
            "local_density_deduplicated_trajlets_with_finite_value": 0,
        }

    def test_write_tex_renders_placeholder_for_none_density(self):
        row = self._complete_row_with_none_density()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.tex"
            _write_tex([row], path)
            contents = path.read_text()
        self.assertIn("n/a", contents)
        self.assertNotIn("None", contents)
        # The placeholder must not break the LaTeX row: exactly the expected
        # number of `&` column separators and a trailing `\\` row terminator.
        data_line = next(line for line in contents.splitlines() if line.startswith("SPARSE"))
        self.assertEqual(data_line.count("&"), 6)
        self.assertTrue(data_line.rstrip().endswith("\\\\"))

    def test_write_markdown_renders_placeholder_for_none_density(self):
        row = self._complete_row_with_none_density()
        result = {
            "opentraj_commit": "deadbeef",
            "definitions": {},
            "compatibility_note": "note",
            "datasets": [row],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_markdown(result, path)
            contents = path.read_text()
        self.assertIn("n/a", contents)
        self.assertNotIn("None", contents)
        data_line = next(line for line in contents.splitlines() if line.startswith("| SPARSE"))
        self.assertEqual(data_line.count("|"), 9)

    def test_summarize_group_marks_undefined_density_not_complete(self):
        # BC-001 (human-authorized follow-up): when every retained trajlet's
        # `local_density_max` is non-finite (e.g. a scene with at most one
        # pedestrian present at a time, so `frame_local_density` is NaN
        # throughout), `average_local_density_ped_m2` is None and the group
        # must NOT be reported "complete" -- the density average genuinely
        # cannot be computed, which is a different situation from a
        # successfully-computed statistic.
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory)
            dataset_dir = dataset_root / "sparse_dataset"
            converted_dir = dataset_dir / "converted"
            converted_dir.mkdir(parents=True)

            schema = schema_with_metadata(
                canonical_metadata(
                    dataset="sparse_dataset",
                    scene_id="place",
                    scene_sequence="seq1",
                    recording="seq1",
                )
            )
            table = pa.Table.from_arrays(
                [
                    pa.array([1, 1, 1], pa.int32()),
                    pa.array([0, 1, 2], pa.int32()),
                    pa.array([0.0, 1.0, 2.0], pa.float32()),
                    pa.array([0.0, 0.0, 0.0], pa.float32()),
                    pa.array([None, None, None], pa.float32()),
                ],
                schema=schema,
            )
            pq.write_table(table, converted_dir / "seq1.traj.parquet")

            indices_dir = dataset_dir / "indices"
            indices_dir.mkdir(parents=True)
            index_table = pa.table(
                {
                    "scene_sequence": ["seq1", "seq1"],
                    "retained_after_static_filter": [True, True],
                }
            )
            pq.write_table(index_table, indices_dir / "opentraj-analysis-trajlets.parquet")

            reports_dir = dataset_dir / "analysis" / "opentraj_reports"
            reports_dir.mkdir(parents=True)
            pq.write_table(
                indicator_table(
                    {
                        "scene_sequence": ["seq1", "seq1"],
                        "local_density_max": [math.nan, math.nan],
                        "local_density_max_deduplicated": [math.nan, math.nan],
                    }
                ),
                reports_dir / "trajlet_indicators.parquet",
            )

            group = DatasetGroup("SPARSE_GROUP", "sparse_dataset")
            result = summarize_group(group, dataset_root)

        self.assertIsNone(result["average_local_density_ped_m2"])
        self.assertNotEqual(result["status"], "complete")
        self.assertEqual(result["status"], "incomplete_density_undefined")

    def test_main_print_path_renders_placeholder_for_none_density(self):
        # BC-005: `main()`'s CLI print statement formats
        # `average_local_density_ped_m2` the same unguarded `:.3f` way that
        # `_write_tex`/`_write_markdown` used to crash on before BC-001. A
        # "complete" row with `average_local_density_ped_m2=None` reaching
        # `main()`'s print loop must not raise `TypeError` and must render
        # the same "n/a" placeholder instead.
        row = self._complete_row_with_none_density()
        result = {
            "opentraj_commit": "deadbeef",
            "definitions": {},
            "compatibility_note": "note",
            "datasets": [row],
        }
        with mock.patch("opentraj_dataset_stats.generate", return_value=result), mock.patch.object(
            sys, "argv", ["opentraj_dataset_stats"]
        ):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main()
        output = stdout.getvalue()
        self.assertIn("mean=n/a", output)
        self.assertIn("median=n/a", output)
        self.assertNotIn("None", output)

    def test_analysis_stats_hashes_match_independent_sha256_file(self):
        # BC-010: `_analysis_stats` must record `source_analysis_index_sha256`/
        # `source_indicator_report_sha256` fields that equal an independently
        # computed `artifact_provenance.sha256_file()` of the exact index/
        # indicator fixtures it read, so a regenerated artifact is traceable.
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory) / "hash_dataset"

            indices_dir = dataset_dir / "indices"
            indices_dir.mkdir(parents=True)
            index_path = indices_dir / "opentraj-analysis-trajlets.parquet"
            index_table = pa.table(
                {
                    "scene_sequence": ["seq1", "seq1"],
                    "retained_after_static_filter": [True, True],
                }
            )
            pq.write_table(index_table, index_path)

            reports_dir = dataset_dir / "analysis" / "opentraj_reports"
            reports_dir.mkdir(parents=True)
            indicator_path = reports_dir / "trajlet_indicators.parquet"
            pq.write_table(
                indicator_table(
                    {
                        "scene_sequence": ["seq1", "seq1"],
                        "local_density_max": [1.5, 2.5],
                        "local_density_max_deduplicated": [1.5, 2.0],
                    }
                ),
                indicator_path,
            )

            result = _analysis_stats(dataset_dir, sequences=None)

            self.assertEqual(result["source_analysis_index_path"], str(index_path))
            self.assertEqual(result["source_analysis_index_sha256"], sha256_file(index_path))
            self.assertEqual(result["source_indicator_report_path"], str(indicator_path))
            self.assertEqual(result["source_indicator_report_sha256"], sha256_file(indicator_path))
            # M-001: the duplicate-filter thresholds and implementation hash are
            # reported as recorded in the artifact the numbers came from.
            self.assertEqual(
                result["duplicate_separation_m"], FIXTURE_DUPLICATE_SEPARATION_M
            )
            self.assertEqual(result["duplicate_min_frames"], FIXTURE_DUPLICATE_MIN_FRAMES)
            self.assertEqual(
                result["indicator_implementation_hash"],
                EXPECTED_INDICATOR_IMPLEMENTATION_HASH,
            )

    def test_analysis_stats_reports_median_alongside_mean(self):
        # C-001/C-002 (user decision): the per-trajlet peak local density is a
        # mean of maxima and is dominated by its upper tail, so the median must
        # be published beside it, for both the unfiltered upstream-parity value
        # and the duplicate-filtered one.
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory) / "median_dataset"

            indices_dir = dataset_dir / "indices"
            indices_dir.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "scene_sequence": ["seq1"] * 4,
                        "retained_after_static_filter": [True] * 4,
                    }
                ),
                indices_dir / "opentraj-analysis-trajlets.parquet",
            )

            reports_dir = dataset_dir / "analysis" / "opentraj_reports"
            reports_dir.mkdir(parents=True)
            pq.write_table(
                indicator_table(
                    {
                        "scene_sequence": ["seq1"] * 4,
                        # One extreme value: mean 25.75, median 1.5.
                        "local_density_max": [1.0, 1.0, 2.0, 99.0],
                        # The extreme trajlet belongs to a removed duplicate
                        # identity, so it is non-finite in the filtered column.
                        "local_density_max_deduplicated": [1.0, 1.0, 2.0, math.nan],
                    }
                ),
                reports_dir / "trajlet_indicators.parquet",
            )

            result = _analysis_stats(dataset_dir, sequences=None)

        self.assertAlmostEqual(result["average_local_density_ped_m2"], 25.75)
        self.assertAlmostEqual(result["median_local_density_ped_m2"], 1.5)
        self.assertEqual(result["local_density_trajlets_with_finite_value"], 4)
        self.assertAlmostEqual(result["average_local_density_deduplicated_ped_m2"], 4 / 3)
        self.assertAlmostEqual(result["median_local_density_deduplicated_ped_m2"], 1.0)
        self.assertEqual(result["local_density_deduplicated_trajlets_with_finite_value"], 3)

    def test_general_stats_match_opentraj_span_definitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.traj.parquet"
            schema = schema_with_metadata(
                canonical_metadata(
                    dataset="sample",
                    scene_id="place",
                    scene_sequence="recording_1",
                    recording="recording_1",
                )
            )
            # Recording span: steps 0..10 = 4 s. Track spans: 0..10 and
            # 2..4 = 4.0 + 0.8 = 4.8 s. The first track has a gap, and the
            # OpenTraj general-stat definition intentionally includes it.
            # Rows are ordered by ``step`` (interleaving the two tracks), which is
            # the canonical row order every adapter writes and which
            # ``validate_canonical_file`` enforces; these statistics are computed
            # per track, so the ordering does not affect the expected values.
            table = pa.Table.from_arrays(
                [
                    pa.array([1, 1, 2, 2, 1], pa.int32()),
                    pa.array([0, 1, 2, 4, 10], pa.int32()),
                    pa.array([0, 1, 0, 1, 2], pa.float32()),
                    pa.array([0, 0, 1, 1, 0], pa.float32()),
                    pa.array([None] * 5, pa.float32()),
                ],
                schema=schema,
            )
            pq.write_table(table, path)
            result = canonical_general_stats([path])
            self.assertEqual(result["raw_trajectories"], 2)
            self.assertAlmostEqual(result["recording_duration_seconds"], 4.0)
            self.assertAlmostEqual(result["total_trajectory_duration_seconds"], 4.8)


if __name__ == "__main__":
    unittest.main()
