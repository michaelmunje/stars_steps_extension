import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis_trajlets import OpenTrajTrajletConvention, extract_opentraj_trajlets
from analysis_pipeline import _analysis_index_table, attach_trajnet_categories, build_analysis_view
from artifact_provenance import sha256_file
from canonical import canonical_metadata, decode_metadata, schema_with_metadata
from normalization import inverse_scene, normalize_scene
from opentraj_report import compute_opentraj_report
from preprocessing_views import OpenTrajKalmanModel, preprocess_forecast_scene
from split_manifests import assert_no_sequence_leakage, grouped_manifest, loso_manifest
from opentraj_indicators import frame_local_density


ROOT = Path(__file__).resolve().parents[1]


def _trajectory(n: int) -> tuple[np.ndarray, np.ndarray]:
    steps = np.arange(n, dtype=np.int32)
    xy = np.c_[steps * 0.11, np.zeros(n)]
    return steps, xy


class ExactOpenTrajTrajletTests(unittest.TestCase):
    def extract(self, n: int, xy: np.ndarray | None = None):
        steps, default_xy = _trajectory(n)
        return extract_opentraj_trajlets(
            dataset="test", scene_id="scene", scene_sequence="sequence", agent_id=1,
            steps=steps, xy=default_xy if xy is None else xy,
        )

    def test_twelve_samples_produce_no_candidate(self):
        self.assertEqual(self.extract(12), [])

    def test_thirteen_samples_produce_one_candidate_with_external_endpoint(self):
        rows = self.extract(13)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.source_indices, tuple(range(12)))
        self.assertEqual(row.source_returned_end_step, 11)
        self.assertEqual(row.source_static_endpoint_step, 12)
        self.assertEqual(row.n_returned_samples, 12)
        self.assertAlmostEqual(row.returned_end_time - row.start_time, 4.4)
        self.assertAlmostEqual(row.static_endpoint_time - row.start_time, 4.8)

    def test_stride_is_seven_samples(self):
        rows = self.extract(20)
        self.assertEqual([row.source_start_step for row in rows], [0, 7])

    def test_returned_last_sample_does_not_affect_static_test(self):
        steps, xy = _trajectory(13)
        reference = self.extract(13, xy)[0].static_displacement_m
        xy[11] = [10_000, -10_000]
        self.assertEqual(self.extract(13, xy)[0].static_displacement_m, reference)

    def test_external_endpoint_changes_static_test_but_is_not_returned(self):
        steps, xy = _trajectory(13)
        reference = self.extract(13, xy)[0]
        xy[12] = [0, 0]
        changed = self.extract(13, xy)[0]
        self.assertNotEqual(changed.static_displacement_m, reference.static_displacement_m)
        self.assertNotIn(12, changed.source_indices)

    def test_threshold_is_strictly_greater_than_one(self):
        steps = np.arange(13)
        xy = np.zeros((13, 2)); xy[12, 0] = 1.0
        equal = self.extract(13, xy)[0]
        self.assertTrue(equal.opentraj_static)
        self.assertFalse(equal.retained_after_static_filter)
        xy[12, 0] = np.nextafter(1.0, 2.0)
        self.assertTrue(self.extract(13, xy)[0].retained_after_static_filter)

    def test_gap_rejects_candidate_and_endpoint_crossing(self):
        steps = np.r_[np.arange(12), 13]
        xy = np.c_[steps, np.zeros(len(steps))]
        rows = extract_opentraj_trajlets(
            dataset="test", scene_id="scene", scene_sequence="sequence", agent_id=1,
            steps=steps, xy=xy,
        )
        self.assertEqual(rows, [])

    def test_count_formula_matches_upstream_reference_logic(self):
        convention = OpenTrajTrajletConvention()
        returned, stride = convention.sample_counts(0.4)
        self.assertEqual((returned, stride), (12, 7))
        for n in range(2, 100):
            upstream_starts = list(range(0, n - returned, stride))
            self.assertEqual(
                [row.source_start_step for row in self.extract(n)], upstream_starts
            )

    def test_independent_trajnet_category_join_uses_sequence_agent_and_start(self):
        rows=self.extract(20)
        joined=attach_trajnet_categories(
            rows,
            [
                {"scene_sequence":"sequence","recording":"sequence","primary_track_id":1,"start_step":0,"scene_tag":2},
                {"scene_sequence":"other","recording":"other","primary_track_id":1,"start_step":7,"scene_tag":4},
            ],
        )
        self.assertEqual([row.trajnet_category for row in joined],[2,None])


class PreprocessingLeakageTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.xy = np.cumsum(rng.normal(size=(20, 2, 2)), axis=0).astype(np.float32)
        self.valid = np.ones((20, 2), dtype=bool)

    def test_safe_modes_do_not_change_when_only_future_changes(self):
        changed = self.xy.copy(); changed[8:] += 10_000
        for mode in ("raw", "causal_kalman", "observation_only_smooth"):
            first = preprocess_forecast_scene(self.xy, self.valid, input_mode=mode)
            second = preprocess_forecast_scene(changed, self.valid, input_mode=mode)
            np.testing.assert_allclose(first.input_xy[:8], second.input_xy[:8], atol=1e-6)
            self.assertFalse(first.uses_future_information_for_input)

    def test_offline_compatibility_mode_is_explicitly_noncausal(self):
        changed = self.xy.copy(); changed[8:] += 10_000
        first = preprocess_forecast_scene(self.xy, self.valid, input_mode="opentraj_offline_smooth_compat")
        second = preprocess_forecast_scene(changed, self.valid, input_mode="opentraj_offline_smooth_compat")
        self.assertTrue(first.uses_future_information_for_input)
        self.assertGreater(np.max(np.abs(first.input_xy[:8] - second.input_xy[:8])), 1e-4)

    def test_raw_ground_truth_is_preserved(self):
        result = preprocess_forecast_scene(self.xy, self.valid)
        np.testing.assert_array_equal(result.target_xy, self.xy[8:])


class NormalizationSafetyTests(unittest.TestCase):
    def test_transform_is_future_invariant(self):
        xy = np.zeros((20, 2, 2)); xy[:8, 0, 0] = np.arange(8)
        changed = xy.copy(); changed[8:, 0] = [900, -500]
        _, first = normalize_scene(xy, 8)
        _, second = normalize_scene(changed, 8)
        self.assertEqual(first.rotation, second.rotation)
        np.testing.assert_array_equal(first.center, second.center)

    def test_constant_velocity_is_rigid_transform_invariant(self):
        rng = np.random.default_rng(2)
        world = np.cumsum(rng.normal(size=(20, 1, 2)), axis=0)
        normalized, transform = normalize_scene(world, 8)
        world_cv = world[7, 0] + np.arange(1, 13)[:, None] * (world[7, 0] - world[6, 0])
        normalized_cv = normalized[7, 0] + np.arange(1, 13)[:, None] * (normalized[7, 0] - normalized[6, 0])
        restored = inverse_scene(normalized_cv[:, None], transform.rotation, transform.center)[:, 0]
        np.testing.assert_allclose(restored, world_cv, atol=1e-10)


class KalmanParityTests(unittest.TestCase):
    def test_numerical_parity_with_pinned_upstream_when_available(self):
        candidates = [
            Path(os.environ.get("OPENTRAJ_ROOT", "")),
            ROOT / "references" / "opentraj",
        ]
        upstream = next((path for path in candidates if (path / "opentraj/toolkit/utils/kalman_smoother.py").exists()), None)
        if upstream is None:
            self.skipTest("Pinned OpenTraj checkout unavailable; set OPENTRAJ_ROOT")
        module_path = upstream / "opentraj/toolkit/utils/kalman_smoother.py"
        spec = importlib.util.spec_from_file_location("pinned_opentraj_kalman", module_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        xy = np.c_[np.linspace(0, 4, 20), np.sin(np.linspace(0, 2, 20))]
        expected_pos, expected_vel = module.KalmanModel(0.4, n_dim=2, n_iter=7).smooth(xy)
        actual_pos, actual_vel = OpenTrajKalmanModel(0.4, n_dim=2, n_iter=7).smooth(xy)
        np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-10)
        np.testing.assert_allclose(actual_vel, expected_vel, atol=1e-10)


class ParallelAnalysisViewTests(unittest.TestCase):
    def test_track_parallel_view_matches_serial_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = np.tile(np.arange(20, dtype=np.int32), 2)
            track_ids = np.repeat(np.array([1, 2], dtype=np.int32), 20)
            x = np.r_[np.arange(20) * 0.2, np.arange(20) * 0.3].astype(np.float32)
            y = np.r_[np.sin(np.arange(20) / 4), np.cos(np.arange(20) / 5)].astype(np.float32)
            # Canonical files are stored step-sorted (every adapter writes them that
            # way and validate_canonical_file enforces it), so interleave the two
            # tracks by step. Stable argsort keeps each track's own row order.
            order = np.argsort(steps, kind="stable")
            steps, track_ids, x, y = steps[order], track_ids[order], x[order], y[order]
            metadata = canonical_metadata(
                dataset="test",
                scene_id="scene",
                scene_sequence="sequence",
                recording="recording",
            )
            table = pa.Table.from_arrays(
                [
                    pa.array(track_ids),
                    pa.array(steps),
                    pa.array(x),
                    pa.array(y),
                    pa.nulls(len(steps), type=pa.float32()),
                ],
                schema=schema_with_metadata(metadata),
            )
            source = root / "recording.traj.parquet"
            pq.write_table(table, source)
            serial_path, serial_rows, serial_stats = build_analysis_view(
                source, root / "serial.analysis.parquet", track_workers=1
            )
            parallel_path, parallel_rows, parallel_stats = build_analysis_view(
                source, root / "parallel.analysis.parquet", track_workers=2
            )
            serial = pq.read_table(serial_path)
            parallel = pq.read_table(parallel_path)
            for column in ("smoothed_x", "smoothed_y", "smoothed_vx", "smoothed_vy"):
                np.testing.assert_allclose(
                    serial[column].to_numpy(), parallel[column].to_numpy(), atol=0.0
                )
            self.assertEqual(
                [row.as_index_row() for row in serial_rows],
                [row.as_index_row() for row in parallel_rows],
            )
            for key in (
                "full_tracks",
                "continuous_segments",
                "valid_opentraj_trajlets",
                "retained_analysis_trajlets",
            ):
                self.assertEqual(serial_stats[key], parallel_stats[key])

    def test_indicator_artifact_records_exact_source_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "converted" / "recording.traj.parquet"
            source.parent.mkdir(parents=True)
            steps = np.tile(np.arange(20, dtype=np.int32), 2)
            # Step-sorted row order, as every adapter writes and
            # validate_canonical_file enforces; stable argsort preserves each
            # track's own ordering.
            order = np.argsort(steps, kind="stable")
            table = pa.Table.from_arrays(
                [
                    pa.array(np.repeat([1, 2], 20)[order], type=pa.int32()),
                    pa.array(steps[order], type=pa.int32()),
                    pa.array(np.r_[np.arange(20) * 0.2, np.arange(20) * 0.3][order], type=pa.float32()),
                    pa.array(np.r_[np.zeros(20), np.ones(20)][order], type=pa.float32()),
                    pa.nulls(40, type=pa.float32()),
                ],
                schema=schema_with_metadata(
                    canonical_metadata(
                        dataset="test",
                        scene_id="scene",
                        scene_sequence="sequence",
                        recording="recording",
                    )
                ),
            )
            pq.write_table(table, source)
            view_path = root / "analysis" / "opentraj_kalman_smooth" / "recording.analysis.parquet"
            view_path.parent.mkdir(parents=True)
            _, rows, _ = build_analysis_view(source, view_path)
            index_path = root / "indices" / "opentraj-analysis-trajlets-publication.parquet"
            index_path.parent.mkdir(parents=True)
            index_table = _analysis_index_table(
                rows,
                sources=[source],
                dataset_dir=root,
                purpose="publication_dataset_characterization",
            )
            pq.write_table(index_table, index_path)
            metric_path, _ = compute_opentraj_report(
                root,
                index_path=index_path,
                sequences={"sequence"},
                output_dir=root / "analysis" / "opentraj_reports_publication",
            )
            metadata = decode_metadata(pq.ParquetFile(metric_path).schema_arrow)
            self.assertEqual(
                metadata["source_analysis_index_sha256"], sha256_file(index_path)
            )
            self.assertEqual(
                metadata["indicator_contract"],
                "opentraj-retained-trajlet-indicators-v1",
            )


class IndicatorParityTests(unittest.TestCase):
    """Pin `frame_local_density` to expectations derived from the upstream OpenTraj formula.

    Source: `references/opentraj/opentraj/toolkit/indicators/crowd_density.py::local_density`
    (pinned commit 97ec1d5e1b579f26febeaa9083705952deac338a). With a=1 it computes
    rho_i = 1/(2*pi) * sum_j exp(-d_ij**2 / (2*b_j**2)) / b_j**2, where b_j is pedestrian j's
    nearest-neighbour distance: min over d_jk > 0.01, falling back to 0.01 when pedestrian j
    has no neighbour farther than that. The literals below are hand arithmetic on that formula
    for the given fixtures, not a re-run of our implementation, so they still fail if the kernel
    is reweighted by the target pedestrian's own scalar bandwidth b_i instead of each
    contributor's b_j.
    """

    def test_local_density_uses_each_contributors_bandwidth(self):
        # xy=[[0,0],[.5,0],[3,0]] gives distance rows [0,.5,3], [.5,0,2.5], [3,2.5,0], hence
        # nearest-neighbour bandwidths b=[.5,.5,2.5] and b**2=[.25,.25,6.25]:
        #   rho_0 = (1/2pi)(1/.25 + e**-.5/.25 + e**-.72/6.25)   = 1.0351442273033429
        #   rho_1 = (1/2pi)(e**-.5/.25 + 1/.25 + e**-.5/6.25)    = 1.0381943593086056
        #   rho_2 = (1/2pi)(e**-18/.25 + e**-12.5/.25 + 1/6.25)  = 0.025467173051503596
        # (the exponents are -d_ij**2/(2*b_j**2); the pinned upstream module reproduces these
        # values, and the by-hand arithmetic agrees with it to 2.2e-16, well inside atol below).
        xy=np.array([[0.,0.],[.5,0.],[3.,0.]])
        expected=np.array([1.0351442273033429,1.0381943593086056,0.025467173051503596])
        np.testing.assert_allclose(frame_local_density(xy),expected,atol=1e-12)

    def test_local_density_falls_back_to_the_upstream_bandwidth_floor(self):
        # Upstream substitutes min_dist=0.01 when a pedestrian's distance row holds no entry
        # above 0.01; two coincident pedestrians is the minimal fixture that triggers it (any
        # neighbour farther than 0.01 m would supply a real bandwidth instead). Then b=[.01,.01]
        # and every d_ij is 0, so for both pedestrians
        #   rho = (1/2pi)(1/.01**2 + 1/.01**2) = 10000/pi = 3183.098861837907.
        xy=np.array([[1.,2.],[1.,2.]])
        expected=np.array([3183.098861837907,3183.098861837907])
        np.testing.assert_allclose(frame_local_density(xy),expected,atol=1e-12)


class LocalDensityLiveParityTests(unittest.TestCase):
    """Re-run the pinned upstream `local_density` and compare it with ours frame by frame.

    Mirrors `KalmanParityTests`: the upstream module is the reference, so no expectation here
    restates our implementation. Unlike the Kalman module, upstream
    `opentraj/toolkit/indicators/crowd_density.py` imports sibling `toolkit.*` modules (and an
    unused OpenCV dependency), so both the checkout root and its `opentraj/` package directory
    go on `sys.path` and `cv2` is stubbed when absent; all of that is undone afterwards.
    """

    def _upstream_module(self):
        candidates=[Path(os.environ.get("OPENTRAJ_ROOT","")),ROOT/"references"/"opentraj"]
        root=next((path for path in candidates if (path/"opentraj/toolkit/indicators/crowd_density.py").exists()),None)
        if root is None:
            self.skipTest("Pinned OpenTraj checkout unavailable; set OPENTRAJ_ROOT")
        added=[entry for entry in (str(root),str(root/"opentraj")) if entry not in sys.path]
        sys.path[:0]=added
        self.addCleanup(lambda:[sys.path.remove(entry) for entry in added if entry in sys.path])
        if "cv2" not in sys.modules:
            sys.modules["cv2"]=types.ModuleType("cv2")  # vendored loaders import it but never use it here
            self.addCleanup(sys.modules.pop,"cv2",None)
        spec=importlib.util.spec_from_file_location(
            "pinned_opentraj_crowd_density",root/"opentraj/toolkit/indicators/crowd_density.py"
        )
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_frame_local_density_matches_pinned_upstream(self):
        module=self._upstream_module()
        generator=np.random.default_rng(0)
        frames=[]; ours=[]
        for index in range(20):
            xy=generator.uniform(-5.,5.,size=(3,2))
            frames.append(pd.DataFrame({
                "frame_id":[index]*3,"agent_id":[0,1,2],"label":["pedestrian"]*3,
                "scene_id":["scene"]*3,"timestamp":[index*0.4]*3,
                "pos_x":xy[:,0],"pos_y":xy[:,1],
                "vel_x":np.zeros(3),"vel_y":np.zeros(3),
            }))
            ours.append(frame_local_density(xy))
        # Upstream returns per-trajlet maxima but annotates each frame with the per-agent
        # density in-place as 'p_local'; that column is the quantity frame_local_density returns.
        module.local_density(frames,{},"synthetic")
        upstream=np.concatenate([frame["p_local"].to_numpy() for frame in frames])
        self.assertEqual(upstream.shape,(60,))
        self.assertTrue(np.isfinite(upstream).all())
        np.testing.assert_allclose(np.concatenate(ours),upstream,atol=1e-12)


class PinnedParityArtifactTests(unittest.TestCase):
    def test_eth_ucy_parity_artifact_is_exact_and_pinned(self):
        payload=json.loads((ROOT/"references/opentraj_parity_eth_ucy.json").read_text())
        self.assertEqual(payload["pinned_commit"],"97ec1d5e1b579f26febeaa9083705952deac338a")
        self.assertTrue(payload["indicators"]["passed"])
        for metric,value in payload["indicators"].items():
            if metric.endswith("error"):
                self.assertEqual(value,0.0)
        for recording in payload["trajlets"]:
            self.assertTrue(recording["passed"])
            self.assertEqual(recording["upstream_candidates"],recording["ours_candidates"])
            self.assertEqual(recording["upstream_retained"],recording["ours_retained"])
            self.assertTrue(recording["returned_indices_equal"])


if __name__ == "__main__":
    unittest.main()
