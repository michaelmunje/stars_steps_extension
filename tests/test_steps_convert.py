import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import orjson
import pyarrow.parquet as pq

from datasets_preprocessing.steps import convert as steps_convert


def _person(track_id=7, *, observed=True, confirmed=True, source="global_kalman_posterior"):
    return {
        "track_id": track_id,
        "confirmed": confirmed,
        "observed_this_frame": observed,
        "state_source": source,
        "stale": not observed,
        "missed_seconds": 0.0 if observed else 0.05,
        "position_ground_xy": [1.0 + track_id, 2.0],
        "velocity_ground_xy_mps": [0.5, 0.0],
        "position_covariance": [[0.1, 0.01], [0.01, 0.2]],
        "spread_m": 0.2,
        "yaw_concentration": 0.9,
        "yaw_ground_rad": 0.25,
        "last_observed_camera_indices": [0, 2],
    }


def _frame(index, people, reason="age_cutoff"):
    return {
        "frame_index": index,
        "timestamp_ns": 1_000_000_000 + index * 50_000_000,
        "timestamp_s": 1.0 + index * 0.05,
        "sync_skew_ns": 100,
        "tracked_people": people,
        "fused_people": [],
        "present_cameras": [0, 1],
        "fresh_cameras": [0, 1],
        "missing_cameras": [],
        "stale_cameras": [],
        "output_finalization": {
            "mode": "fixed_lag",
            "reason": reason,
            "actual_lag_seconds": 1.0 if reason == "age_cutoff" else 0.0,
            "configured_delay_seconds": 1.0,
        },
    }


class StepsConversionTests(unittest.TestCase):
    def test_filters_predictions_and_end_flush_but_preserves_audit_rows(self):
        frames = [_frame(index, [_person()]) for index in range(17)]
        frames[8] = _frame(8, [_person(), _person(8, observed=False, source="global_kalman_prediction")])
        frames[16] = _frame(16, [_person()], reason="end_of_stream_flush")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "recording_bag/frames.cleaned.jsonl"
            source.parent.mkdir()
            source.write_bytes(b"".join(json.dumps(frame).encode() + b"\n" for frame in frames))

            output = steps_convert.convert_file(source, root / "dataset")
            trajectory = pq.read_table(output)
            quality = pq.read_table(root / "dataset/quality/recording.quality.parquet")
            frame_table = pq.read_table(root / "dataset/quality/recording.frames.parquet")
            summary = json.loads((root / "dataset/quality/recording.summary.json").read_text())

            self.assertEqual(trajectory.schema.remove_metadata(), steps_convert.OUTPUT_SCHEMA)
            self.assertEqual(trajectory["step"].to_pylist(), [0, 1])
            self.assertEqual(quality.num_rows, 4)
            self.assertEqual(quality["included_in_trajectory"].to_pylist(), [True, True, False, False])
            self.assertEqual(frame_table["included_in_trajectory"].to_pylist(), [True, True, False])
            self.assertEqual(summary["tracker_predictions_excluded"], 1)
            self.assertEqual(summary["end_of_stream_frames_excluded"], 1)
            self.assertAlmostEqual(summary["source_hz_measured_from_median_dt"], 20.0)

    def test_duplicate_track_in_one_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "duplicate_bag/frames.cleaned.jsonl"
            source.parent.mkdir()
            source.write_text(json.dumps(_frame(0, [_person(), _person()])) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate tracked person"):
                steps_convert.convert_file(source, root / "dataset")


class DecimationKeyScanTests(unittest.TestCase):
    """``_decimation_keys`` must return exactly what a full parse would, or nothing at all.

    The converter reads ``frame_index``/``timestamp_ns`` from every source frame but parses
    only the frames that survive decimation, so this byte scan stands in for ``orjson`` on
    seven of every eight lines. Its contract is therefore ``(int(frame[k]) for the two keys)``
    or ``None`` to request a full parse -- never a different number.
    """

    def assert_matches_full_parse(self, line):
        frame = orjson.loads(line)
        expected = (int(frame["frame_index"]), int(frame["timestamp_ns"]))
        self.assertEqual(steps_convert._decimation_keys(line), expected, line[:120])

    def assert_falls_back(self, line):
        orjson.loads(line)
        self.assertIsNone(steps_convert._decimation_keys(line), line[:120])

    def assert_never_wrong(self, line):
        """Weaker contract for layouts the scan may decline: right answer or ``None``."""
        frame = orjson.loads(line)
        expected = (int(frame["frame_index"]), int(frame["timestamp_ns"]))
        result = steps_convert._decimation_keys(line)
        self.assertIn(result, (expected, None), line[:120])

    def test_matches_full_parse_for_real_frame_layouts(self):
        nested = {
            "detections": [{"frame_index": -1, "timestamp_ns": 11}, {"timestamp_ns": 12}],
            "per_camera": {"3": {"frame_index": 98, "timestamp_ns": 99}},
        }
        person = {"track_id": 1, "last_observed_timestamp_ns": 77, "position_ground_xy": [1.0, 2.0]}
        cases = [
            # Real recordings emit sorted keys, so frame_index sits near the front and the
            # top-level timestamp_ns sits behind megabytes of nested payload.
            {"association": nested, "frame_index": 5, "timestamp_ns": 1234567890123456789,
             "tracked_people": [person]},
            # Reversed: both keys at the very end, timestamp_ns as the final member.
            {"tracked_people": [person], "association": nested, "frame_index": 5,
             "timestamp_ns": 1234567890123456789},
            # Same-named nested keys ahead of both real keys.
            {"a": nested, "b": nested, "timestamp_ns": 42, "frame_index": 5},
            # Nested keys only inside arrays of arrays.
            {"frame_index": 0, "deep": [[{"timestamp_ns": 1}], [{"frame_index": 2}]],
             "timestamp_ns": 3},
            # Braces and brackets inside string values must not shift the measured depth.
            {"note": "{{{[[[", "frame_index": 7, "other": "}}}]]]", "timestamp_ns": 8,
             "tail": "}{"},
            # Negative and boundary int64 values.
            {"frame_index": -9, "timestamp_ns": -(2 ** 63)},
            {"frame_index": 2 ** 31, "timestamp_ns": 2 ** 63 - 1},
            # Duplicate top-level keys: JSON resolves last-wins and so must the scan.
            {"frame_index": 5, "timestamp_ns": 8},
        ]
        for frame in cases:
            with self.subTest(keys=sorted(frame)):
                self.assert_matches_full_parse(orjson.dumps(frame))
        duplicated = b'{"frame_index": 1, "timestamp_ns": 2, "frame_index": 3, "timestamp_ns": 4}'
        self.assertEqual(orjson.loads(duplicated)["frame_index"], 3)
        self.assert_matches_full_parse(duplicated)

    def test_never_reports_a_wrong_value_behind_many_nested_decoys(self):
        """Enough same-named nested keys after the real one exhaust the candidate budget.

        The scan is then expected to decline (``None``) so the caller parses the frame; what it
        must never do is report one of the nested values.
        """
        decoys = {"d": [{"frame_index": 900 + i, "timestamp_ns": 900 + i} for i in range(6)]}
        line = orjson.dumps({"timestamp_ns": 42, "frame_index": 5, "tail": decoys})
        self.assertIsNone(steps_convert._decimation_keys(line))
        self.assert_never_wrong(line)

    def test_matches_full_parse_for_unusual_whitespace_and_spacing(self):
        for line in (
            b'{"frame_index":5,"timestamp_ns":6}',
            b'{ "frame_index" : 5 , "timestamp_ns" : 6 }',
            b'{\n  "frame_index": 5,\n  "timestamp_ns": 6\n}\n',
            b'{"frame_index":\t5,\r\n"timestamp_ns":\t6}\n',
        ):
            with self.subTest(line=line):
                self.assert_matches_full_parse(line)

    def test_requests_a_full_parse_when_it_cannot_prove_the_value(self):
        for line in (
            # Any backslash anywhere: escaped quotes would break quote-delimited scanning.
            orjson.dumps({"note": 'a "quoted" word', "frame_index": 1, "timestamp_ns": 2}),
            orjson.dumps({"note": "back\\slash", "frame_index": 1, "timestamp_ns": 2}),
            # Values that are not JSON integer literals.
            b'{"frame_index": 1.0, "timestamp_ns": 2}',
            b'{"frame_index": "1", "timestamp_ns": 2}',
            b'{"frame_index": 1, "timestamp_ns": true}',
            b'{"frame_index": 1, "timestamp_ns": 2e3}',
            # Present only inside a nested object: there is no top-level member to read.
            b'{"inner": {"frame_index": 1, "timestamp_ns": 2}}',
            # Top-level container is an array, so nothing sits at object depth 1.
            b'[{"frame_index": 1, "timestamp_ns": 2}]',
        ):
            with self.subTest(line=line):
                self.assert_falls_back(line)

    def test_integers_wider_than_int64_are_left_to_the_parser(self):
        """orjson decodes an over-wide integer literal as a float, so the literal is not the value.

        Reading the digits here would produce 99999999999999999999 while the converter's
        ``int(frame["timestamp_ns"])`` produces int(1e20) == 100000000000000000000.
        """
        line = b'{"frame_index": 1, "timestamp_ns": 99999999999999999999}'
        parsed = orjson.loads(line)["timestamp_ns"]
        self.assertIsInstance(parsed, float)
        self.assertNotEqual(int(parsed), 99999999999999999999)
        self.assertIsNone(steps_convert._decimation_keys(line))
        self.assert_never_wrong(line)


def _write_frames(source, frames):
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"".join(json.dumps(frame).encode() + b"\n" for frame in frames))


def _digests(root, recording):
    return {
        name: hashlib.sha256((root / name.format(recording=recording)).read_bytes()).hexdigest()
        for name in (
            "converted/{recording}.traj.parquet",
            "quality/{recording}.quality.parquet",
            "quality/{recording}.frames.parquet",
        )
    }


class DecimatedFrameHandlingTests(unittest.TestCase):
    def test_out_of_order_frame_index_still_raises_on_a_decimated_frame(self):
        frames = [_frame(index, [_person()]) for index in range(12)]
        frames[5]["frame_index"] = 3
        self.assertTrue((3 - 0) % steps_convert.SOURCE_FRAMES_PER_OUTPUT, "frame must be decimated")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "backwards_bag/frames.cleaned.jsonl"
            _write_frames(source, frames)
            with self.assertRaisesRegex(ValueError, r":6: frame_index is not strictly increasing \(4 -> 3\)"):
                steps_convert.convert_file(source, root / "dataset")

    def test_adversarial_key_layout_converts_identically_to_the_plain_layout(self):
        plain = [_frame(index, [_person()]) for index in range(17)]
        decoy = {"per_camera": {"3": {"frame_index": 777, "timestamp_ns": 777}},
                 "detections": [{"frame_index": 888, "timestamp_ns": 888}]}
        adversarial = []
        for frame in plain:
            # Same frame, but with decoy keys on both sides of the real ones, the real keys
            # moved to the end of the object, and braces inside a string value.
            reordered = {"pre_pose_quality_by_camera": decoy, "note": "}{[]"}
            reordered.update({k: v for k, v in frame.items() if k not in ("frame_index", "timestamp_ns")})
            reordered["projected_detections"] = decoy
            reordered["frame_index"] = frame["frame_index"]
            reordered["timestamp_ns"] = frame["timestamp_ns"]
            adversarial.append(reordered)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "recording_bag/frames.cleaned.jsonl"
            digests = {}
            summaries = {}
            for label, frames in (("plain", plain), ("adversarial", adversarial)):
                # One source path for both runs: the recorded source_file metadata is part of
                # the canonical Parquet, so a per-label path would differ for that reason alone.
                _write_frames(source, frames)
                dataset = root / label
                steps_convert.convert_file(source, dataset, provenance={"git_commit": "x", "git_dirty": False})
                digests[label] = _digests(dataset, "recording")
                summaries[label] = json.loads((dataset / "quality/recording.summary.json").read_text())
            self.assertEqual(digests["plain"], digests["adversarial"])
            self.assertEqual(summaries["plain"], summaries["adversarial"])
            self.assertEqual(summaries["plain"]["source_frames"], 17)
            self.assertEqual(summaries["plain"]["sampled_2p5hz_frames"], 3)

    def test_only_the_sampled_frames_are_fully_parsed(self):
        frames = [_frame(index, [_person()]) for index in range(17)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "recording_bag/frames.cleaned.jsonl"
            _write_frames(source, frames)
            with mock.patch.object(steps_convert.orjson, "loads", wraps=orjson.loads) as loads:
                steps_convert.convert_file(source, root / "dataset")
            self.assertEqual(loads.call_count, 3, "one parse per sampled frame, none per skipped frame")

    def test_blank_lines_between_frames_are_skipped(self):
        frames = [_frame(index, [_person()]) for index in range(9)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "blank_bag/frames.cleaned.jsonl"
            source.parent.mkdir(parents=True)
            source.write_bytes(
                b"\n".join(json.dumps(frame).encode() for frame in frames).replace(b"\n", b"\n\n   \n")
                + b"\n"
            )
            steps_convert.convert_file(source, root / "dataset")
            summary = json.loads((root / "dataset/quality/blank.summary.json").read_text())
            self.assertEqual(summary["source_frames"], 9)
            self.assertEqual(summary["sampled_2p5hz_frames"], 2)


if __name__ == "__main__":
    unittest.main()
