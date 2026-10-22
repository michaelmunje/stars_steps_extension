from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import paper_dataset_metrics
from paper_dataset_metrics import (
    REQUIRED_ROW_KEYS,
    PaperDataset,
    PaperSource,
    _aggregate_row,
    _validate_publication_payload,
    load_paper_datasets,
    publication_cv_for_xy,
    render_table1,
    render_table2,
    select_canonical_files,
)
from conftest import write_canonical_fixture


def _primitive(name: str, n: int, density: float, efficiency: float) -> dict:
    return {
        "canonical_dataset": name,
        "included_scene_sequences": [name],
        "source_trajectory_identities": {(name, name, value) for value in range(n)},
        "candidate_count": n,
        "retained_count": n,
        "local_density": np.full(n, density),
        "local_density_deduplicated": np.full(n, density),
        "path_efficiency": np.full(n, efficiency),
        "dca": np.full(n, 0.5),
        "ttc": np.arange(n, dtype=float),
        "cv_count": n,
        "cv_fde_sum": float(n),
        "index_path": f"{name}/index.parquet",
        "indicator_path": f"{name}/indicators.parquet",
        "canonical_file_revisions": [],
        "cv_source_revisions": [],
        "index_revision": {"path": f"{name}/index.parquet"},
        "indicator_revision": {"path": f"{name}/indicators.parquet"},
    }


def _dataset() -> PaperDataset:
    return PaperDataset("test", "Test", "Synthetic", 2, "Manual", ())


def test_publication_duration_definition():
    row = _aggregate_row(_dataset(), [_primitive("a", 750, 1.0, 1.0)])
    assert np.isclose(row["retained_trajlet_duration_hours"], 750 * 4.8 / 3600.0)


def test_density_mean_and_peak_are_per_trajlet_max_aggregates():
    primitive = _primitive("a", 3, 0.0, 1.0)
    primitive["local_density"] = np.array([1.0, 2.0, 6.0])
    primitive["local_density_deduplicated"] = np.array([1.0, 2.0, 6.0])
    row = _aggregate_row(_dataset(), [primitive])
    assert row["local_density_mean"] == 3.0
    assert row["local_density_peak"] == 6.0
    assert row["local_density_median"] == 2.0


def test_deduplicated_density_is_aggregated_separately_from_the_parity_value():
    # C-002: duplicate annotations inflate only the upper tail, so the filtered
    # column must be summarised on its own rather than reusing the unfiltered
    # one, and trajlets of a removed identity (NaN) must drop out of its count.
    primitive = _primitive("a", 3, 0.0, 1.0)
    primitive["local_density"] = np.array([1.0, 2.0, 600.0])
    primitive["local_density_deduplicated"] = np.array([1.0, 2.0, np.nan])
    row = _aggregate_row(_dataset(), [primitive])
    assert row["local_density_mean"] == 201.0
    assert row["local_density_median"] == 2.0
    assert row["finite_local_density_count"] == 3
    assert row["local_density_deduplicated_mean"] == 1.5
    assert row["local_density_deduplicated_median"] == 1.5
    assert row["local_density_deduplicated_peak"] == 2.0
    assert row["finite_local_density_deduplicated_count"] == 2


def test_combined_row_pools_primitive_records_not_row_means():
    row = _aggregate_row(
        _dataset(),
        [_primitive("many", 100, 1.0, 0.9), _primitive("few", 1, 100.0, 0.1)],
    )
    assert np.isclose(row["local_density_mean"], 200.0 / 101.0)
    assert not np.isclose(row["local_density_mean"], (1.0 + 100.0) / 2.0)


def test_ttc_denominator_is_every_retained_trajlet():
    primitive = _primitive("a", 4, 1.0, 1.0)
    primitive["ttc"] = np.array([0.5, 2.0, np.nan, 5.0])
    row = _aggregate_row(_dataset(), [primitive])
    assert row["ttc_lt_1_pct"] == 25.0
    assert row["ttc_lt_3_pct"] == 50.0


def test_straight_path_efficiency_primitive_is_one():
    xy = np.c_[np.arange(12), np.zeros(12)]
    efficiency = np.linalg.norm(xy[-1] - xy[0]) / np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    assert efficiency == 1.0


def test_publication_cv_uses_strict_complete_windows_and_last_displacement():
    steps = np.arange(20)
    xy = np.c_[steps * 0.25, np.zeros(20)]
    summary = publication_cv_for_xy(steps, xy)
    assert summary.count == 1
    assert summary.mean == 0.0
    # Quadratic x: manual rule is x7 + 12*(x7-x6), compared with x19.
    curved = np.c_[steps.astype(float) ** 2, np.zeros(20)]
    expected = abs((7.0**2 + 12.0 * (7.0**2 - 6.0**2)) - 19.0**2)
    assert publication_cv_for_xy(steps, curved).mean == expected
    assert publication_cv_for_xy(np.delete(steps, 10), np.delete(xy, 10, axis=0)).count == 0


def test_publication_cv_accepts_a_smoothed_signal_without_changing_window_population():
    steps = np.arange(25)
    raw = np.c_[steps.astype(float) ** 2, np.zeros(25)]
    smoothed = np.c_[steps.astype(float), np.zeros(25)]
    raw_summary = publication_cv_for_xy(steps, raw)
    smoothed_summary = publication_cv_for_xy(steps, smoothed)
    assert raw_summary.count == smoothed_summary.count == 6
    assert raw_summary.mean > 0.0
    assert smoothed_summary.mean == 0.0


def test_publication_table_schemas_exclude_legacy_debug_concepts():
    row = _aggregate_row(_dataset(), [_primitive("a", 4, 1.0, 1.0)])
    table1 = render_table1([row])
    table2 = render_table2([row])
    assert "Dataset & Setting & \\# Scenes & Dur. (h) & \\# Traj." in table1
    assert "Dataset & CV FDE & Path efficiency & DCA" in table2
    forbidden = (
        "Occupancy", "Effective area", "Area method", "Density p95",
        "Heading entropy", "RMS jerk", "Scenes analyzed", "status", "coverage",
    )
    assert not any(value in table1 + table2 for value in forbidden)


def test_publication_payload_rejects_nonfinite_cells_before_rendering():
    row = _aggregate_row(_dataset(), [_primitive("a", 4, 1.0, 1.0)])
    row["key"] = "steps"
    row["cv_fde_mean"] = float("nan")
    payload = {
        "rows": [row],
        "missing": {key: "fixture" for key in (
            "eth", "ucy", "madras", "grand_central", "ptdpes"
        )},
        "complete": False,
    }
    with np.testing.assert_raises_regex(ValueError, "invalid publication values"):
        _validate_publication_payload(payload, allow_partial=True)


def _missing_fixture(present: tuple[str, ...]) -> dict:
    """Build the payload['missing'] dict for every REQUIRED_ROW_KEYS not in `present`."""
    return {key: "fixture" for key in REQUIRED_ROW_KEYS if key not in present}


def _payload_with_single_row(row: dict, *, key: str = "steps", complete: bool = False) -> dict:
    row = dict(row)
    row["key"] = key
    return {
        "rows": [row],
        "missing": _missing_fixture((key,)),
        "complete": complete,
        "publication_cv_preprocessing_mode": row["publication_cv_preprocessing_mode"],
    }


def test_validate_publication_payload_rejects_row_order_membership_mismatch():
    row_steps = _aggregate_row(_dataset(), [_primitive("a", 4, 1.0, 1.0)])
    row_steps["key"] = "steps"
    row_eth = _aggregate_row(_dataset(), [_primitive("b", 4, 1.0, 1.0)])
    row_eth["key"] = "eth"
    # Rows are present out of REQUIRED_ROW_KEYS order relative to `missing`.
    payload = {
        "rows": [row_eth, row_steps],
        "missing": _missing_fixture(("steps", "eth")),
        "complete": False,
        "publication_cv_preprocessing_mode": row_steps["publication_cv_preprocessing_mode"],
    }
    with pytest.raises(ValueError, match="row order/membership mismatch"):
        _validate_publication_payload(payload, allow_partial=True)


def test_validate_publication_payload_rejects_strict_incomplete_without_allow_partial():
    row = _aggregate_row(_dataset(), [_primitive("a", 4, 1.0, 1.0)])
    payload = _payload_with_single_row(row, key="steps", complete=False)
    with pytest.raises(ValueError, match="Strict publication output must contain every required row"):
        _validate_publication_payload(payload, allow_partial=False)


def test_validate_publication_payload_rejects_ttc_denominator_mismatch():
    row = _aggregate_row(_dataset(), [_primitive("a", 4, 1.0, 1.0)])
    row["ttc_denominator"] = row["retained_opentraj_trajlet_count"] + 1
    payload = _payload_with_single_row(row, key="steps", complete=False)
    with pytest.raises(ValueError, match="TTC denominator is not all retained trajlets"):
        _validate_publication_payload(payload, allow_partial=True)


def test_validate_publication_payload_rejects_cv_preprocessing_mode_provenance_mismatch():
    row = _aggregate_row(_dataset(), [_primitive("a", 4, 1.0, 1.0)])
    assert row["publication_cv_preprocessing_mode"] == "raw_world_coordinates"
    payload = _payload_with_single_row(row, key="steps", complete=False)
    payload["publication_cv_preprocessing_mode"] = "opentraj_kalman_smooth"
    with pytest.raises(ValueError, match="publication CV preprocessing provenance mismatch"):
        _validate_publication_payload(payload, allow_partial=True)


def test_validate_publication_payload_rejects_empty_source_artifacts():
    row = _aggregate_row(_dataset(), [_primitive("a", 4, 1.0, 1.0)])
    row["source_artifacts"] = []
    payload = _payload_with_single_row(row, key="steps", complete=False)
    with pytest.raises(ValueError, match="source provenance is empty"):
        _validate_publication_payload(payload, allow_partial=True)


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "paper_datasets.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_load_paper_datasets_rejects_malformed_scene_sequences(tmp_path):
    payload = {
        "rows": [
            {
                "key": "steps",
                "label": "STEPS",
                "setting": "Test",
                "num_physical_scenes": 1,
                "annotation_source": "manual",
                "sources": [
                    # Neither a list nor the sentinel "all_canonical".
                    {"canonical_dataset": "steps", "scene_sequences": 123}
                ],
            }
        ]
    }
    manifest_path = _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match="scene_sequences must be a list or all_canonical"):
        load_paper_datasets(manifest_path)


def test_load_paper_datasets_rejects_row_keys_that_dont_match_required_row_keys(tmp_path):
    payload = {
        "rows": [
            {
                "key": "not_a_required_key",
                "label": "Bogus",
                "setting": "Test",
                "num_physical_scenes": 1,
                "annotation_source": "manual",
                "sources": [
                    {"canonical_dataset": "steps", "scene_sequences": ["seq_a"]}
                ],
            }
        ]
    }
    manifest_path = _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match="Publication manifest rows must be exactly"):
        load_paper_datasets(manifest_path)


def _write_canonical_parquet(
    path: Path, *, dataset: str, scene_id: str, scene_sequence: str, recording: str
) -> None:
    write_canonical_fixture(
        path, {1: [(step, float(step), 0.0, None) for step in range(5)]},
        dataset=dataset, scene_id=scene_id, scene_sequence=scene_sequence, recording=recording,
    )


def test_select_canonical_files_rejects_duplicate_scene_sequence_recording_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_dataset_metrics, "ROOT", tmp_path)
    converted = tmp_path / "datasets" / "synth" / "converted"
    # Two distinct files that both carry the same (scene_sequence, recording) identity
    # in their metadata (identity is metadata-derived, not filename-derived).
    _write_canonical_parquet(
        converted / "file_one.traj.parquet",
        dataset="synth", scene_id="s", scene_sequence="seq_a", recording="rec_same",
    )
    _write_canonical_parquet(
        converted / "file_two.traj.parquet",
        dataset="synth", scene_id="s", scene_sequence="seq_a", recording="rec_same",
    )
    source = PaperSource(canonical_dataset="synth", scene_sequences="all_canonical")
    with pytest.raises(ValueError, match="duplicate \\(scene_sequence, recording\\) identities"):
        select_canonical_files(source)


def test_select_canonical_files_raises_on_missing_requested_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_dataset_metrics, "ROOT", tmp_path)
    converted = tmp_path / "datasets" / "synth" / "converted"
    _write_canonical_parquet(
        converted / "rec_a.traj.parquet",
        dataset="synth", scene_id="s", scene_sequence="seq_a", recording="rec_a",
    )
    source = PaperSource(canonical_dataset="synth", scene_sequences=("seq_missing",))
    with pytest.raises(FileNotFoundError, match="publication scene sequences are missing"):
        select_canonical_files(source)


def test_select_canonical_files_raises_when_no_canonical_files_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_dataset_metrics, "ROOT", tmp_path)
    source = PaperSource(canonical_dataset="empty_synth", scene_sequences="all_canonical")
    with pytest.raises(FileNotFoundError, match="no canonical trajectories under"):
        select_canonical_files(source)
