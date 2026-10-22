"""Regression coverage for AG-007: the publication analysis writer must never
produce an index path that aliases the forecasting analysis index.

``build_publication_analysis_dataset`` (src/analysis_pipeline.py) and
``data._analysis_index_path`` (the forecasting analysis index used by
``SceneDataset``) share the same stem,
``indices/opentraj-analysis-trajlets{suffix}.parquet``. Before this fix, four
``artifact_variant`` values collided with the forecasting namespace:
``"default"``, ``"temporal"``, ``"held-out-recording"``, and the unbounded
``"loso-<scene>"`` family. A ``"default"`` publication run silently overwrote
the forecasting index with publication-membership rows.

These tests pin:
1. Structural disjointness between every path the writer can produce and every
   path ``data._analysis_index_path`` can produce, over the four supported
   split strategies (including a representative ``leave_one_scene_out`` scene).
2. Each of the four known-colliding variants is rejected by
   ``build_publication_analysis_dataset`` before any file is written.
3. The CLI's ``--artifact-variant`` argument rejects the same variants via its
   ``type=`` validator.
4. ``artifact_variant="publication"`` is untouched: same suffix, same path.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import sys
import pytest

from data import _analysis_index_path
from analysis_pipeline import (
    _forecasting_collision,
    _publication_index_suffix,
    _validated_artifact_variant,
    build_publication_analysis_dataset,
    main,
)

# The four split strategies data._split_suffix supports (data.py:146-155),
# with a representative held-out scene for the unbounded leave-one-scene-out
# family.
FORECASTING_STRATEGIES = [
    ("grouped_scene_sequence", None),
    ("temporal", None),
    ("held_out_recording", None),
    ("leave_one_scene_out", "univ"),
]

COLLIDING_VARIANTS = ["default", "temporal", "held-out-recording", "loso-univ"]


def test_publication_variant_path_is_disjoint_from_every_forecasting_index_path(
    tmp_path: Path,
) -> None:
    forecasting_paths = {
        _analysis_index_path(tmp_path, strategy, held_out_scene)
        for strategy, held_out_scene in FORECASTING_STRATEGIES
    }
    publication_path = tmp_path / "indices" / (
        f"opentraj-analysis-trajlets{_publication_index_suffix('publication')}.parquet"
    )
    assert publication_path not in forecasting_paths
    assert publication_path == tmp_path / "indices" / "opentraj-analysis-trajlets-publication.parquet"


@pytest.mark.parametrize("variant", COLLIDING_VARIANTS)
def test_colliding_variants_are_detected_structurally(variant: str) -> None:
    assert _forecasting_collision(variant) is not None


@pytest.mark.parametrize("variant", ["publication", "held-out", "loso"])
def test_non_colliding_variants_are_not_flagged(variant: str) -> None:
    assert _forecasting_collision(variant) is None


@pytest.mark.parametrize("variant", COLLIDING_VARIANTS)
def test_build_publication_analysis_dataset_raises_before_writing_anything(
    tmp_path: Path, variant: str
) -> None:
    """Write-time guard: the raise must precede any file discovery or write.

    ``tmp_path`` has no ``converted`` directory at all, so if the guard were
    absent or placed after file discovery, this would instead raise
    ``FileNotFoundError`` (no recordings found). Getting ``ValueError`` here
    confirms the collision check runs first, and no ``indices``/``analysis``
    directory is created as a side effect.
    """
    with pytest.raises(ValueError, match=variant.replace("-", ".")):
        build_publication_analysis_dataset(tmp_path, artifact_variant=variant)
    assert not (tmp_path / "indices").exists()
    assert not (tmp_path / "analysis").exists()


def test_publication_variant_is_unaffected_by_the_guard(tmp_path: Path) -> None:
    # No converted files exist, so this still fails -- but with the *original*
    # FileNotFoundError, proving the guard did not fire for "publication".
    with pytest.raises(FileNotFoundError):
        build_publication_analysis_dataset(tmp_path, artifact_variant="publication")


@pytest.mark.parametrize("variant", COLLIDING_VARIANTS)
def test_cli_artifact_variant_type_rejects_colliding_variants(variant: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _validated_artifact_variant(variant)


def test_cli_artifact_variant_type_accepts_publication() -> None:
    assert _validated_artifact_variant("publication") == "publication"


@pytest.mark.parametrize("variant", COLLIDING_VARIANTS)
def test_cli_parser_wiring_rejects_colliding_variants(variant: str, monkeypatch) -> None:
    """Exercise argparse itself, not just the validator function.

    The sibling test calls ``_validated_artifact_variant`` directly, so it passes
    even if ``type=_validated_artifact_variant`` is dropped from the
    ``--artifact-variant`` argument -- the review of AG-007 confirmed that
    reverting the wiring alone left every other test green. This test drives
    ``main()`` with patched argv so the failure comes out of ``parse_args``:
    argparse converts an ``ArgumentTypeError`` into ``SystemExit(2)`` before any
    pipeline work begins.
    """
    monkeypatch.setattr(
        sys, "argv",
        ["analysis_pipeline", "datasets/does-not-exist", "--publication", "--artifact-variant", variant],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2


class TestSummaryPathKeyedBySplitStrategy:
    """Regression coverage for AG-009.

    ``build_analysis_dataset`` wrote its index to a strategy-keyed path
    (``_analysis_index_path``, suffixed ``-temporal`` / ``-held-out-recording``
    / ``-loso-<scene>``) but its summary to a flat
    ``analysis/opentraj_summary.json``, even though the summary's
    ``split_counts`` block is itself strategy-dependent. Two LOSO runs with
    different held-out scenes therefore wrote two index files but only one
    summary, silently attributed to whichever ran last.

    These tests pin, on a tiny synthetic two-recording fixture:
    1. Two distinct ``(strategy, held_out_scene)`` pairs produce two distinct
       summary paths.
    2. ``grouped_scene_sequence`` keeps today's flat filename unchanged, so no
       summary already on disk is orphaned.
    3. The summary body records the strategy/scene it describes.
    """

    @staticmethod
    def _write_fixture(root: Path) -> None:
        import json as _json
        from conftest import write_canonical_fixture

        converted = root / "converted"
        converted.mkdir()
        # Asymmetric track counts (2 vs. 3) so held-out-scene "a" and "b" leave
        # genuinely different-sized train populations behind, and the summary's
        # split_counts block actually differs between the two runs below.
        track_counts = {"recording_a": 2, "recording_b": 3}
        recordings = list(track_counts)
        for recording, offset in zip(recordings, (0.0, 100.0)):
            tracks = {
                track_id: [
                    (step, float(step) + offset + track_id, 0.0, None)
                    for step in range(40)
                ]
                for track_id in range(1, track_counts[recording] + 1)
            }
            write_canonical_fixture(
                converted / f"{recording}.traj.parquet", tracks,
                dataset="probe", scene_id=recording, scene_sequence=recording,
                recording=recording,
            )
        (root / "source.json").write_text(
            _json.dumps({"canonical_recordings": recordings})
        )

    def test_distinct_loso_scenes_write_distinct_summary_paths(self, tmp_path):
        from analysis_pipeline import build_analysis_dataset

        self._write_fixture(tmp_path)
        _, summary_path_a, summary_a = build_analysis_dataset(
            tmp_path, data_seed=42,
            split_strategy="leave_one_scene_out", held_out_scene="recording_a",
        )
        _, summary_path_b, summary_b = build_analysis_dataset(
            tmp_path, data_seed=42,
            split_strategy="leave_one_scene_out", held_out_scene="recording_b",
        )
        assert summary_path_a != summary_path_b
        assert summary_path_a.exists() and summary_path_b.exists()
        assert summary_a["split_counts"] != summary_b["split_counts"]
        assert summary_a["split_strategy"] == "leave_one_scene_out"
        assert summary_a["held_out_scene"] == "recording_a"
        assert summary_b["held_out_scene"] == "recording_b"

    def test_grouped_scene_sequence_keeps_flat_filename(self, tmp_path):
        from analysis_pipeline import build_analysis_dataset

        self._write_fixture(tmp_path)
        _, summary_path, summary = build_analysis_dataset(
            tmp_path, data_seed=42, split_strategy="grouped_scene_sequence",
        )
        assert summary_path == tmp_path / "analysis" / "opentraj_summary.json"
        assert summary["split_strategy"] == "grouped_scene_sequence"
        assert summary["held_out_scene"] is None
