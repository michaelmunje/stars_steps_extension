"""Direct coverage for forecast-window materialization and its loud-failure guards.

``materialize_forecast_window`` builds every tensor for the ``individual_track``,
``scholler_masked_track`` and ``social_gan_complete_scene`` memberships, but its
padding/offset branch is not reachable from the preset or dataset-analysis tests
(those only build windows whose agents span the whole recording). The first tests
below exercise that branch directly, with expectations derived from the fixture
rather than from the implementation.

The remaining tests cover two guards that turn silent corruption into an
exception: the ``canonical_stride == 1`` assumption of the generic
enumeration path in ``data.SceneDataset``, and the non-positive ``random_scale``
draw in ``forecasting_representation``. The scale guard is kept here, next to
the other loud-failure guards it was added with, rather than in
``test_forecasting_presets.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from config import Config, DatasetSpec
from data import SceneDataset, prepare_splits
from forecasting_config import resolve_forecasting_config
from forecasting_representation import prepare_forecast_representation
from forecasting_windows import (
    ForecastWindowRecord,
    RecordingWindowCache,
    enumerate_forecast_windows,
    materialize_forecast_window,
)
from conftest import write_canonical_fixture


ROOT = Path(__file__).resolve().parents[1]


def _write_window_fixture(path: Path, tracks, *, recording: str = "recording") -> Path:
    """Canonical fixture whose position is ``x = step`` and ``y = track_id``.

    Same shape as ``test_forecasting_presets._write_window_fixture``; the
    position encoding is what makes the materialized window's contents
    predictable without consulting the implementation.
    """
    return write_canonical_fixture(
        path,
        {
            track_id: [(int(step), float(step), float(track_id), np.nan) for step in steps]
            for track_id, steps in tracks.items()
        },
        dataset="test", scene_id="scene", scene_sequence="sequence", recording=recording,
    )


def test_scholler_masked_window_zero_pads_after_its_actual_samples(tmp_path):
    # Schöller's convention (references/scholler-cv/ped_dataset.py): every suffix
    # with >= 10 actual samples becomes a 20-step window, zero-padded and masked.
    # On a 12-step track the third window starts at step 2 and therefore has
    # exactly 10 real samples (steps 2..11) followed by 10 padded ones.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(12)})
    config = resolve_forecasting_config(
        preset="scholler", config_overrides={"scene_membership": "scholler_masked_track"}
    )
    record = enumerate_forecast_windows(path, config)[2]
    assert (record.start_step, record.actual_samples) == (2, 10)
    assert (record.observation_end_step, record.future_end_step) == (9, 21)

    xy, valid = materialize_forecast_window(record)
    assert xy.shape == (20, 1, 2)
    assert valid.shape == (20, 1)
    assert valid[:10, 0].all()
    assert not valid[10:, 0].any()
    np.testing.assert_array_equal(xy[10:], np.zeros((10, 1, 2), dtype=np.float32))
    # Fixture encoding: the sample at window offset k is the track's step 2 + k,
    # written as (x, y) = (step, track_id) = (2 + k, 1).
    expected = np.stack(
        (np.arange(2, 12, dtype=np.float32), np.ones(10, dtype=np.float32)), axis=1
    )
    np.testing.assert_array_equal(xy[:10, 0], expected)


def test_complete_window_materializes_every_step_of_every_agent(tmp_path):
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(20), 2: np.arange(20)})
    config = resolve_forecasting_config(preset="social_gan")
    record = enumerate_forecast_windows(path, config)[0]
    assert record.actual_samples == 20
    assert record.complete_agent_ids == (1, 2)

    xy, valid = materialize_forecast_window(record)
    assert valid.all()
    np.testing.assert_array_equal(xy[:, 0, 0], np.arange(20, dtype=np.float32))
    np.testing.assert_array_equal(xy[:, 1, 1], np.full(20, 2.0, dtype=np.float32))


def test_interleaved_tracks_land_in_their_own_agent_columns(tmp_path):
    # Pins which window row is matched to which agent column. Canonical
    # recordings are written in (step, track_id) order, so a track's rows are
    # scattered one per step through the window rather than being contiguous:
    # matching rows to agents may not assume anything about `track_id` order,
    # only about `step` order (which `RecordingWindowCache` asserts per read).
    #
    # The fixture makes every wrong pairing visible: the track ids are
    # non-consecutive (3, 7, 40), one of them covers only part of the window,
    # one listed agent is absent from the recording entirely, and the agent
    # order requested is deliberately not ascending, so the agent columns are
    # NOT in sorted-track-id order.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {7: np.arange(20), 3: np.arange(20), 40: np.arange(4, 14)})

    on_file = pq.read_table(path, columns=["track_id", "step"])
    file_track_ids = on_file["track_id"].to_numpy()
    file_steps = on_file["step"].to_numpy()
    assert np.all(np.diff(file_steps) >= 0), "canonical fixtures are step-sorted"
    assert not np.all(np.diff(file_track_ids) >= 0), (
        "the fixture must interleave tracks by step; a track-major file would "
        "not exercise the row-to-agent matching this test pins"
    )

    # `actual_samples=0` declares no complete samples, so the completeness
    # assertion cannot fire and the NaN/valid pattern below is what is under
    # test. `target_agent_id=None` likewise suppresses the zero-fill, so an
    # unmatched slot stays NaN instead of being masked by a zero.
    record = ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path, 0, 7, 19, None, (7, 3, 40, 99), 0,
    )
    xy, valid = materialize_forecast_window(record)
    assert xy.shape == (20, 4, 2) and xy.dtype == np.float32
    assert valid.shape == (20, 4) and valid.dtype == np.bool_

    # Hand-derived from the fixture encoding (x, y) = (step, track_id) and the
    # requested agent order (7, 3, 40, 99):
    #   column 0 -> track 7,  steps 0..19  -> valid everywhere, y == 7
    #   column 1 -> track 3,  steps 0..19  -> valid everywhere, y == 3
    #   column 2 -> track 40, steps 4..13  -> valid only at offsets 4..13, y == 40
    #   column 3 -> track 99, absent       -> never valid, all NaN
    expected_valid = np.zeros((20, 4), dtype=bool)
    expected_valid[:, 0] = True
    expected_valid[:, 1] = True
    expected_valid[4:14, 2] = True
    np.testing.assert_array_equal(valid, expected_valid)

    steps = np.arange(20, dtype=np.float32)
    np.testing.assert_array_equal(xy[:, 0], np.stack((steps, np.full(20, 7.0, np.float32)), axis=1))
    np.testing.assert_array_equal(xy[:, 1], np.stack((steps, np.full(20, 3.0, np.float32)), axis=1))
    np.testing.assert_array_equal(
        xy[4:14, 2],
        np.stack((np.arange(4, 14, dtype=np.float32), np.full(10, 40.0, np.float32)), axis=1),
    )
    assert np.isnan(xy[:4, 2]).all() and np.isnan(xy[14:, 2]).all()
    assert np.isnan(xy[:, 3]).all()


def test_materialize_rejects_a_complete_window_the_recording_cannot_supply(tmp_path):
    # A record that declares `actual_samples == total` promises the recording
    # holds every step of the window. If it does not -- e.g. the canonical file
    # was regenerated between enumeration and this read -- the window would come
    # back all-NaN (or partly zero-filled) and be trained on silently.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(20)})
    complete = ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path, 0, 7, 19, 1, (1,), 20,
    )
    xy, valid = materialize_forecast_window(complete)
    assert valid.all()
    assert np.isfinite(xy).all()

    # One agent, 20 declared steps, none of them on file -> 0 of 20.
    absent = replace(complete, target_agent_id=99, complete_agent_ids=(99,))
    with pytest.raises(AssertionError, match="0 of 20"):
        materialize_forecast_window(absent)

    # Same guard for a recording that supplies only part of the declared window:
    # steps 0..19 minus step 10 is 19 rows for the one agent -> 19 of 20.
    _write_window_fixture(path, {1: np.delete(np.arange(20), 10)})
    with pytest.raises(AssertionError, match="19 of 20"):
        materialize_forecast_window(complete)


def test_materialize_rejects_a_padded_window_missing_part_of_its_valid_prefix(tmp_path):
    # The partially padded case: the record declares 10 real samples followed by
    # 10 padded ones, but the recording lost step 5 from inside that prefix. The
    # zero-fill for padded steps would make the lost sample look like ordinary
    # Schoeller padding, so it has to raise. Hand-derived count: the prefix is
    # steps 0..9 for one agent = 10 slots, of which step 5 is missing -> 9 of 10.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.delete(np.arange(12), 5)})
    padded = ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path, 0, 7, 19, 1, (1,), 10,
    )
    with pytest.raises(AssertionError, match="9 of 10"):
        materialize_forecast_window(padded)

    # The guard must not fire on a legitimately padded window: with step 5 back
    # on file the prefix is complete and only the padded tail is invalid. (The
    # enumerated form of this window is covered by the first test above.)
    _write_window_fixture(path, {1: np.arange(10)})
    xy, valid = materialize_forecast_window(padded)
    assert valid[:10, 0].all()
    assert not valid[10:, 0].any()
    np.testing.assert_array_equal(xy[10:], np.zeros((10, 1, 2), dtype=np.float32))


def _generic_dataset(root: Path) -> Path:
    converted = root / "converted"
    converted.mkdir(parents=True)
    _write_window_fixture(converted / "recording.traj.parquet", {1: np.arange(20), 2: np.arange(20)})
    (root / "source.json").write_text(json.dumps({"canonical_recordings": ["recording"]}))
    prepare_splits(root, 42, strategy="grouped_scene_sequence")
    return root


def _task_with_stride(path: Path, stride: int) -> Path:
    task = json.loads((ROOT / "configs" / "task.json").read_text(encoding="utf-8"))
    task["canonical_stride"] = stride
    path.write_text(json.dumps(task), encoding="utf-8")
    return path


def test_generic_membership_requires_unit_canonical_stride(tmp_path):
    # enumerate_forecast_windows walks consecutive canonical steps, so it always
    # spans `total_steps` raw steps; the index path spans
    # `(total_steps - 1) * canonical_stride + 1`. The two agree only at stride 1,
    # and a mismatch would change which raw steps the model reads.
    root = _generic_dataset(tmp_path / "dataset")
    forecasting = resolve_forecasting_config(preset="social_gan")
    unit = Config(forecasting=forecasting, task_path=_task_with_stride(tmp_path / "stride1.json", 1))
    assert len(SceneDataset(DatasetSpec("test", root), "train", unit)) == 1

    strided = Config(forecasting=forecasting, task_path=_task_with_stride(tmp_path / "stride2.json", 2))
    with pytest.raises(AssertionError, match="canonical_stride"):
        SceneDataset(DatasetSpec("test", root), "train", strided)


class _FixedScaleRng:
    """Real generator for every draw except ``normal``, which returns ``value``."""

    def __init__(self, value: float) -> None:
        self._rng = np.random.default_rng(0)
        self._value = value

    def normal(self, *args, **kwargs) -> float:
        return self._value

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _straight_line_window() -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((20, 1, 2), dtype=np.float32)
    xy[:, 0, 0] = np.arange(20, dtype=np.float32)
    return xy, np.ones((20, 1), dtype=bool)


@pytest.mark.parametrize("draw", [0.0, -0.5])
def test_non_positive_random_scale_draw_raises_instead_of_collapsing_the_window(draw):
    config = resolve_forecasting_config(preset="social_vae")
    assert config.augmentation.random_scale.enabled
    xy, valid = _straight_line_window()
    with pytest.raises(ValueError, match="random_scale draw was non-positive"):
        prepare_forecast_representation(
            xy, valid, config, training=True, rng=_FixedScaleRng(draw)
        )


def test_default_random_scale_sampling_still_augments_normally():
    config = resolve_forecasting_config(preset="social_vae")
    xy, valid = _straight_line_window()
    representation = prepare_forecast_representation(
        xy, valid, config, training=True, rng=np.random.default_rng(0)
    )
    assert representation.augmentation["scale"] > 0
    assert np.isfinite(representation.xy).all()
    # mean 1.0, std 0.05: the draw stays within a few sigma of 1, so the window
    # keeps its extent instead of collapsing toward zero.
    assert abs(representation.augmentation["scale"] - 1.0) < 0.25
    assert np.ptp(representation.xy[:, 0, :], axis=0).max() > 10.0


def _social_gan_record(path: Path, start: int) -> ForecastWindowRecord:
    """Two-agent complete-scene window over steps ``start .. start + 19``."""
    return ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path,
        start, start + 7, start + 19, None, (1, 2), 20,
    )


def test_cache_serves_repeated_windows_with_the_values_independent_reads_give(tmp_path):
    # The cache must be invisible in the data: a second window taken from a
    # cached recording has to equal both the first one and what an independent
    # uncached read produces. Fixture encoding (x, y) = (step, track_id), so the
    # window starting at step 5 is hand-derivable: xy[k, agent] = (5 + k, agent + 1).
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(30), 2: np.arange(30)})
    record = _social_gan_record(path, 5)
    expected_xy = np.empty((20, 2, 2), dtype=np.float32)
    expected_xy[:, 0, 0] = expected_xy[:, 1, 0] = np.arange(5, 25, dtype=np.float32)
    expected_xy[:, 0, 1], expected_xy[:, 1, 1] = 1.0, 2.0

    cache = RecordingWindowCache()
    first_xy, first_valid = materialize_forecast_window(record, cache)
    second_xy, second_valid = materialize_forecast_window(record, cache)
    # Independent reads: a fresh cache per call reads the file again each time.
    uncached_xy, uncached_valid = materialize_forecast_window(record, RecordingWindowCache())

    for xy, valid in ((first_xy, first_valid), (second_xy, second_valid), (uncached_xy, uncached_valid)):
        assert xy.dtype == np.float32 and xy.shape == (20, 2, 2)
        assert valid.all()
        np.testing.assert_array_equal(xy, expected_xy)
        assert xy.tobytes() == expected_xy.tobytes()
    # The second call was served from memory rather than re-read: one read, and
    # the hit count covers both this record and the neighbouring window below.
    assert cache.stats()["reads"] == 1
    assert cache.stats()["hits"] == 1

    # A different window of the same recording is still sliced correctly.
    shifted_xy, shifted_valid = materialize_forecast_window(_social_gan_record(path, 9), cache)
    assert shifted_valid.all()
    np.testing.assert_array_equal(shifted_xy[:, 0, 0], np.arange(9, 29, dtype=np.float32))
    assert cache.stats()["reads"] == 1


def test_cache_rereads_a_recording_that_changed_on_disk(tmp_path):
    # A cache makes staleness durable, so the key carries (st_size, st_mtime_ns)
    # as well as the path -- the same pair analysis_pipeline uses for
    # canonical_source_revision. Rewriting the recording with y = track_id + 100
    # keeps the row count (and therefore very likely the file size) identical, so
    # this exercises the mtime half of the key; utime pins a distinct mtime
    # instead of relying on clock resolution.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(30), 2: np.arange(30)})
    record = _social_gan_record(path, 5)
    cache = RecordingWindowCache()
    before_xy, _ = materialize_forecast_window(record, cache)
    np.testing.assert_array_equal(before_xy[:, 0, 1], np.ones(20, dtype=np.float32))

    write_canonical_fixture(
        path,
        {
            track_id: [(int(step), float(step), float(track_id) + 100.0, np.nan) for step in np.arange(30)]
            for track_id in (1, 2)
        },
        dataset="test", scene_id="scene", scene_sequence="sequence", recording="recording",
    )
    os.utime(path, ns=(1_000_000_000, 1_500_000_000))

    after_xy, after_valid = materialize_forecast_window(record, cache)
    assert after_valid.all()
    assert cache.stats()["reads"] == 2
    # Hand-derived: y is now track_id + 100 for every step of the window.
    np.testing.assert_array_equal(after_xy[:, 0, 1], np.full(20, 101.0, dtype=np.float32))
    np.testing.assert_array_equal(after_xy[:, 1, 1], np.full(20, 102.0, dtype=np.float32))
    np.testing.assert_array_equal(after_xy[:, 0, 0], np.arange(5, 25, dtype=np.float32))


def test_cache_stays_within_its_byte_budget_and_still_serves_correct_windows(tmp_path):
    # Eviction must not corrupt anything: with a budget that fits one recording
    # but not two, alternating between them thrashes, and every window still has
    # to carry the values its own file holds.
    first, second = tmp_path / "a.traj.parquet", tmp_path / "b.traj.parquet"
    _write_window_fixture(first, {1: np.arange(30), 2: np.arange(30)}, recording="a")
    _write_window_fixture(second, {1: np.arange(30), 2: np.arange(30)}, recording="b")
    # 30 steps x 2 tracks x 4 columns x 4 bytes = 960 bytes per recording.
    cache = RecordingWindowCache(budget_bytes=960)
    for _ in range(3):
        for path in (first, second):
            xy, valid = materialize_forecast_window(_social_gan_record(path, 5), cache)
            assert valid.all()
            np.testing.assert_array_equal(xy[:, 1, 0], np.arange(5, 25, dtype=np.float32))
            assert cache.stats()["entries"] == 1
            assert cache.stats()["cached_bytes"] == 960 <= cache.stats()["budget_bytes"]
    assert cache.stats() == {"reads": 6, "hits": 0, "entries": 1, "cached_bytes": 960, "budget_bytes": 960}

    # A recording larger than the whole budget is served but never retained, so
    # the bound holds instead of being exceeded by one oversized entry.
    tiny = RecordingWindowCache(budget_bytes=100)
    xy, valid = materialize_forecast_window(_social_gan_record(first, 5), tiny)
    assert valid.all()
    np.testing.assert_array_equal(xy[:, 0, 0], np.arange(5, 25, dtype=np.float32))
    assert tiny.stats()["entries"] == 0 and tiny.stats()["cached_bytes"] == 0


def test_cached_recording_columns_are_read_only(tmp_path):
    # Windows are slices of the cached columns, so a downstream write would
    # corrupt every later sample of that recording rather than one window.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(30), 2: np.arange(30)})
    cache = RecordingWindowCache()
    track_ids, steps, xs, ys = cache.window(path, 5, 24)
    assert len(track_ids) == len(steps) == len(xs) == len(ys) == 40
    for column in (track_ids, steps, xs, ys):
        with pytest.raises(ValueError):
            column[0] = 0
