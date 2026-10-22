"""Behavioural tests for the dataset-analysis modules that feed the paper.

Covers A2-003 for the five analysis entry points that had no behavioural coverage:
``sizing.write_sizing_table``, ``data_quality.speed_summary``,
``crowd_density.crowd_density``, ``diversity._scene_features`` and
``coverage.recording_features``.

Every expected value below is hand-derived from the published contract of the
quantity (the LaTeX column layout, the definition of speed as distance over
elapsed time, the forecasting window definition in ``configs/task.json``, and
the textbook definitions of path straightness and turning angle) rather than by
re-running the implementation.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from canonical import CANONICAL_DT, CANONICAL_HZ
from conftest import write_canonical_fixture


# --------------------------------------------------------------------------
# sizing.write_sizing_table
# --------------------------------------------------------------------------

# The header row of the generated table names each column; this mapping is the
# specification of which ``*_sizing.json`` field belongs under which column.
# Asserting against it (by key, not by position) is what catches a transposed
# pair in the positional ``stats[...]`` tuple inside ``write_sizing_table``.
SIZING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Dataset", "dataset"),
    ("Hours", "hours"),
    ("Raw tracks", "T_d_raw"),
    ("Post-cat. tracks", "T_d_post_categorization"),
    (r"$L_{med}$", "L_med"),
    ("$s_d$", "s_d"),
    ("$w_d$", "w_d"),
    ("$p_d$", "p_d"),
    ("Train", "final_train_scenes"),
    ("Val", "final_val_scenes"),
    ("Test", "final_test_scenes"),
)


def sizing_record(**overrides) -> dict:
    """A full sizing record whose values are pairwise distinct.

    Distinct values matter: with duplicated numbers a swapped column pair would
    still compare equal and the mapping assertion would not catch it.
    """
    record = {
        "dataset": "alpha",
        "hours": 11.5,
        "T_d_raw": 21,
        "T_d_post_categorization": 31,
        "L_med": 41.5,
        "s_d": 51,
        "w_d": 61.25,
        "p_d": 0.25,
        "final_train_scenes": 71,
        "final_val_scenes": 81,
        "final_test_scenes": 91,
    }
    record.update(overrides)
    return record


def parse_sizing_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Split a ``write_sizing_table`` document into (header fields, data rows)."""
    lines = text.splitlines()
    assert lines[0] == r"\begin{tabular}{lrrrrrrrrrr}", lines[0]
    assert lines[-1] == r"\end{tabular}", lines[-1]
    assert lines[2] == r"\hline", lines[2]
    header = [field.strip() for field in lines[1].removesuffix(r"\\").split("&")]
    rows = [
        [field.strip() for field in line.removesuffix(r"\\").split("&")]
        for line in lines[3:-1]
    ]
    return header, rows


def test_sizing_table_rows_have_exactly_the_declared_number_of_columns(tmp_path: Path) -> None:
    """``\\begin{tabular}{lrrrrrrrrrr}`` declares 11 columns (1 ``l`` + 10 ``r``).

    LaTeX requires every row of the tabular to supply that many ``&``-separated
    fields; a row with a different count is a malformed table. The expectation
    therefore comes from the LaTeX column specification, not from the code.
    """
    from sizing import write_sizing_table

    (tmp_path / "a_sizing.json").write_text(json.dumps(sizing_record(dataset="alpha")))
    (tmp_path / "b_sizing.json").write_text(json.dumps(sizing_record(dataset="bravo")))

    output = write_sizing_table(tmp_path, tmp_path / "out" / "sizing.tex")
    header, rows = parse_sizing_table(output.read_text(encoding="utf-8"))

    column_specification = r"lrrrrrrrrrr"
    assert len(column_specification) == 11
    assert len(header) == 11, f"header declares {len(header)} fields: {header}"
    assert len(rows) == 2
    for row in rows:
        assert len(row) == len(column_specification), (
            f"row has {len(row)} fields but the tabular declares "
            f"{len(column_specification)} columns: {row}"
        )


def test_sizing_table_places_each_json_field_under_its_named_column(tmp_path: Path) -> None:
    """Each emitted field must equal the input value for the key that the header names."""
    from sizing import write_sizing_table

    record = sizing_record(dataset="alpha")
    (tmp_path / "a_sizing.json").write_text(json.dumps(record))

    output = write_sizing_table(tmp_path, tmp_path / "sizing.tex")
    header, rows = parse_sizing_table(output.read_text(encoding="utf-8"))

    assert header == [label for label, _ in SIZING_COLUMNS]
    (row,) = rows
    for (label, key), field in zip(SIZING_COLUMNS, row):
        # Columns the writer formats; every other field is emitted via str().
        # Keep in step with src/sizing.py when a column's precision changes.
        formats = {"p_d": ".4f", "hours": ".3f"}
        expected = format(record[key], formats[key]) if key in formats else str(record[key])
        assert field == expected, (
            f"column {label!r} holds {field!r} but its source field {key!r} is {record[key]!r}"
        )


def test_sizing_table_restricts_rows_to_the_requested_datasets(tmp_path: Path) -> None:
    from sizing import write_sizing_table

    (tmp_path / "a_sizing.json").write_text(json.dumps(sizing_record(dataset="alpha")))
    (tmp_path / "b_sizing.json").write_text(json.dumps(sizing_record(dataset="bravo")))

    output = write_sizing_table(tmp_path, tmp_path / "sizing.tex", datasets={"bravo"})
    text = output.read_text(encoding="utf-8")
    _, rows = parse_sizing_table(text)

    assert [row[0] for row in rows] == ["bravo"]
    assert "alpha" not in text


def test_sizing_table_marks_a_missing_hours_field_rather_than_failing(tmp_path: Path) -> None:
    """``hours`` is the only optional field; an unmeasured duration prints as ``--``."""
    from sizing import write_sizing_table

    record = sizing_record(dataset="alpha")
    record.pop("hours")
    (tmp_path / "a_sizing.json").write_text(json.dumps(record))

    output = write_sizing_table(tmp_path, tmp_path / "sizing.tex")
    header, (row,) = parse_sizing_table(output.read_text(encoding="utf-8"))

    assert row[header.index("Hours")] == "--"
    # Every other column is unaffected by the missing optional field.
    assert row[header.index("Dataset")] == "alpha"
    assert row[header.index("Raw tracks")] == str(record["T_d_raw"])


@pytest.mark.parametrize("p_d,expected", [(0.25, "0.2500"), (0.123456, "0.1235"), (1.0, "1.0000")])
def test_sizing_table_prints_retention_probability_with_four_decimals(
    tmp_path: Path, p_d: float, expected: str
) -> None:
    """``p_d`` is a retention probability; the table reports it to 4 decimal places."""
    from sizing import write_sizing_table

    (tmp_path / "a_sizing.json").write_text(json.dumps(sizing_record(p_d=p_d)))

    output = write_sizing_table(tmp_path, tmp_path / "sizing.tex")
    header, (row,) = parse_sizing_table(output.read_text(encoding="utf-8"))

    field = row[header.index("$p_d$")]
    assert field == expected
    assert re.fullmatch(r"\d+\.\d{4}", field), field


def test_sizing_table_row_order_is_deterministic_and_follows_the_filenames(tmp_path: Path) -> None:
    """A published table must not depend on directory enumeration order.

    Two directories holding the same records created in opposite orders must
    produce byte-identical output, ordered by the sizing-file name.
    """
    from sizing import write_sizing_table

    records = {"a_sizing.json": sizing_record(dataset="zulu"), "b_sizing.json": sizing_record(dataset="alpha")}
    forward, reverse = tmp_path / "forward", tmp_path / "reverse"
    for directory, names in ((forward, list(records)), (reverse, list(records)[::-1])):
        directory.mkdir()
        for name in names:
            (directory / name).write_text(json.dumps(records[name]))

    first = write_sizing_table(forward, forward / "sizing.tex").read_text(encoding="utf-8")
    second = write_sizing_table(reverse, reverse / "sizing.tex").read_text(encoding="utf-8")

    assert first == second
    _, rows = parse_sizing_table(first)
    assert [row[0] for row in rows] == ["zulu", "alpha"], "rows follow sorted filenames, not dataset names"


# --------------------------------------------------------------------------
# data_quality.speed_summary
# --------------------------------------------------------------------------

def write_speed_fixture(root: Path, tracks: dict[int, list[tuple[int, float, float]]]) -> Path:
    converted = root / "converted"
    write_canonical_fixture(
        converted / "recording.traj.parquet",
        {tid: [(step, x, y, None) for step, x, y in rows] for tid, rows in tracks.items()},
        dataset="synthetic", scene_id="scene", scene_sequence="sequence", recording="recording",
    )
    return root


def test_speed_summary_reports_distance_over_elapsed_time(tmp_path: Path) -> None:
    """Speed is displacement divided by elapsed time; both are known exactly here.

    With ``canonical_stride = 2`` at ``storage_hz = 2.5`` the sampled steps are
    0.8 s apart. Each retained pair moves exactly 1.0 m, so the hand-computed
    speed is ``1.0 m / 0.8 s = 1.25 m/s`` for every pair, hence mean = median =
    p999 = 1.25 m/s.
    """
    from data_quality import speed_summary

    storage_hz, stride = 2.5, 2
    # Odd steps sit 1 km away: if the stride filter failed to drop them the
    # reported speeds would be enormous instead of 1.25 m/s.
    root = write_speed_fixture(tmp_path, {
        1: [(0, 0.0, 0.0), (1, 1000.0, 0.0), (2, 1.0, 0.0), (3, 1000.0, 0.0), (4, 2.0, 0.0)],
    })

    result = speed_summary(root, storage_hz, stride)

    expected = 1.0 / (stride / storage_hz)
    assert expected == 1.25
    assert result["pairs"] == 2, "steps 1 and 3 are not multiples of the stride and must be dropped"
    assert result["mean_mps"] == pytest.approx(expected, rel=1e-12)
    assert result["median_mps"] == pytest.approx(expected, rel=1e-12)
    assert result["p999_mps"] == pytest.approx(expected, rel=1e-12)


def test_speed_summary_never_pairs_samples_from_different_tracks(tmp_path: Path) -> None:
    """A pair must belong to one track; the A-to-B transition is not a displacement.

    Track A ends at x = 1 m and track B starts at x = 100 m. Two in-track pairs
    of 1.0 m over 0.8 s exist, so mean = 1.25 m/s; pairing A's last sample with
    B's first would add a spurious 99 m displacement.
    """
    from data_quality import speed_summary

    storage_hz, stride = 2.5, 2
    root = write_speed_fixture(tmp_path, {
        1: [(0, 0.0, 0.0), (2, 1.0, 0.0)],
        2: [(4, 100.0, 0.0), (6, 101.0, 0.0)],
    })

    result = speed_summary(root, storage_hz, stride)

    assert result["pairs"] == 2
    assert result["mean_mps"] == pytest.approx(1.25, rel=1e-12)


def test_speed_summary_skips_pairs_separated_by_a_gap(tmp_path: Path) -> None:
    """A missing sample means the elapsed time is unknown for that interval."""
    from data_quality import speed_summary

    storage_hz, stride = 2.5, 2
    root = write_speed_fixture(tmp_path, {
        1: [(0, 0.0, 0.0), (4, 50.0, 0.0)],  # stride-aligned but 2 strides apart
    })

    result = speed_summary(root, storage_hz, stride)

    assert result["pairs"] == 0
    assert math.isnan(result["mean_mps"])


def test_speed_summary_on_an_empty_dataset_reports_no_pairs_and_nan_statistics(tmp_path: Path) -> None:
    """Documented empty-input contract: zero pairs and undefined (NaN) statistics."""
    from data_quality import speed_summary

    (tmp_path / "converted").mkdir()

    result = speed_summary(tmp_path, CANONICAL_HZ, 1)

    assert result["pairs"] == 0
    for key in ("mean_mps", "median_mps", "p999_mps"):
        assert math.isnan(result[key]), f"{key} should be NaN with no pairs, got {result[key]!r}"


def test_speed_summary_reports_true_pair_count_when_subsampled(tmp_path: Path) -> None:
    """``pairs`` must be the true in-track count, not the post-subsample count (Q-009).

    Ten in-track pairs exist (steps 0, 2, ..., 20 on one track at stride 2), but
    ``max_pairs=3`` forces subsampling. The artifact must still report the true
    ``pairs == 10`` alongside ``sampled_pairs == 3`` and ``subsampled is True``,
    since ``p999_mps`` is then only a point estimate from that subsample.
    """
    from data_quality import speed_summary

    storage_hz, stride = 2.5, 2
    root = write_speed_fixture(tmp_path, {
        1: [(step, float(step), 0.0) for step in range(0, 22, 2)],  # 11 samples -> 10 in-track pairs
    })

    result = speed_summary(root, storage_hz, stride, max_pairs=3)

    assert result["pairs"] == 10, "pairs must be the true in-track count, not the subsampled count"
    assert result["sampled_pairs"] == 3
    assert result["subsampled"] is True


def test_speed_summary_subsampled_flag_is_false_when_no_subsampling_occurs(tmp_path: Path) -> None:
    """The ``subsampled`` flag must not be trivially always-true."""
    from data_quality import speed_summary

    storage_hz, stride = 2.5, 2
    root = write_speed_fixture(tmp_path, {
        1: [(0, 0.0, 0.0), (2, 1.0, 0.0)],
    })

    result = speed_summary(root, storage_hz, stride, max_pairs=1_000_000)

    assert result["subsampled"] is False
    assert result["pairs"] == result["sampled_pairs"] == 1


def test_main_raises_on_empty_converted_dir_instead_of_writing_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main()`` must not write bare ``NaN`` (invalid JSON, RFC 8259) on empty input (Q-009)."""
    import data_quality

    dataset_dir = tmp_path / "emptyds"
    (dataset_dir / "converted").mkdir(parents=True)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        ["data_quality", str(dataset_dir), "--output-dir", str(output_dir)],
    )

    with pytest.raises(ValueError):
        data_quality.main()

    written = list(output_dir.glob("*.json")) if output_dir.exists() else []
    assert written == [], "main() must not write any output file (valid or otherwise) on zero pairs"


# --------------------------------------------------------------------------
# crowd_density.crowd_density and diversity._scene_features
# --------------------------------------------------------------------------

# The forecasting window of the default preset (configs/task.json -> steps_default)
# observes ``OBSERVATION_STEPS`` frames starting at ``start_step`` with stride 1,
# so the final *observed* absolute step of a window is
# ``start_step + OBSERVATION_STEPS - 1``.
OBSERVATION_STEPS = 8
PREDICTION_STEPS = 12


@pytest.fixture(scope="module")
def crowd_dataset_dir(tmp_path_factory) -> Path:
    """A two-window synthetic dataset with agents placed around the observation boundary.

    Primary track 1 spans steps 0..20 (long enough for windows starting at 0 and 1).
    Tracks 2 and 3 exist only at step 7; track 4 exists only at step 8.

    Hand-derived expectation, from "agents present at the final observed frame":
      * window start 0 -> final observed step 7 -> agents {1, 2, 3} -> 3
      * window start 1 -> final observed step 8 -> agents {1, 4}    -> 2
    """
    root = tmp_path_factory.mktemp("crowd") / "synthetic"
    total_steps = OBSERVATION_STEPS + PREDICTION_STEPS
    tracks = {
        1: [(step, 0.5 * step, 0.0, None) for step in range(total_steps + 1)],
        2: [(7, 3.5, 1.0, None)],
        3: [(7, 3.5, 2.0, None)],
        4: [(8, 4.0, 1.0, None)],
    }
    write_canonical_fixture(
        root / "converted" / "recording.traj.parquet", tracks,
        dataset="synthetic", scene_id="scene", scene_sequence="sequence", recording="recording",
    )

    from preprocess import INDEX_SCHEMA

    rows = [
        {
            "dataset": "synthetic", "scene_id": "scene", "scene_sequence": "sequence",
            "recording": "recording", "start_step": start,
            "observation_end_step": start + OBSERVATION_STEPS - 1,
            "future_end_step": start + total_steps - 1,
            "target_agent_id": 1, "primary_track_id": 1, "split": "train",
            "scene_tag": 1, "sub_tags": [], "accepted_if_train": True, "accepted": True,
            "track_kept": True, "eval_kept": True,
        }
        for start in (0, 1)
    ]
    index_dir = root / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA),
                   index_dir / "scenes-seed42.parquet")
    return root


def test_crowd_density_counts_agents_at_the_final_observed_frame(crowd_dataset_dir: Path) -> None:
    """Off-by-one guard for the observation boundary.

    Counting one frame late would report 2 and 1 agents (mean 1.5) instead of
    the hand-derived 3 and 2 (mean 2.5); counting one frame early would report
    1 and 3.
    """
    from crowd_density import crowd_density

    result = crowd_density(crowd_dataset_dir, "train", seed=42)

    assert result["scenes"] == 2
    assert result["max"] == 3, "the two step-7 neighbours must be counted for the window at start 0"
    assert result["min"] == 2, "the step-8 neighbour must not be counted for the window at start 0"
    assert result["mean"] == pytest.approx(2.5)
    assert result["std"] == pytest.approx(0.5)
    assert result["definition"] == "agents per primary-agent window present at the final observed frame"


def test_crowd_density_and_diversity_agree_on_the_same_scene(crowd_dataset_dir: Path) -> None:
    """``diversity._scene_features`` recomputes the same density; they must match.

    Both are compared against the independently hand-counted per-window values
    (3 and 2), so agreement between two wrong implementations cannot pass.
    """
    from config import Config, DatasetSpec, SeedConfig
    from data import SceneDataset
    from diversity import _scene_features

    cfg = Config(seeds=SeedConfig(42, 42), train_rotation=False)
    dataset = SceneDataset(DatasetSpec(crowd_dataset_dir.name, crowd_dataset_dir), "train", cfg)
    assert dataset.history_steps == OBSERVATION_STEPS

    by_start = {
        dataset.rows[index]["start_step"]: _scene_features(dataset[index], dataset.history_steps)[0]
        for index in range(len(dataset))
    }

    assert by_start == {0: 3, 1: 2}


def test_window_agent_counts_match_per_scene_dataset_access(crowd_dataset_dir: Path) -> None:
    """A2-009: the batched per-recording counter is an exact no-op on the result.

    ``crowd_density`` used to build a full ``Scene`` per window just to read
    ``valid[history_steps - 1].sum()``. The batched replacement reads each
    recording once; because ``Scene.valid`` is ``_scene_arrays(...)["valid"]``
    verbatim, the two must agree elementwise as integers, in ``dataset.rows``
    order -- not merely in aggregate.
    """
    from config import Config, DatasetSpec, SeedConfig
    from crowd_density import window_agent_counts
    from data import SceneDataset

    cfg = Config(seeds=SeedConfig(42, 42), train_rotation=False)
    dataset = SceneDataset(DatasetSpec(crowd_dataset_dir.name, crowd_dataset_dir), "train", cfg)
    observation_index = dataset.history_steps - 1

    per_scene = np.asarray(
        [int(dataset[index].valid[observation_index].sum()) for index in range(len(dataset))]
    )
    batched = window_agent_counts(dataset)

    assert batched.dtype.kind == "i", "counts must stay integral, not float"
    assert np.array_equal(batched, per_scene)
    # Independently hand-counted expectation of the fixture (see ``crowd_dataset_dir``),
    # ordered by the index rows, so two matching wrong implementations cannot pass.
    starts = [row["start_step"] for row in dataset.rows]
    assert dict(zip(starts, batched.tolist())) == {0: 3, 1: 2}


def test_window_agent_counts_preserve_row_order_across_recordings(tmp_path: Path) -> None:
    """Batching by recording must not reorder the returned counts.

    The index rows below interleave two recordings that have different crowd
    sizes at the final observed frame, so a result grouped by recording instead
    of by row would differ from the ``dataset.rows`` order.
    """
    from config import Config, DatasetSpec, SeedConfig
    from crowd_density import window_agent_counts
    from data import SceneDataset
    from preprocess import INDEX_SCHEMA

    root = tmp_path / "interleaved"
    total_steps = OBSERVATION_STEPS + PREDICTION_STEPS
    # ``rec_a``: one extra agent at the final observed frame of the window at
    # start 0 (step 7); ``rec_b``: two extra agents there. Hand-derived counts
    # are therefore 2 for every ``rec_a`` window and 3 for every ``rec_b`` one.
    neighbours = {"rec_a": [2], "rec_b": [2, 3]}
    for recording, extra in neighbours.items():
        tracks = {1: [(step, 0.5 * step, 0.0, None) for step in range(total_steps + 1)]}
        for offset, track_id in enumerate(extra):
            tracks[track_id] = [(OBSERVATION_STEPS - 1, 3.5, 1.0 + offset, None)]
        write_canonical_fixture(
            root / "converted" / f"{recording}.traj.parquet", tracks,
            dataset="interleaved", scene_id="scene", scene_sequence=recording, recording=recording,
        )

    order = [("rec_b", 0), ("rec_a", 0), ("rec_b", 1), ("rec_a", 1)]
    rows = [
        {
            "dataset": "interleaved", "scene_id": "scene", "scene_sequence": recording,
            "recording": recording, "start_step": start,
            "observation_end_step": start + OBSERVATION_STEPS - 1,
            "future_end_step": start + total_steps - 1,
            "target_agent_id": 1, "primary_track_id": 1, "split": "train",
            "scene_tag": 1, "sub_tags": [], "accepted_if_train": True, "accepted": True,
            "track_kept": True, "eval_kept": True,
        }
        for recording, start in order
    ]
    (root / "indices").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA),
                   root / "indices" / "scenes-seed42.parquet")

    cfg = Config(seeds=SeedConfig(42, 42), train_rotation=False)
    dataset = SceneDataset(DatasetSpec(root.name, root), "train", cfg)
    observation_index = dataset.history_steps - 1

    per_scene = np.asarray(
        [int(dataset[index].valid[observation_index].sum()) for index in range(len(dataset))]
    )
    batched = window_agent_counts(dataset)

    assert [row["recording"] for row in dataset.rows] == [recording for recording, _ in order]
    # Window start 1 ends at step 8, past the single-frame neighbours, so only
    # the primary remains: 3, 2, 1, 1 in the interleaved row order above.
    assert batched.tolist() == [3, 2, 1, 1]
    assert np.array_equal(batched, per_scene)


def test_window_agent_counts_rejects_unsorted_recording_steps(
    crowd_dataset_dir: Path, tmp_path: Path
) -> None:
    """The batched window cut uses ``np.searchsorted`` on file-order ``step``.

    An out-of-order recording must fail loudly with the offending path rather
    than silently slicing the wrong rows. Two independent layers enforce this:
    ``validate_canonical_file`` rejects such a recording at load time, and
    ``window_agent_counts`` keeps its own guard for a recording that is
    corrupted after the dataset was built. Both are asserted here.
    """
    from config import Config, DatasetSpec, SeedConfig
    from crowd_density import window_agent_counts
    from data import SceneDataset

    # Copy the module-scoped fixture before corrupting it.
    unsorted_dir = tmp_path / crowd_dataset_dir.name
    shutil.copytree(crowd_dataset_dir, unsorted_dir)
    path = unsorted_dir / "converted" / "recording.traj.parquet"
    table = pq.read_table(path)
    descending = np.argsort(-table["step"].combine_chunks().to_numpy(), kind="stable")

    cfg = Config(seeds=SeedConfig(42, 42), train_rotation=False)
    spec = DatasetSpec(unsorted_dir.name, unsorted_dir)

    # Build the dataset while the recording is still sorted, then corrupt it on
    # disk so the batched cut -- not canonical validation -- is what rejects it.
    dataset = SceneDataset(spec, "train", cfg)
    pq.write_table(table.take(descending), path)
    with pytest.raises(ValueError, match="out of order"):
        window_agent_counts(dataset)

    # And canonical validation refuses the same recording up front.
    with pytest.raises(ValueError, match="out of order"):
        SceneDataset(spec, "train", cfg)


def test_window_agent_counts_honour_generic_scene_membership(tmp_path: Path) -> None:
    """Generic membership must stay membership-filtered, not a raw recording count.

    Under the ``social_gan`` preset (``social_gan_complete_scene`` membership) a
    scene contains only agents observed across the whole window, so the
    single-frame neighbour sitting exactly on the observation boundary is *not*
    a member. The batched per-recording reader cannot express that filter -- it
    would see the neighbour in the raw recording and report 3 -- so this path
    must fall back to the per-scene ``dataset[i]`` access.
    """
    from config import Config, DatasetSpec, SeedConfig
    from crowd_density import window_agent_counts
    from data import SceneDataset
    from forecasting_config import resolve_forecasting_config

    root = tmp_path / "generic"
    total_steps = OBSERVATION_STEPS + PREDICTION_STEPS
    # Tracks 1 and 2 span the whole recording and so are complete-scene members
    # of every window; track 3 exists only at step 7, the final observed frame
    # of the window starting at 0.
    tracks = {
        1: [(step, 0.5 * step, 0.0, None) for step in range(total_steps + 2)],
        2: [(step, 0.5 * step, 2.0, None) for step in range(total_steps + 2)],
        3: [(OBSERVATION_STEPS - 1, 3.5, 3.0, None)],
    }
    write_canonical_fixture(
        root / "converted" / "recording.traj.parquet", tracks,
        dataset="generic", scene_id="scene", scene_sequence="sequence", recording="recording",
    )

    cfg = Config(
        seeds=SeedConfig(42, 42),
        train_rotation=False,
        forecasting=resolve_forecasting_config(preset="social_gan"),
    )
    dataset = SceneDataset(DatasetSpec(root.name, root), "train", cfg)
    assert dataset.generic_membership, "the social_gan preset must not use target_complete membership"
    observation_index = dataset.history_steps - 1

    per_scene = np.asarray(
        [int(dataset[index].valid[observation_index].sum()) for index in range(len(dataset))]
    )
    counts = window_agent_counts(dataset)

    assert counts.dtype.kind == "i", "counts must stay integral, not float"
    assert np.array_equal(counts, per_scene)
    # Hand-derived: only tracks 1 and 2 are complete-scene members, so every
    # window holds 2 agents at its final observed frame. Counting the raw
    # recording instead would report 3 for the window starting at 0.
    assert counts.tolist() == [2] * len(dataset)
    boundary = [index for index, row in enumerate(dataset.rows) if row["start_step"] == 0]
    assert boundary, "the fixture must contain a window starting at step 0"
    assert counts[boundary[0]] == 2, "the single-frame non-member neighbour must not be counted"


# --------------------------------------------------------------------------
# diversity._scene_features (pure)
# --------------------------------------------------------------------------

def stub_scene(primary: list[tuple[float, float]], valid_final: list[bool] | None = None):
    """Minimal stand-in exposing the two attributes ``_scene_features`` reads."""
    steps = len(primary)
    agents = len(valid_final) if valid_final is not None else 1
    xy = np.zeros((steps, agents, 2), dtype=np.float64)
    xy[:, 0] = np.asarray(primary, dtype=np.float64)
    valid = np.ones((steps, agents), dtype=bool)
    if valid_final is not None:
        valid[OBSERVATION_STEPS - 1] = np.asarray(valid_final, dtype=bool)
    return SimpleNamespace(xy=xy, valid=valid)


def test_scene_features_mean_speed_is_displacement_per_canonical_timestep() -> None:
    """0.5 m every canonical step of 0.4 s is 1.25 m/s, constant along the track."""
    assert CANONICAL_DT == pytest.approx(0.4)
    from diversity import _scene_features

    track = [(0.5 * step, 0.0) for step in range(OBSERVATION_STEPS)]
    _, speed, turning = _scene_features(stub_scene(track), OBSERVATION_STEPS)

    assert speed == pytest.approx(0.5 / CANONICAL_DT, rel=1e-12)
    assert speed == pytest.approx(1.25, rel=1e-12)
    assert turning == pytest.approx(0.0, abs=1e-12), "a straight track has zero total turning"


def test_scene_features_total_turning_of_a_right_angle_is_half_pi() -> None:
    """Heading goes from +x to +y exactly once: total turning is pi/2 radians."""
    from diversity import _scene_features

    track = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)]
    _, _, turning = _scene_features(stub_scene(track), len(track))

    assert turning == pytest.approx(math.pi / 2, rel=1e-12)


def test_scene_features_density_counts_only_agents_valid_at_the_observation_boundary() -> None:
    from diversity import _scene_features

    track = [(0.5 * step, 0.0) for step in range(OBSERVATION_STEPS + PREDICTION_STEPS)]
    scene = stub_scene(track, valid_final=[True, False, True, True])

    density, _, _ = _scene_features(scene, OBSERVATION_STEPS)

    assert density == 3


# --------------------------------------------------------------------------
# coverage.recording_features
# --------------------------------------------------------------------------

def test_recording_features_of_straight_parallel_tracks(tmp_path: Path) -> None:
    """Hand-computable per-track features for three straight, constant-speed tracks.

    Tracks 1 and 2 run parallel 1.0 m apart (inside the 5 m neighbour radius);
    track 3 runs 20 m away (outside it). Each track advances 0.5 m per stored
    step at ``storage_hz = 2.5`` (0.4 s), so speed = 1.25 m/s throughout.
    Straightness is |end - start| / path length = 2.0 / 2.0 = 1 for a straight
    path, total turning is 0, and no sample is slower than 0.2 m/s.
    """
    from coverage import recording_features

    storage_hz = 2.5
    steps = range(5)
    path = tmp_path / "recording.traj.parquet"
    write_canonical_fixture(path, {
        1: [(step, 0.5 * step, 0.0, None) for step in steps],
        2: [(step, 0.5 * step, 1.0, None) for step in steps],
        3: [(step, 0.5 * step, 20.0, None) for step in steps],
    }, dataset="synthetic", scene_id="scene", scene_sequence="sequence", recording="recording")

    features, times, span = recording_features(path, storage_hz)

    assert features.shape == (3, 6)
    mean_speed, p90_speed, mean_neighbors, straightness, turning, slow_fraction = features[0]
    assert mean_speed == pytest.approx(0.5 / (1 / storage_hz), rel=1e-6)
    assert mean_speed == pytest.approx(1.25, rel=1e-6)
    assert p90_speed == pytest.approx(1.25, rel=1e-6)
    assert mean_neighbors == pytest.approx(1.0), "track 2 is 1 m away, track 3 is 20 m away"
    assert straightness == pytest.approx(1.0, rel=1e-6), "a straight path has straightness 1"
    assert turning == pytest.approx(0.0, abs=1e-6)
    assert slow_fraction == pytest.approx(0.0)
    assert features[2][2] == pytest.approx(0.0), "the distant track has no neighbour within 5 m"

    # Track midpoint offset from the recording start, in hours.
    assert times == pytest.approx(np.full(3, (2 - 0) / storage_hz / 3600), rel=1e-12)
    # Recording span: (max_step - min_step) / storage_hz seconds, in hours.
    assert span == pytest.approx((4 - 0) / storage_hz / 3600, rel=1e-12)


def test_recording_features_ignores_tracks_shorter_than_four_samples(tmp_path: Path) -> None:
    """Per-track statistics need several displacements; very short tracks are skipped."""
    from coverage import recording_features

    path = tmp_path / "recording.traj.parquet"
    write_canonical_fixture(path, {
        1: [(step, 0.5 * step, 0.0, None) for step in range(4)],
        2: [(step, 0.5 * step, 1.0, None) for step in range(3)],
    }, dataset="synthetic", scene_id="scene", scene_sequence="sequence", recording="recording")

    features, times, _ = recording_features(path, CANONICAL_HZ)

    assert features.shape == (1, 6)
    assert len(times) == 1


def test_recording_features_neighbour_count_is_inclusive_at_the_5m_radius_boundary(
    tmp_path: Path,
) -> None:
    """``recording_features``'s neighbour count treats the 5 m radius as inclusive.

    ``recording_features`` counts neighbours via
    ``cKDTree.query_ball_point(xy, 5.0, return_length=True) - 1``. A direct check of scipy's
    contract (independent of this test file) shows a point placed at exactly distance 5.0 is
    returned by ``query_ball_point``, while one at ``5.0 + 1e-9`` is not: the ball query is a
    ``distance <= r`` test, not a strict ``<``.

    Three pairs of mutually stationary tracks are placed far apart (>> 5 m between pairs, so
    only within-pair distances can put two tracks within radius of each other):

    - pair A sits at x = 0.0 and x = 5.0 on the same axis, an exact float subtraction giving a
      distance of exactly 5.0 m -- it must count as a neighbour pair;
    - pair B is 5.001 m apart (just outside the radius) -- it must not;
    - pair C is 4.999 m apart (just inside the radius) -- it must.

    Because every track is stationary, the ball query returns the same count at every one of
    its steps, so ``mean_neighbors`` (the mean over steps) equals that constant count exactly,
    with no floating-point accumulation to account for.
    """
    from coverage import recording_features

    steps = range(4)

    def stationary(x: float, y: float) -> list[tuple[int, float, float, float | None]]:
        return [(step, x, y, None) for step in steps]

    path = tmp_path / "recording.traj.parquet"
    write_canonical_fixture(path, {
        1: stationary(0.0, 0.0),             # pair A: origin
        2: stationary(5.0, 0.0),             # pair A: exactly 5.0 m away -> inside (boundary)
        3: stationary(100.0, 0.0),           # pair B
        4: stationary(100.0 + 5.001, 0.0),   # pair B: 5.001 m away -> just outside
        5: stationary(200.0, 0.0),           # pair C
        6: stationary(200.0 + 4.999, 0.0),   # pair C: 4.999 m away -> just inside
    }, dataset="synthetic", scene_id="scene", scene_sequence="sequence", recording="recording")

    features, _, _ = recording_features(path, CANONICAL_HZ)

    assert features.shape == (6, 6)
    mean_neighbors = features[:, 2]
    assert mean_neighbors == pytest.approx([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]), (
        "pair A (exactly 5.0 m) and pair C (4.999 m) must count each other as a neighbour; "
        "pair B (5.001 m) must not"
    )


def test_scipy_query_ball_point_return_length_matches_len_of_materialised_lists() -> None:
    """Precondition check on scipy itself -- NOT on this repository's ``coverage`` module.

    Confirms that ``cKDTree.query_ball_point(..., return_length=True)`` agrees exactly with
    ``len(query_ball_point(...))`` for a point cloud engineered to exercise exact duplicates
    (distance 0), separations of exactly 5.0 m produced both axis-aligned and via 3-4-5 right
    triangles (whose squared distance 9 + 16 = 25 is exact in binary floating point), points at
    5.0 +/- 1e-12, and the adjacent representable floats either side of 5.0.

    This test never imports ``coverage`` and does not exercise ``recording_features``'s
    neighbour-count column at the boundary; see
    ``test_recording_features_neighbour_count_is_inclusive_at_the_5m_radius_boundary`` above for
    that.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(20260911)
    points = [(0.0, 0.0)] * 5  # exact duplicates at the origin
    points += [(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)]  # exactly 5 m, axis-aligned
    points += [(3.0, 4.0), (4.0, 3.0), (-3.0, -4.0), (-4.0, 3.0)]  # exactly 5 m, 3-4-5 triples
    for distance in (5.0 - 1e-12, 5.0 + 1e-12, np.nextafter(5.0, 0.0), np.nextafter(5.0, np.inf)):
        points += [(distance, 0.0), (0.0, -distance)]
    for centre in rng.uniform(-12.0, 12.0, size=(20, 2)):  # dense clusters with duplicates
        points += [tuple(centre), tuple(centre)]
        points += [tuple(centre + (5.0, 0.0)), tuple(centre + (3.0, 4.0))]
        points += [tuple(p) for p in centre + rng.uniform(-6.0, 6.0, size=(8, 2))]
    xy = np.asarray(points, dtype=float)

    tree = cKDTree(xy)
    materialised = np.asarray([len(v) - 1 for v in tree.query_ball_point(xy, 5.0)])
    counted = tree.query_ball_point(xy, 5.0, return_length=True) - 1

    assert np.array_equal(materialised, counted)
    # The cloud must actually exercise both the duplicate and the isolated-point cases.
    assert materialised.min() == 0, "expected at least one point with no neighbour within 5 m"
    assert materialised.max() >= 4, "expected the duplicated origin cluster to be dense"

    # Equality must also survive the assignment into the float accumulator that
    # ``recording_features`` uses for the per-sample neighbour counts.
    from_lists = np.zeros(len(xy), float)
    from_counts = np.zeros(len(xy), float)
    from_lists[:] = materialised
    from_counts[:] = counted
    assert np.array_equal(from_lists, from_counts)
