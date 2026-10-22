import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from paper.generate_trajnet_characterization import (
    characterize,
    generate,
    render_sub_tag_table,
    render_table,
    scene_index_path,
)

# (scene_tag, sub_tags, split, accepted) per window.  Two type-1 windows, one of
# which the TrajNet++ acceptance sampling rejects; one type-2; three type-3 with
# overlapping sub-tags; one type-4.
WINDOWS = [
    (1, [], "train", True),
    (1, [], "train", False),
    (2, [], "train", True),
    (3, [1, 3], "train", True),
    (3, [2], "val", True),
    (3, [4], "test", True),
    (4, [], "test", True),
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "task.json").write_text(
        json.dumps({"history_steps": 8, "future_steps": 12, "sample_hz": 2.5})
    )
    indices = tmp_path / "datasets" / "example" / "indices"
    indices.mkdir(parents=True)
    table = pa.table(
        {
            "scene_tag": pa.array([row[0] for row in WINDOWS], pa.int8()),
            "sub_tags": pa.array([row[1] for row in WINDOWS], pa.list_(pa.int8())),
            "split": pa.array([row[2] for row in WINDOWS]),
            "accepted": pa.array([row[3] for row in WINDOWS]),
        }
    )
    pq.write_table(table, indices / "scenes-seed42.parquet")
    return tmp_path


def test_distribution_covers_every_window_not_only_accepted(root: Path):
    result = characterize("example", root / "datasets", 42)
    assert result["all_windows"]["total_windows"] == 7
    assert result["all_windows"]["counts"] == {"1": 2, "2": 1, "3": 3, "4": 1}
    # The rejected type-1 window is data, not a sampling artefact, so it counts
    # here and is excluded only from the accepted-only view.
    assert result["accepted"]["total_windows"] == 6
    assert result["accepted"]["counts"]["1"] == 1


def test_sub_tags_are_counted_over_type_three_windows(root: Path):
    sub = characterize("example", root / "datasets", 42)["interaction_sub_tags"]
    assert sub["type3_windows"] == 3
    assert sub["counts"] == {"1": 1, "2": 1, "3": 1, "4": 1}
    # Sub-tags are not mutually exclusive: one window carries both 1 and 3, so
    # the fractions sum above one against the type-3 denominator.
    assert sum(sub["fractions_of_type3"].values()) == pytest.approx(4 / 3)


def test_split_breakdown_partitions_the_windows(root: Path):
    by_split = characterize("example", root / "datasets", 42)["by_split"]
    assert [by_split[split]["total_windows"] for split in ("train", "val", "test")] == [4, 1, 2]


def test_missing_index_is_reported_rather_than_silently_skipped(tmp_path: Path):
    (tmp_path / "indices").mkdir()
    with pytest.raises(FileNotFoundError):
        scene_index_path(tmp_path, 42)


def test_ambiguous_split_strategy_is_rejected(tmp_path: Path):
    indices = tmp_path / "indices"
    indices.mkdir()
    for suffix in ("", "-temporal"):
        (indices / f"scenes-seed42{suffix}.parquet").touch()
    with pytest.raises(ValueError):
        scene_index_path(tmp_path, 42)


def test_generate_writes_tables_and_records_the_observation_length(root: Path):
    result = generate(root, 42, datasets=(("example", "Example"),))
    assert result["observation_steps"] == 8
    assert result["prediction_steps"] == 12
    assert json.loads((root / "paper" / "stats" / "trajnet_characterization.json").read_text()) == result

    table = (root / "paper" / "table_trajnet_characterization.tex").read_text()
    assert r"Example & 7 & 28.6 & 14.3 & 42.9 & 14.3 \\" in table
    provenance = {"git_commit": result["git_commit"], "git_dirty": result["git_dirty"]}
    assert table == render_table(result["datasets"], {"example": "Example"}, provenance)

    sub_table = (root / "paper" / "table_trajnet_interaction_sub_tags.tex").read_text()
    assert r"Example & 3 & 33.3 & 33.3 & 33.3 & 33.3 \\" in sub_table
    assert sub_table == render_sub_tag_table(result["datasets"], {"example": "Example"}, provenance)


def test_generate_records_git_provenance_matching_generate_tables_field_names(root: Path):
    """F-007 convention (see src/paper/generate_tables.py): provenance travels under the
    exact keys ``git_commit``/``git_dirty`` so artifacts stay mutually comparable.

    ``root`` here is a synthetic ``tmp_path`` tree, not a git checkout, so
    ``git_provenance`` is expected to degrade to ``{"git_commit": "unknown", "git_dirty":
    None}`` per its documented contract (see I-010's recorded reservation against
    asserting ``git_commit != "unknown"``, which would be stricter than that contract and
    would fail here for an environmental reason rather than a real regression). The
    assertions below therefore check *presence and shape*, not a live, non-degraded value.
    """
    result = generate(root, 42, datasets=(("example", "Example"),))

    for payload in (result, json.loads((root / "paper" / "stats" / "trajnet_characterization.json").read_text())):
        assert "git_commit" in payload, "generated JSON is missing the git_commit provenance field"
        assert "git_dirty" in payload, "generated JSON is missing the git_dirty provenance field"
        assert isinstance(payload["git_commit"], str)
        assert payload["git_dirty"] is None or isinstance(payload["git_dirty"], bool)

    for tex_path in (
        root / "paper" / "table_trajnet_characterization.tex",
        root / "paper" / "table_trajnet_interaction_sub_tags.tex",
    ):
        text = tex_path.read_text()
        assert "git_commit=" in text, f"{tex_path} is missing a git_commit provenance comment"
        assert "git_dirty=" in text, f"{tex_path} is missing a git_dirty provenance comment"
