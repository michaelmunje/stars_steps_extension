from pathlib import Path

from config import DatasetSpec
from paper.generate_tables import (
    PAPER_STATS_REQUIRED_FIELDS,
    _stats_cache_is_current,
)
from stats import STATS_SCHEMA_VERSION


def _current_stats() -> dict:
    value = {field: 0 for field in PAPER_STATS_REQUIRED_FIELDS}
    value.update({"stats_schema_version": STATS_SCHEMA_VERSION, "scene_limit": None})
    return value


def test_stats_cache_requires_every_table_field():
    spec = DatasetSpec("example", Path("datasets/example"))
    value = _current_stats()
    assert _stats_cache_is_current(value, spec, None)
    value.pop("cv_fde_mean")
    assert not _stats_cache_is_current(value, spec, None)


def test_stats_cache_requires_current_schema_and_scene_limit():
    spec = DatasetSpec("example", Path("datasets/example"))
    value = _current_stats()
    value["stats_schema_version"] = STATS_SCHEMA_VERSION - 1
    assert not _stats_cache_is_current(value, spec, None)
    value = _current_stats()
    assert not _stats_cache_is_current(value, spec, 50_000)
