from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baselines.aggregate import aggregate
from conftest import write_run_result


def _write_run(tmp_path: Path, name: str, ade: float, fde: float) -> Path:
    return write_run_result(
        tmp_path / name,
        topk_ade=ade,
        topk_fde=fde,
        k=20,
        n_predictions=100,
    )


def test_aggregate_raises_on_missing_evaluation_json(tmp_path: Path):
    """A run directory missing evaluation.json must raise, not be silently dropped.

    Mirrors test_transfer_aggregation_uses_seed_level_standard_error's fixture
    pattern from tests/test_transfer_workflow.py.
    """
    run_dirs = [
        _write_run(tmp_path, "seed0", 1.0, 2.0),
        _write_run(tmp_path, "seed1", 2.0, 3.0),
    ]
    incomplete = tmp_path / "seed2"
    incomplete.mkdir()
    (incomplete / "config.json").write_text(json.dumps({"model": "moflow_teacher"}))
    # No evaluation.json written for this run directory.
    run_dirs.append(incomplete)

    with pytest.raises(FileNotFoundError, match="Missing evaluation result"):
        aggregate(run_dirs)


def test_aggregate_computes_mean_and_standard_error_across_seeds(tmp_path: Path):
    run_dirs = [
        _write_run(tmp_path, "seed0", 1.0, 2.0),
        _write_run(tmp_path, "seed1", 2.0, 3.0),
        _write_run(tmp_path, "seed2", 3.0, 4.0),
    ]

    result = aggregate(run_dirs)

    assert result["n_seeds"] == 3
    assert len(result["runs"]) == 3
    assert result["metrics"]["topk_ade"]["mean"] == 2.0
    assert result["metrics"]["topk_ade"]["n"] == 3
    assert np.isclose(result["metrics"]["topk_ade"]["standard_error"], 1 / np.sqrt(3))
    assert result["metrics"]["topk_fde"]["mean"] == 3.0
    # k and n_predictions are excluded from aggregated metrics.
    assert "k" not in result["metrics"]
    assert "n_predictions" not in result["metrics"]
