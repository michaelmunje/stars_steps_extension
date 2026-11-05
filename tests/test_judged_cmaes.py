"""Judged CMA-ES: one ask/tell generation with the rule judge, and the file judge's checks."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplenavsim" / "scripts"))

from judged_cmaes import JudgedSearch  # noqa: E402
from simplenavsim.optimization.judge import FileJudge, RuleJudge  # noqa: E402


def test_one_generation(tmp_path):
    search = JudgedSearch(tmp_path)
    out = search.ask()
    pending = json.loads((out / "pending.json").read_text())
    assert (out / "sheet.png").exists() and len(pending["episodes"]) == 12
    with pytest.raises(FileNotFoundError):
        search.tell(FileJudge())
    record = search.tell(RuleJudge())
    assert record["generation"] == 1 and len(record["fitness"]) == 6
    rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and {"parameters", "label", "fitness", "sim"} <= set(rows[0])
    assert (out / "scored_sheet.png").exists() and (tmp_path / "mean_policy.yaml").exists()


def test_file_judge_rejects_unknown_labels(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps({"a": "great"}))
    with pytest.raises(ValueError):
        FileJudge().label([{"id": "a"}], tmp_path)
