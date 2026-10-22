"""Regression tests for ``artifact_provenance.py``'s ``git_provenance`` helper.

Covers A1-007: the ``stats`` dict written per dataset (``{name}_sizing.json``) must record
``git_commit`` and ``git_dirty`` (plus ``created_utc``, not produced by this helper) alongside
the existing ``config_hash``, matching the provenance convention already used by
``manifests.py::write_subset_manifest`` and ``artifact_provenance.py::git_provenance``.

The end-to-end contract on the real writer's output -- that ``preprocess_dataset`` actually
carries ``git_commit``, ``git_dirty``, and ``created_utc`` into the written ``stats`` JSON --
is covered by ``tests/test_preprocess_pipeline.py::test_sizing_stats_carry_run_provenance_fields``,
which drives ``preprocess_dataset`` on a synthetic fixture instead of reconstructing the
implementation's dict literal here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from artifact_provenance import git_provenance
from preprocess import git_provenance as preprocess_git_provenance


def test_git_provenance_reports_commit_and_dirty_status_in_this_repo() -> None:
    # preprocess.py imports this helper rather than defining its own copy (Q-013);
    # pin that ``is`` identity here so drift back to a local redefinition, or an
    # import of a different symbol under the same name, fails loudly.
    assert preprocess_git_provenance is git_provenance

    root = Path(__file__).resolve().parents[1]
    provenance = git_provenance(root)

    assert set(provenance) == {"git_commit", "git_dirty"}
    assert isinstance(provenance["git_commit"], str)
    assert provenance["git_commit"] != "unknown"
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert provenance["git_commit"] == expected_commit
    assert isinstance(provenance["git_dirty"], bool)


def test_git_provenance_degrades_gracefully_outside_a_git_checkout(tmp_path: Path) -> None:
    provenance = git_provenance(tmp_path, timeout=5.0)

    assert provenance == {"git_commit": "unknown", "git_dirty": None}


def test_git_provenance_degrades_gracefully_on_timeout(tmp_path: Path) -> None:
    provenance = git_provenance(tmp_path, timeout=1e-9)

    assert provenance == {"git_commit": "unknown", "git_dirty": None}
