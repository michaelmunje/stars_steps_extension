"""D-005-scope: the five adapters that previously wrote no run-provenance sidecar
(``eth``, ``grand_central``, ``madras``, ``PTDPES``, ``ucy``) now each write a minimal
``<output>.provenance.json`` sibling next to their converted output, using the same
``artifact_provenance.git_provenance`` helper (and the same ``git_commit``/``git_dirty``
field names) as ``atc``/``steps``'s existing sidecars.

Modeled on ``tests/test_preprocess.py``'s coverage of ``git_provenance`` itself: this file
covers each adapter's own ``_write_provenance_sidecar`` wrapper, including the graceful
degradation ``git_provenance`` falls back to outside a git checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_provenance import git_provenance
from datasets_preprocessing.eth import convert as eth
from datasets_preprocessing.grand_central import convert as grand_central
from datasets_preprocessing.madras import convert as madras
from datasets_preprocessing.PTDPES import convert as ptdpes
from datasets_preprocessing.ucy import convert as ucy


MODULES = {
    "eth": eth,
    "grand_central": grand_central,
    "madras": madras,
    "PTDPES": ptdpes,
    "ucy": ucy,
}


@pytest.mark.parametrize("name", sorted(MODULES))
def test_sidecar_carries_the_shared_git_provenance_fields(name: str, tmp_path: Path) -> None:
    module = MODULES[name]
    output = tmp_path / "converted" / "recording.traj.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"")  # sidecar naming only depends on the output path

    provenance = git_provenance(Path(__file__).resolve().parents[1])
    module._write_provenance_sidecar(output, provenance)

    sidecar = output.with_suffix(".provenance.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())

    # Same field names as the three adapters D-005's core fix already covers.
    assert payload["git_commit"] == provenance["git_commit"]
    assert payload["git_dirty"] == provenance["git_dirty"]
    assert isinstance(payload["git_commit"], str)
    assert payload["git_commit"] != "unknown"

    # Minimal extra provenance fields this issue asked for.
    assert payload["converter_version"] == module.CONVERTER_VERSION
    assert isinstance(payload["invocation"], str)
    assert payload["created_utc"].endswith("+00:00")


@pytest.mark.parametrize("name", sorted(MODULES))
def test_sidecar_degrades_gracefully_outside_a_git_checkout(name: str, tmp_path: Path) -> None:
    module = MODULES[name]
    output = tmp_path / "converted" / "recording.traj.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"")

    # tmp_path is not a git checkout, so git_provenance() degrades rather than raising --
    # same fallback contract as the three existing adapters.
    provenance = git_provenance(tmp_path, timeout=5.0)
    assert provenance == {"git_commit": "unknown", "git_dirty": None}

    module._write_provenance_sidecar(output, provenance)

    payload = json.loads(output.with_suffix(".provenance.json").read_text())
    assert payload["git_commit"] == "unknown"
    assert payload["git_dirty"] is None
