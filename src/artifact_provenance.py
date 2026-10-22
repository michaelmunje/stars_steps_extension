"""Small, deterministic provenance helpers for derived publication artifacts."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    """Return ``"sha256:" + hexdigest`` of a single file's raw bytes, read in fixed-size chunks."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def implementation_hash(paths: Iterable[str | Path], *, base: str | Path) -> str:
    """Hash the concatenated contents of exactly the given files, keyed by base-relative path.

    This is **content-addressed over exactly the listed paths and nothing else**: it does not
    follow imports, so if one of those files imports a module that is not itself named in
    ``paths``, a change to that unlisted module changes the caller's actual implementation
    while this hash stays byte-identical. Closure over "everything that affects the output" is
    the caller's responsibility, not this function's.

    Files are iterated in order of their **resolved absolute path** (for deterministic ordering
    regardless of input order), but the label folded into the digest for each file is its path
    **relative to ``base``** — so the same files reached via different absolute routes, or under
    a different ``base``, still hash identically as long as the base-relative labels match. A
    path that does not resolve under ``base`` falls back to using its bare filename as the label
    instead of raising (dormant: no current call site passes such a path).

    See ``source_set_revision`` below for the sibling function this one is easy to mistake for:
    both return a same-shaped ``"sha256:"`` string but answer different questions.
    """
    base = Path(base).resolve()
    hasher = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        try:
            label = path.relative_to(base).as_posix()
        except ValueError:
            label = path.name
        hasher.update(label.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def git_provenance(root: str | Path, timeout: float = 5.0) -> dict[str, Any]:
    """Capture this repository's commit and uncommitted-change status for run provenance.

    Same pattern as ``baselines.status.git_provenance``, adapted for dataset-conversion
    call sites that do not import the baseline run-tracking module. Returns
    ``{"git_commit": <sha>, "git_dirty": <bool>}``. Any resolution failure (not a git
    checkout, ``git`` unavailable, or a timeout) is logged as a warning and reported as
    ``{"git_commit": "unknown", "git_dirty": None}`` rather than raised, since a missing
    provenance field should never abort a conversion run.
    """
    root = Path(root)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                 text=True, timeout=timeout, check=True).stdout.strip()
        porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True, timeout=timeout, check=True).stdout
        return {"git_commit": commit, "git_dirty": bool(porcelain.strip())}
    except Exception as error:
        logger.warning("Could not resolve git provenance for %s: %r", root, error)
        return {"git_commit": "unknown", "git_dirty": None}


def source_set_revision(paths: Iterable[str | Path], *, base: str | Path) -> str:
    """Hash exact file identities and revisions without rereading large Parquet bodies.

    Unlike ``implementation_hash`` above, this is **identity-addressed, not content-addressed**:
    each row is exactly ``(relative path, size_bytes, mtime_ns)`` — the file bytes themselves are
    never read. Both functions return a string with the same ``"sha256:"`` prefix, so their
    outputs are visually indistinguishable, but they answer different questions:
    ``implementation_hash`` changes only when file contents change; this function changes
    whenever the filesystem's recorded modification time changes, even if the content is
    byte-for-byte identical (e.g. after a ``touch``, an ``rsync`` without ``-t``, or checking the
    same files out again on a different machine or clone). Consequently a value from this
    function is **not stable across clones or machines**, and is not recomputable by a third
    party from the file contents alone.

    ``src/scripts/audit_opentraj_table1.py:233-246`` recomputes this and hard-fails publication
    analysis with "stale/unproven analysis index metadata … regenerate publication analysis"
    if it disagrees with the value stamped into the analysis index — so an mtime-only change to
    otherwise-unchanged source files can trip that failure. This is accepted, by-design
    behaviour (see F-008, A1-015), not a defect; do not change the hash basis to be
    content-addressed without a human decision, since that would change every published
    ``canonical_source_set_revision``.
    """
    base = Path(base).resolve()
    rows = []
    for value in sorted((Path(item).resolve() for item in paths), key=str):
        stat = value.stat()
        try:
            label = value.relative_to(base).as_posix()
        except ValueError:
            label = value.name
        rows.append(
            {
                "path": label,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
