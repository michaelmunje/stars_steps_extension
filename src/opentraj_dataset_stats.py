from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from artifact_provenance import sha256_file
from canonical import decode_metadata, validate_canonical_file
# The indicator artifact's contract and expected implementation hash are defined
# once, in the other publication aggregator; this module gates the same artifact
# on the same expectations rather than keeping a second, driftable copy (M-001).
from paper_dataset_metrics import (
    EXPECTED_INDICATOR_IMPLEMENTATION_HASH,
    INDICATOR_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]
OPENTRAJ_COMMIT = "97ec1d5e1b579f26febeaa9083705952deac338a"


@dataclass(frozen=True)
class DatasetGroup:
    name: str
    dataset: str
    sequences: tuple[str, ...] | None = None


ETH_UCY_FOLDS = (
    DatasetGroup("ETH", "eth", ("seq_eth",)),
    DatasetGroup("HOTEL", "eth", ("seq_hotel",)),
    DatasetGroup("UNIV", "ucy", ("students001", "students003")),
    DatasetGroup("ZARA1", "ucy", ("crowds_zara01",)),
    DatasetGroup("ZARA2", "ucy", ("crowds_zara02",)),
)


def _display_path(path: Path) -> str:
    """Render ``path`` repo-relative when possible, so published artifacts stay
    reproducible across machines. Paths outside ``ROOT`` (an externally configured
    dataset root, a test fixture's temp directory, ...) fall back to the absolute
    form rather than raising -- there is no repo-relative form to give them."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _dataset_directory(dataset_root: Path, name: str) -> Path:
    matches = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.lower() == name.lower()]
    if len(matches) > 1:
        raise ValueError(f"Multiple dataset directories match {name}: {matches}")
    return matches[0] if matches else dataset_root / name


def _selected_recordings(dataset_dir: Path, sequences: Iterable[str] | None) -> list[Path]:
    requested = None if sequences is None else set(sequences)
    output = []
    # Each `validate_canonical_file` call opens the Parquet footer twice, so the
    # selected sequences are collected during the single validation pass rather
    # than re-validating every selected path to recover an identity we just had.
    found: set[str] = set()
    for path in sorted((dataset_dir / "converted").glob("*.traj.parquet")):
        identity = validate_canonical_file(path)
        if requested is None or identity.scene_sequence in requested:
            output.append(path)
            found.add(identity.scene_sequence)
    if requested is not None:
        missing = requested - found
        if missing:
            raise FileNotFoundError(f"{dataset_dir.name} is missing canonical sequences {sorted(missing)}")
    return output


def canonical_general_stats(paths: Iterable[Path]) -> dict:
    """Apply OpenTraj ``general_stats.py`` duration/count definitions.

    A canonical ``scene_sequence`` is the equivalent of an upstream OpenTraj
    ``scene_id`` (one source recording), so track identifiers are never merged
    across recordings.
    """
    recording_duration_s = 0.0
    total_trajectory_duration_s = 0.0
    raw_trajectories = 0
    recordings = []
    sequences = []
    for path in tqdm(list(paths), desc="Canonical general stats", unit="file"):
        identity = validate_canonical_file(path)
        parquet = pq.ParquetFile(path)
        metadata = decode_metadata(parquet.schema_arrow)
        sample_hz = float(metadata["canonical_hz"])
        table = parquet.read(columns=["track_id", "step"])
        track_ids = table["track_id"].combine_chunks().to_numpy()
        steps = table["step"].combine_chunks().to_numpy()
        if not len(steps):
            continue
        recording_duration_s += float(steps.max() - steps.min()) / sample_hz
        order = np.lexsort((steps, track_ids))
        ordered_ids = track_ids[order]
        ordered_steps = steps[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_ids)) + 1]
        ends = np.r_[starts[1:] - 1, len(ordered_ids) - 1]
        raw_trajectories += len(starts)
        total_trajectory_duration_s += float(
            np.sum(ordered_steps[ends] - ordered_steps[starts])
        ) / sample_hz
        recordings.append(identity.recording)
        sequences.append(identity.scene_sequence)
    return {
        "recordings": recordings,
        "scene_sequences": sequences,
        "recording_duration_seconds": recording_duration_s,
        "recording_duration_hours": recording_duration_s / 3600.0,
        "total_trajectory_duration_seconds": total_trajectory_duration_s,
        "total_trajectory_duration_hours": total_trajectory_duration_s / 3600.0,
        "raw_trajectories": int(raw_trajectories),
    }


def _checked_indicator_metadata(dataset_dir: Path, indicator_path: Path) -> dict[str, str]:
    """Return the indicator artifact's decoded metadata, refusing a stale one.

    This module publishes ``local_density_max_deduplicated`` from that artifact,
    so it checks the indicator metadata before publishing the same column
    `paper_dataset_metrics`' ``_primitive_source`` publishes from. It applies
    two of the five indicator checks that sibling applies -
    ``indicator_contract`` and ``indicator_implementation_hash`` - plus a
    presence check on ``duplicate_separation_m`` and ``duplicate_min_frames``
    that the sibling does not need (it never reads those thresholds).

    It does **not** check ``analysis_preprocessing``, ``opentraj_reference``,
    or ``source_analysis_index_sha256`` against the index artifact, so an
    indicator report that is paired with the *wrong* index (mismatched
    ``source_analysis_index_sha256``, a different ``analysis_preprocessing``,
    or an unrelated ``opentraj_reference``) still passes here and can reach a
    published number. Closing that gap is tracked as M-012, sequenced behind
    the M-010 decision (extending it today would fail every source, since
    every current index artifact already fails the sibling's implementation-
    hash check).
    """
    metadata = decode_metadata(pq.ParquetFile(indicator_path).schema_arrow)
    expected = {
        "indicator_contract": INDICATOR_CONTRACT,
        "indicator_implementation_hash": EXPECTED_INDICATOR_IMPLEMENTATION_HASH,
    }
    invalid = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if invalid:
        raise ValueError(
            f"{dataset_dir.name}: stale/unproven indicator metadata {invalid} in "
            f"{_display_path(indicator_path)}; regenerate the OpenTraj analysis reports"
        )
    # The thresholds are written by the same code the hash above pins, so a
    # missing one means hand-edited metadata rather than an older artifact.
    missing = [
        key for key in ("duplicate_separation_m", "duplicate_min_frames")
        if metadata.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"{dataset_dir.name}: indicator metadata in {_display_path(indicator_path)} "
            f"passes the implementation-hash gate but omits {missing}"
        )
    return metadata


def _recorded_float(metadata: dict[str, str], key: str, indicator_path: Path) -> float:
    try:
        return float(metadata[key])
    except ValueError as error:
        raise ValueError(
            f"{_display_path(indicator_path)}: {key}={metadata[key]!r} is not a number"
        ) from error


def _recorded_int(metadata: dict[str, str], key: str, indicator_path: Path) -> int:
    try:
        return int(metadata[key])
    except ValueError as error:
        raise ValueError(
            f"{_display_path(indicator_path)}: {key}={metadata[key]!r} is not an integer"
        ) from error


def _analysis_stats(dataset_dir: Path, sequences: Iterable[str] | None) -> dict:
    requested = None if sequences is None else set(sequences)
    publication_index = dataset_dir / "indices" / "opentraj-analysis-trajlets-publication.parquet"
    publication_indicators = dataset_dir / "analysis" / "opentraj_reports_publication" / "trajlet_indicators.parquet"
    if publication_index.exists() and publication_indicators.exists():
        index_path = publication_index
        indicator_path = publication_indicators
    else:
        index_path = dataset_dir / "indices" / "opentraj-analysis-trajlets.parquet"
        indicator_path = dataset_dir / "analysis" / "opentraj_reports" / "trajlet_indicators.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"missing exact trajlet index: {_display_path(index_path)}")
    if not indicator_path.exists():
        raise FileNotFoundError(f"missing OpenTraj indicator report: {_display_path(indicator_path)}")
    # Mirrors opentraj_report.py's `index_revision` provenance field so a
    # regenerated `trajlet_indicators.parquet`/index leaves a traceable hash
    # in this module's own output rather than only the pinned upstream commit.
    source_analysis_index_sha256 = sha256_file(index_path)
    source_indicator_report_sha256 = sha256_file(indicator_path)
    indicator_metadata = _checked_indicator_metadata(dataset_dir, indicator_path)

    index = pq.read_table(
        index_path,
        columns=["scene_sequence", "retained_after_static_filter"],
    ).to_pandas()
    if requested is not None:
        index = index[index.scene_sequence.isin(requested)]
    candidate_trajlets = len(index)
    retained_trajlets = int(index.retained_after_static_filter.sum())

    indicators = pq.read_table(
        indicator_path,
        columns=["scene_sequence", "local_density_max", "local_density_max_deduplicated"],
    ).to_pandas()
    if requested is not None:
        indicators = indicators[indicators.scene_sequence.isin(requested)]
    density = indicators.local_density_max.to_numpy(dtype=float)
    density = density[np.isfinite(density)]
    # Same per-trajlet quantity, recomputed with duplicate-annotation track
    # identities removed from each frame; trajlets belonging to a removed
    # identity are non-finite and drop out here, so this count can be lower.
    deduplicated = indicators.local_density_max_deduplicated.to_numpy(dtype=float)
    deduplicated = deduplicated[np.isfinite(deduplicated)]
    if len(indicators) != retained_trajlets:
        raise ValueError(
            f"{dataset_dir.name}: indicator rows ({len(indicators)}) do not match "
            f"retained trajlets ({retained_trajlets})"
        )
    return {
        "trajlets_4_8s": int(candidate_trajlets),
        "non_static_trajlets_4_8s": retained_trajlets,
        "non_static_trajlets_percent": (
            100.0 * retained_trajlets / candidate_trajlets if candidate_trajlets else None
        ),
        "average_local_density_ped_m2": float(density.mean()) if len(density) else None,
        "median_local_density_ped_m2": float(np.median(density)) if len(density) else None,
        "local_density_trajlets_with_finite_value": int(len(density)),
        "average_local_density_deduplicated_ped_m2": (
            float(deduplicated.mean()) if len(deduplicated) else None
        ),
        "median_local_density_deduplicated_ped_m2": (
            float(np.median(deduplicated)) if len(deduplicated) else None
        ),
        "local_density_deduplicated_trajlets_with_finite_value": int(len(deduplicated)),
        "source_analysis_index_path": _display_path(index_path),
        "source_analysis_index_sha256": source_analysis_index_sha256,
        "source_indicator_report_path": _display_path(indicator_path),
        "source_indicator_report_sha256": source_indicator_report_sha256,
        # Read back from the artifact, not from the live constants: these
        # describe the thresholds the published de-duplicated numbers were
        # actually computed under (M-001).
        "duplicate_separation_m": _recorded_float(indicator_metadata, "duplicate_separation_m", indicator_path),
        "duplicate_min_frames": _recorded_int(indicator_metadata, "duplicate_min_frames", indicator_path),
        "indicator_implementation_hash": indicator_metadata["indicator_implementation_hash"],
    }


def summarize_group(group: DatasetGroup, dataset_root: Path) -> dict:
    dataset_dir = _dataset_directory(dataset_root, group.dataset)
    result = {
        "name": group.name,
        "dataset": group.dataset,
        "requested_scene_sequences": list(group.sequences) if group.sequences is not None else "all",
        "status": "unavailable",
        "dataset_dir": _display_path(dataset_dir),
    }
    paths = _selected_recordings(dataset_dir, group.sequences) if dataset_dir.exists() else []
    if not paths:
        result["reason"] = "no canonical converted recordings"
        return result
    result.update(canonical_general_stats(paths))
    try:
        result.update(_analysis_stats(dataset_dir, group.sequences))
    except FileNotFoundError as error:
        result["status"] = "canonical_ready_analysis_pending"
        result["reason"] = str(error)
        return result
    if result["average_local_density_ped_m2"] is None:
        # No retained trajlet had a finite local-density value (e.g. every
        # retained trajlet has <2 concurrent pedestrians, so
        # `frame_local_density` is NaN throughout). The group is not
        # "complete" because the density average is undefined, not merely
        # zero or missing data.
        result["status"] = "incomplete_density_undefined"
        result["reason"] = "no retained trajlet has a finite local_density_max value"
        return result
    result["status"] = "complete"
    return result


def registered_dataset_groups(root: Path = ROOT) -> tuple[DatasetGroup, ...]:
    task = json.loads((root / "configs" / "task.json").read_text())
    return tuple(DatasetGroup(name, name) for name in task.get("datasets", {}))


def _recorded_duplicate_filter(rows: Iterable[dict]) -> tuple[float, int] | None:
    """The duplicate-filter thresholds the published rows were computed under.

    Read back from the indicator artifacts rather than from the live
    `duplicate_annotations` constants: editing those constants without
    regenerating the reports must not silently restate the published numbers'
    provenance (M-001). Returns None when no row carries a de-duplicated value.
    """
    settings = {
        (row["duplicate_separation_m"], row["duplicate_min_frames"])
        for row in rows
        if row.get("duplicate_separation_m") is not None
    }
    if not settings:
        return None
    if len(settings) > 1:
        raise ValueError(
            "published rows were computed under different duplicate-annotation "
            f"filters {sorted(settings)}; regenerate the OpenTraj analysis reports"
        )
    return settings.pop()


def _deduplicated_local_density_definition(rows: Iterable[dict]) -> str:
    recorded = _recorded_duplicate_filter(rows)
    if recorded is None:
        threshold_text = (
            "a track identity held close to another identity for long enough is one person "
            "annotated twice, and is removed from every frame before the density kernel runs. "
            "No row in this summary carries a de-duplicated value, so this summary records no "
            "artifact thresholds"
        )
    else:
        separation_m, min_frames = recorded
        threshold_text = (
            f"a track identity held within {separation_m} m of another identity for at least "
            f"{min_frames} canonical frames is one person annotated twice, and is removed from "
            "every frame before the density kernel runs. Those two thresholds are read back "
            "from the indicator artifacts the numbers come from (see each row's "
            "duplicate_separation_m/duplicate_min_frames), not from the current source constants"
        )
    return (
        "The same quantity with duplicate pedestrian annotations excluded: "
        f"{threshold_text}. The filter flags nothing at all in ETH or UCY, and "
        "the per-frame kernel itself is unchanged and still bit-exact against pinned upstream. "
        "The published table reports the de-duplicated mean and median; "
        "average_local_density_ped_m2 and median_local_density_ped_m2 retain the unfiltered "
        "values for comparison."
    )


def _definitions(rows: Iterable[dict]) -> dict[str, str]:
    rows = list(rows)
    return {
        "recording_duration": "OpenTraj general_stats.dataset_duration: sum(max(timestamp)-min(timestamp)) over source recordings/scene_sequences.",
        "total_trajectory_duration": "OpenTraj general_stats.total_trajectory_duration: sum(max(timestamp)-min(timestamp)) over (scene_sequence, agent_id) tracks; gaps remain inside the span.",
        "raw_trajectories": (
            "OpenTraj general_stats.num_pedestrians: distinct (scene_sequence, agent_id) "
            "identities in the loaded canonical data, before trajlet/window filters (not raw "
            "detector observations). This count RETAINS duplicate identities: unlike the "
            "de-duplicated local density in the same row, a person annotated under two track "
            "ids is counted twice here."
        ),
        "trajlets_4_8s": "OpenTraj split_trajectories candidates: 12 returned samples, a 13th static endpoint, starts every 7 samples at 2.5 Hz; count is before the >1 m filter.",
        "local_density": (
            "Per-trajlet PEAK local density over the 4.8 s window, not a mean over frames: "
            "OpenTraj takes the per-frame maximum (crowd_density.py:64), where per-frame density "
            "is the a=1 adaptive nearest-neighbour Gaussian estimator, not a fixed-radius count. "
            "Reported as both the mean and the median over retained trajlets; the mean of maxima "
            "is dominated by its upper tail, so the two differ substantially in crowded scenes. "
            "These two columns RETAIN duplicate identities (upstream parity)."
        ),
        "deduplicated_local_density": _deduplicated_local_density_definition(rows),
        "global_density": (
            "Per-frame pedestrians divided by the scene's annotated area, reported in the "
            "per-dataset OpenTraj analysis summaries and plotted in the density figure "
            "(src/opentraj_report.py), not in this table. Like raw_trajectories, and unlike the "
            "de-duplicated local density, it RETAINS duplicate identities: a person annotated "
            "under two track ids contributes twice to the frame count."
        ),
    }


def generate(root: Path = ROOT) -> dict:
    dataset_root = root / "datasets"
    dataset_rows = [summarize_group(group, dataset_root) for group in registered_dataset_groups(root)]
    fold_rows = [summarize_group(group, dataset_root) for group in ETH_UCY_FOLDS]
    result = {
        "opentraj_repository": "https://github.com/crowdbotp/OpenTraj",
        "opentraj_commit": OPENTRAJ_COMMIT,
        "canonical_frequency_hz": 2.5,
        "definitions": _definitions(dataset_rows + fold_rows),
        "compatibility_note": (
            "ETH/UCY reproduce the pinned source exactly. For datasets with temporal gaps, "
            "our trajlet extractor intentionally rejects candidates crossing a gap or scene_sequence "
            "boundary; upstream split_trajectories assumes already-continuous trajectories."
        ),
        "datasets": dataset_rows,
        "eth_ucy_folds": fold_rows,
    }
    output_dir = root / "paper" / "stats"
    paper_dir = root / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "opentraj_dataset_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_csv(dataset_rows, output_dir / "opentraj_dataset_summary.csv")
    _write_tex(dataset_rows, paper_dir / "table_opentraj_dataset_summary.tex")
    _write_tex(fold_rows, paper_dir / "table_opentraj_eth_ucy_folds.tex")
    _write_markdown(result, paper_dir / "opentraj_dataset_summary.md")
    return result


def _density_cell(value: float | None) -> str:
    """Render a density value, or a placeholder when no retained trajlet had one.

    `None` means every retained trajlet's local density was non-finite (e.g. a
    scene with fewer than two concurrent pedestrians throughout), which must
    render as a placeholder rather than crashing on `None:.3f`.
    """
    return f"{value:.3f}" if value is not None else "n/a"


def _write_csv(rows: list[dict], path: Path) -> None:
    columns = (
        "name", "status", "recording_duration_hours", "total_trajectory_duration_hours",
        "raw_trajectories", "trajlets_4_8s", "non_static_trajlets_4_8s",
        "average_local_density_ped_m2", "median_local_density_ped_m2",
        "local_density_trajlets_with_finite_value",
        "average_local_density_deduplicated_ped_m2", "median_local_density_deduplicated_ped_m2",
        "local_density_deduplicated_trajlets_with_finite_value", "reason",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(rows: list[dict], path: Path) -> None:
    lines = []
    for row in rows:
        if row["status"] != "complete":
            continue
        lines.append(
            f"{row['name']} & {row['recording_duration_hours']:.3f} & "
            f"{row['total_trajectory_duration_hours']:.3f} & {row['raw_trajectories']} & "
            f"{row['trajlets_4_8s']} & "
            f"{_density_cell(row['average_local_density_deduplicated_ped_m2'])} & "
            f"{_density_cell(row['median_local_density_deduplicated_ped_m2'])} \\\\"
        )
    path.write_text(
        "\\begin{tabular}{lrrrrrr}\n"
        "Dataset & Recording duration (h) & Total trajectory duration (h) & Raw trajectories & 4.8 s trajlets "
        "& Mean peak local density, duplicate-filtered ($\\mathrm{ped}/\\mathrm{m}^2$) "
        "& Median peak local density, duplicate-filtered ($\\mathrm{ped}/\\mathrm{m}^2$) \\\\\n"
        "\\hline\n" + "\n".join(lines) + "\n\\end{tabular}\n"
    )


def _write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# OpenTraj-compatible dataset summary",
        "",
        f"Pinned reference: `crowdbotp/OpenTraj@{result['opentraj_commit']}`.",
        "",
        "| Dataset | Recording duration (h) | Total trajectory duration (h) | Raw trajectories "
        "| 4.8 s trajlets | Mean peak local density, duplicate-filtered (ped/m²) "
        "| Median peak local density, duplicate-filtered (ped/m²) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["datasets"]:
        if row["status"] == "complete":
            lines.append(
                f"| {row['name']} | {row['recording_duration_hours']:.3f} | "
                f"{row['total_trajectory_duration_hours']:.3f} | {row['raw_trajectories']} | "
                f"{row['trajlets_4_8s']} | "
                f"{_density_cell(row['average_local_density_deduplicated_ped_m2'])} | "
                f"{_density_cell(row['median_local_density_deduplicated_ped_m2'])} | complete |"
            )
        else:
            lines.append(
                f"| {row['name']} | {row.get('recording_duration_hours', '—')} | "
                f"{row.get('total_trajectory_duration_hours', '—')} | {row.get('raw_trajectories', '—')} | "
                f"— | — | — | {row['status']}: {row.get('reason', '')} |"
            )
    lines.extend(["", "## Definitions", ""])
    lines.extend(f"- **{key.replace('_', ' ')}:** {value}" for key, value in result["definitions"].items())
    lines.extend(["", result["compatibility_note"], ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated OpenTraj-compatible dataset statistics")
    parser.parse_args()
    result = generate()
    for row in result["datasets"]:
        if row["status"] == "complete":
            print(
                f"[OK] {row['name']}: raw={row['raw_trajectories']}, "
                f"trajlets={row['trajlets_4_8s']}, total_track_h={row['total_trajectory_duration_hours']:.3f}, "
                f"local_density mean={_density_cell(row['average_local_density_deduplicated_ped_m2'])} "
                f"median={_density_cell(row['median_local_density_deduplicated_ped_m2'])} ped/m2 "
                f"(unfiltered mean={_density_cell(row['average_local_density_ped_m2'])})"
            )
        else:
            print(f"[PENDING] {row['name']}: {row.get('reason', row['status'])}")
    print(ROOT / "paper" / "stats" / "opentraj_dataset_summary.json")
    print(ROOT / "paper" / "table_opentraj_dataset_summary.tex")


if __name__ == "__main__":
    main()
