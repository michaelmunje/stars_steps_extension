"""Publication-only aggregation for STEPS paper Tables 1 and 2.

This module intentionally does not import ``stats.py`` or model forecasting
indices.  It combines exact OpenTraj analysis-trajlet primitives with an
independently enumerated, raw-coordinate publication-CV population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

from artifact_provenance import implementation_hash, sha256_file, source_set_revision
from duplicate_annotations import DUPLICATE_MIN_FRAMES, DUPLICATE_SEPARATION_M
from canonical import CANONICAL_HZ, decode_metadata, recording_identity, validate_canonical_file
from opentraj_indicators import PEDESTRIAN_RADIUS_M


SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = SOURCE_ROOT.parent
LOG_DIR = ROOT / "logs"
logger = logging.getLogger(__name__)
MANIFEST_PATH = ROOT / "configs" / "protocols" / "paper_datasets.json"
OPENTRAJ_REPOSITORY = "https://github.com/crowdbotp/OpenTraj"
OPENTRAJ_COMMIT = "97ec1d5e1b579f26febeaa9083705952deac338a"
OPENTRAJ_REFERENCE = f"crowdbotp/OpenTraj@{OPENTRAJ_COMMIT}"
PUBLICATION_CV_PREPROCESSING_MODES = (
    "raw_world_coordinates",
    "opentraj_kalman_smooth",
)
ANALYSIS_INDEX_CONTRACT = "opentraj-analysis-trajlet-index-v1"
INDICATOR_CONTRACT = "opentraj-retained-trajlet-indicators-v1"
EXPECTED_ANALYSIS_IMPLEMENTATION_HASH = implementation_hash(
    (
        SOURCE_ROOT / "analysis_pipeline.py",
        SOURCE_ROOT / "analysis_trajlets.py",
        SOURCE_ROOT / "preprocessing_views.py",
    ),
    base=SOURCE_ROOT,
)
EXPECTED_INDICATOR_IMPLEMENTATION_HASH = implementation_hash(
    (
        SOURCE_ROOT / "opentraj_indicators.py",
        SOURCE_ROOT / "opentraj_report.py",
        SOURCE_ROOT / "duplicate_annotations.py",
    ),
    base=SOURCE_ROOT,
)
REQUIRED_ROW_KEYS = (
    "steps", "eth", "ucy", "madras",
    "grand_central", "ptdpes",
)


@dataclass(frozen=True)
class PaperSource:
    canonical_dataset: str
    scene_sequences: tuple[str, ...] | str
    analysis_variant: str = "default"


@dataclass(frozen=True)
class PaperDataset:
    key: str
    label: str
    setting: str
    num_physical_scenes: int
    annotation_source: str
    sources: tuple[PaperSource, ...]


@dataclass(frozen=True)
class PublicationCVSummary:
    count: int
    fde_sum: float

    @property
    def mean(self) -> float:
        return self.fde_sum / self.count if self.count else float("nan")


def load_paper_datasets(path: Path = MANIFEST_PATH) -> tuple[PaperDataset, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[PaperDataset] = []
    for item in payload["rows"]:
        sources = []
        for source in item["sources"]:
            selected = source["scene_sequences"]
            if isinstance(selected, list):
                selected = tuple(str(value) for value in selected)
            elif selected != "all_canonical":
                raise ValueError(
                    f"{item['key']}: scene_sequences must be a list or all_canonical"
                )
            sources.append(
                PaperSource(
                    canonical_dataset=str(source["canonical_dataset"]),
                    scene_sequences=selected,
                    analysis_variant=str(source.get("analysis_variant", "default")),
                )
            )
        rows.append(
            PaperDataset(
                key=str(item["key"]),
                label=str(item["label"]),
                setting=str(item["setting"]),
                num_physical_scenes=int(item["num_physical_scenes"]),
                annotation_source=str(item["annotation_source"]),
                sources=tuple(sources),
            )
        )
    keys = tuple(row.key for row in rows)
    if keys != REQUIRED_ROW_KEYS:
        raise ValueError(f"Publication manifest rows must be exactly {REQUIRED_ROW_KEYS}; got {keys}")
    return tuple(rows)


def _all_canonical_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in (dataset_dir / "converted").rglob("*.traj.parquet")
        if "Test_set" not in path.parts and "Test_set" not in path.name
    )


def select_canonical_files(source: PaperSource) -> tuple[list[Path], tuple[str, ...]]:
    dataset_dir = ROOT / "datasets" / source.canonical_dataset
    files = _all_canonical_files(dataset_dir)
    if not files:
        raise FileNotFoundError(
            f"{source.canonical_dataset}: no canonical trajectories under {dataset_dir / 'converted'}"
        )
    identities = [(path, validate_canonical_file(path)) for path in files]
    recording_keys = [
        (identity.scene_sequence, identity.recording) for _, identity in identities
    ]
    if len(recording_keys) != len(set(recording_keys)):
        raise ValueError(
            f"{source.canonical_dataset}: duplicate (scene_sequence, recording) identities"
        )
    available = {identity.scene_sequence for _, identity in identities}
    if source.scene_sequences == "all_canonical":
        selected_sequences = tuple(sorted(available))
    else:
        selected_sequences = tuple(source.scene_sequences)
        missing = sorted(set(selected_sequences) - available)
        if missing:
            raise FileNotFoundError(
                f"{source.canonical_dataset}: publication scene sequences are missing: {missing}"
            )
    selected = [
        path for path, identity in identities
        if identity.scene_sequence in set(selected_sequences)
    ]
    if not selected:
        raise FileNotFoundError(f"{source.canonical_dataset}: publication selection is empty")
    return selected, selected_sequences


def publication_cv_for_xy(steps: np.ndarray, xy: np.ndarray) -> PublicationCVSummary:
    """Enumerate strict 8+12 publication-CV windows for one position signal."""
    steps = np.asarray(steps, dtype=np.int64)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (len(steps), 2):
        raise ValueError("xy must have shape (len(steps), 2)")
    if len(steps) and np.any(np.diff(steps) <= 0):
        raise ValueError("steps must be strictly increasing")
    boundaries = np.flatnonzero(np.diff(steps) != 1) + 1
    count = 0
    fde_sum = 0.0
    for segment in np.split(np.arange(len(steps)), boundaries):
        n_windows = max(0, len(segment) - 20 + 1)
        if not n_windows:
            continue
        starts = np.arange(n_windows)
        values = xy[segment]
        delta = values[starts + 7] - values[starts + 6]
        pred_final = values[starts + 7] + 12.0 * delta
        fde_sum += float(np.linalg.norm(pred_final - values[starts + 19], axis=1).sum())
        count += n_windows
    return PublicationCVSummary(count=count, fde_sum=fde_sum)


def _publication_cv_source(
    canonical_path: Path,
    preprocessing_mode: str,
) -> tuple[Path, tuple[str, str], dict]:
    if preprocessing_mode == "raw_world_coordinates":
        return canonical_path, ("x", "y"), _file_revision(canonical_path)
    if preprocessing_mode != "opentraj_kalman_smooth":
        raise ValueError(f"Unknown publication CV preprocessing: {preprocessing_mode}")

    recording = canonical_path.name.removesuffix(".traj.parquet")
    analysis_path = (
        canonical_path.parent.parent
        / "analysis"
        / "opentraj_kalman_smooth"
        / f"{recording}.analysis.parquet"
    )
    if not analysis_path.is_file():
        raise FileNotFoundError(
            f"Missing OpenTraj-smoothed CV source for {canonical_path}: {analysis_path}"
        )
    metadata = decode_metadata(pq.ParquetFile(analysis_path).schema_arrow)
    canonical_stat = canonical_path.stat()
    expected_revision = f"{canonical_stat.st_size}:{canonical_stat.st_mtime_ns}"
    required = {
        "analysis_preprocessing": "opentraj_kalman_smooth",
        "opentraj_reference": OPENTRAJ_REFERENCE,
        "canonical_source_revision": expected_revision,
    }
    mismatch = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatch:
        raise ValueError(
            f"Stale or unproven OpenTraj analysis view {analysis_path}: {mismatch}"
        )
    return analysis_path, ("smoothed_x", "smoothed_y"), _file_revision(analysis_path)


def publication_cv_for_files(
    files: Iterable[Path],
    preprocessing_mode: str = "raw_world_coordinates",
) -> tuple[PublicationCVSummary, list[dict]]:
    if preprocessing_mode not in PUBLICATION_CV_PREPROCESSING_MODES:
        raise ValueError(f"Unknown publication CV preprocessing: {preprocessing_mode}")
    total_count = 0
    total_fde = 0.0
    revisions = []
    for path in files:
        source, (x_column, y_column), revision = _publication_cv_source(
            path, preprocessing_mode
        )
        revisions.append(revision)
        table = pq.read_table(
            source, columns=["track_id", "step", x_column, y_column]
        )
        frame = table.to_pandas().sort_values(["track_id", "step"], kind="stable")
        for _, track in frame.groupby("track_id", sort=False):
            summary = publication_cv_for_xy(
                track.step.to_numpy(), track[[x_column, y_column]].to_numpy()
            )
            total_count += summary.count
            total_fde += summary.fde_sum
    return PublicationCVSummary(total_count, total_fde), revisions


def _artifact_paths(source: PaperSource) -> tuple[Path, Path]:
    dataset_dir = ROOT / "datasets" / source.canonical_dataset
    suffix = "" if source.analysis_variant == "default" else f"-{source.analysis_variant}"
    index = dataset_dir / "indices" / f"opentraj-analysis-trajlets{suffix}.parquet"
    report_dir = dataset_dir / "analysis" / (
        "opentraj_reports" if source.analysis_variant == "default"
        else f"opentraj_reports_{source.analysis_variant}"
    )
    return index, report_dir / "trajlet_indicators.parquet"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _file_revision(path: Path, *, content_hash: bool = False) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        result["sha256"] = _sha256_file(path)
    if path.suffix == ".parquet":
        metadata = {
            key.decode(): value.decode()
            for key, value in (pq.ParquetFile(path).schema_arrow.metadata or {}).items()
        }
        canonical_keys = (
            "dataset", "scene_id", "scene_sequence", "recording",
            "canonical_schema_version", "canonical_hz", "coordinate_frame",
            "position_representation", "source_file",
        )
        selected = {key: metadata[key] for key in canonical_keys if key in metadata}
        if selected:
            result["canonical_metadata"] = selected
    return result


def _primitive_source(source: PaperSource, cv_preprocessing_mode: str) -> dict:
    files, sequences = select_canonical_files(source)
    sequence_set = set(sequences)
    index_path, indicators_path = _artifact_paths(source)
    missing = [str(path) for path in (index_path, indicators_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{source.canonical_dataset}: missing OpenTraj publication artifacts: {missing}"
        )

    index_metadata = decode_metadata(pq.ParquetFile(index_path).schema_arrow)
    expected_index_metadata = {
        "analysis_index_contract": ANALYSIS_INDEX_CONTRACT,
        "opentraj_reference": OPENTRAJ_REFERENCE,
        "analysis_implementation_hash": EXPECTED_ANALYSIS_IMPLEMENTATION_HASH,
        "canonical_source_set_revision": source_set_revision(
            files,
            base=ROOT / "datasets" / source.canonical_dataset,
        ),
    }
    invalid_index_metadata = {
        key: {"expected": value, "actual": index_metadata.get(key)}
        for key, value in expected_index_metadata.items()
        if index_metadata.get(key) != value
    }
    if invalid_index_metadata:
        raise ValueError(
            f"{source.canonical_dataset}: stale/unproven analysis index metadata "
            f"{invalid_index_metadata}; regenerate publication analysis"
        )

    indicator_metadata = decode_metadata(pq.ParquetFile(indicators_path).schema_arrow)
    expected_indicator_metadata = {
        "indicator_contract": INDICATOR_CONTRACT,
        "analysis_preprocessing": "opentraj_kalman_smooth",
        "opentraj_reference": OPENTRAJ_REFERENCE,
        "indicator_implementation_hash": EXPECTED_INDICATOR_IMPLEMENTATION_HASH,
        "source_analysis_index_sha256": sha256_file(index_path),
    }
    invalid_indicator_metadata = {
        key: {"expected": value, "actual": indicator_metadata.get(key)}
        for key, value in expected_indicator_metadata.items()
        if indicator_metadata.get(key) != value
    }
    if invalid_indicator_metadata:
        raise ValueError(
            f"{source.canonical_dataset}: stale/unproven indicator metadata "
            f"{invalid_indicator_metadata}; regenerate publication analysis"
        )

    index = pq.read_table(index_path).to_pandas()
    index = index[index.scene_sequence.isin(sequence_set)]
    required_index = {
        "scene_sequence", "recording", "agent_id", "source_start_step",
        "source_returned_end_step", "source_static_endpoint_step",
        "n_returned_samples", "static_displacement_m",
        "retained_after_static_filter", "split",
    }
    if not required_index.issubset(index.columns):
        raise ValueError(
            f"{source.canonical_dataset}: analysis index is missing publication fields "
            f"{sorted(required_index-set(index.columns))}"
        )
    if len(index) and not (
        (index.n_returned_samples == 12).all()
        and (index.source_returned_end_step-index.source_start_step == 11).all()
        and (index.source_static_endpoint_step-index.source_start_step == 12).all()
        and (index.retained_after_static_filter == (index.static_displacement_m > 1.0)).all()
        and (index.split == "publication").all()
    ):
        raise ValueError(f"{source.canonical_dataset}: analysis index violates the exact OpenTraj trajlet contract")
    index_keys = ["scene_sequence", "recording", "agent_id", "source_start_step"]
    if index.duplicated(index_keys).any():
        raise ValueError(f"{source.canonical_dataset}: duplicate publication analysis trajlet identities")
    retained = index[index.retained_after_static_filter]
    indicators = pq.read_table(
        indicators_path,
        columns=[
            "scene_sequence", "recording", "agent_id", "source_start_step",
            "local_density_max", "local_density_max_deduplicated",
            "path_efficiency", "dca_min", "ttc_min",
        ],
    ).to_pandas()
    indicators = indicators[indicators.scene_sequence.isin(sequence_set)]
    if len(indicators) != len(retained):
        raise ValueError(
            f"{source.canonical_dataset}: retained trajlet/index mismatch "
            f"({len(retained)} index rows vs {len(indicators)} indicator rows)"
        )
    if indicators.duplicated(index_keys).any():
        raise ValueError(f"{source.canonical_dataset}: duplicate publication indicator identities")
    retained_keys = set(map(tuple, retained[index_keys].itertuples(index=False, name=None)))
    indicator_keys = set(map(tuple, indicators[index_keys].itertuples(index=False, name=None)))
    if retained_keys != indicator_keys:
        raise ValueError(f"{source.canonical_dataset}: publication indicator identities do not match retained trajlets")

    identities: set[tuple[str, str, int]] = set()
    for path in files:
        identity = recording_identity(path)
        track_ids = pq.read_table(path, columns=["track_id"])["track_id"].to_numpy()
        identities.update(
            (identity.scene_sequence, identity.recording, int(track_id))
            for track_id in np.unique(track_ids)
        )
    cv, cv_revisions = publication_cv_for_files(files, cv_preprocessing_mode)
    return {
        "canonical_dataset": source.canonical_dataset,
        "included_scene_sequences": list(sequences),
        "source_trajectory_identities": identities,
        "candidate_count": int(len(index)),
        "retained_count": int(len(retained)),
        "local_density": indicators.local_density_max.to_numpy(dtype=float),
        "local_density_deduplicated": indicators.local_density_max_deduplicated.to_numpy(dtype=float),
        "path_efficiency": indicators.path_efficiency.to_numpy(dtype=float),
        "dca": indicators.dca_min.to_numpy(dtype=float),
        "ttc": indicators.ttc_min.to_numpy(dtype=float),
        "cv_count": cv.count,
        "cv_fde_sum": cv.fde_sum,
        "cv_preprocessing_mode": cv_preprocessing_mode,
        "cv_source_revisions": cv_revisions,
        "index_path": str(index_path.relative_to(ROOT)),
        "indicator_path": str(indicators_path.relative_to(ROOT)),
        "canonical_file_revisions": [_file_revision(path) for path in files],
        "index_revision": _file_revision(index_path, content_hash=True),
        "indicator_revision": _file_revision(indicators_path, content_hash=True),
    }


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _aggregate_row(dataset: PaperDataset, primitives: list[dict]) -> dict:
    retained = sum(item["retained_count"] for item in primitives)
    candidate = sum(item["candidate_count"] for item in primitives)
    # Include canonical dataset in the identity namespace when pooling STEPS sources.
    namespaced = {
        (item["canonical_dataset"],) + identity
        for item in primitives
        for identity in item["source_trajectory_identities"]
    }
    density = np.concatenate([item["local_density"] for item in primitives])
    density_deduplicated = np.concatenate([item["local_density_deduplicated"] for item in primitives])
    efficiency = np.concatenate([item["path_efficiency"] for item in primitives])
    dca = np.concatenate([item["dca"] for item in primitives])
    ttc = np.concatenate([item["ttc"] for item in primitives])
    finite_density = _finite(density)
    finite_density_deduplicated = _finite(density_deduplicated)
    finite_efficiency = _finite(efficiency)
    finite_dca = _finite(dca)
    if retained <= 0:
        raise ValueError(f"{dataset.label}: no retained OpenTraj trajlets")
    if not len(finite_density_deduplicated) or not len(finite_efficiency):
        raise ValueError(f"{dataset.label}: required OpenTraj metrics are empty")
    cv_count = sum(item["cv_count"] for item in primitives)
    cv_fde_sum = sum(item["cv_fde_sum"] for item in primitives)
    cv_modes = {item.get("cv_preprocessing_mode", "raw_world_coordinates") for item in primitives}
    if len(cv_modes) != 1:
        raise ValueError(f"{dataset.label}: mixed publication CV preprocessing modes {cv_modes}")
    cv_preprocessing_mode = cv_modes.pop()
    if cv_count <= 0:
        raise ValueError(f"{dataset.label}: no complete publication CV windows")
    return {
        "key": dataset.key,
        "label": dataset.label,
        "setting": dataset.setting,
        "num_physical_scenes": dataset.num_physical_scenes,
        "annotation_source": dataset.annotation_source,
        "source_trajectory_count": len(namespaced),
        "candidate_opentraj_trajlet_count": candidate,
        "retained_opentraj_trajlet_count": retained,
        "retained_trajlet_duration_hours": retained * 4.8 / 3600.0,
        "finite_local_density_count": int(len(finite_density)),
        "local_density_mean": float(finite_density.mean()),
        "local_density_median": float(np.median(finite_density)),
        "local_density_peak": float(finite_density.max()),
        "finite_local_density_deduplicated_count": int(len(finite_density_deduplicated)),
        "local_density_deduplicated_mean": float(finite_density_deduplicated.mean()),
        "local_density_deduplicated_median": float(np.median(finite_density_deduplicated)),
        "local_density_deduplicated_peak": float(finite_density_deduplicated.max()),
        "finite_path_efficiency_count": int(len(finite_efficiency)),
        "path_efficiency_mean": float(finite_efficiency.mean()),
        "finite_dca_count": int(len(finite_dca)),
        "dca_mean": float(finite_dca.mean()) if len(finite_dca) else None,
        "ttc_lt_1_count": int(np.sum(np.isfinite(ttc) & (ttc < 1.0))),
        "ttc_lt_3_count": int(np.sum(np.isfinite(ttc) & (ttc < 3.0))),
        "ttc_denominator": retained,
        "ttc_lt_1_pct": float(100.0 * np.sum(np.isfinite(ttc) & (ttc < 1.0)) / retained),
        "ttc_lt_3_pct": float(100.0 * np.sum(np.isfinite(ttc) & (ttc < 3.0)) / retained),
        "publication_cv_window_count": cv_count,
        "cv_fde_mean": float(cv_fde_sum / cv_count),
        "included_scene_sequences": {
            item["canonical_dataset"]: item["included_scene_sequences"]
            for item in primitives
        },
        "analysis_preprocessing_mode": "opentraj_kalman_smooth",
        "publication_cv_preprocessing_mode": cv_preprocessing_mode,
        "opentraj_upstream_repo": OPENTRAJ_REPOSITORY,
        "opentraj_pinned_commit": OPENTRAJ_COMMIT,
        "source_artifacts": [
            {key: item[key] for key in (
                "canonical_dataset", "index_path", "indicator_path",
                "canonical_file_revisions", "cv_source_revisions",
                "index_revision", "indicator_revision",
            )}
            for item in primitives
        ],
    }


def _provenance_hash() -> str:
    hasher = hashlib.sha256()
    for path in (
        MANIFEST_PATH,
        SOURCE_ROOT / "artifact_provenance.py",
        SOURCE_ROOT / "analysis_trajlets.py",
        SOURCE_ROOT / "analysis_pipeline.py",
        SOURCE_ROOT / "preprocessing_views.py",
        SOURCE_ROOT / "opentraj_indicators.py",
        SOURCE_ROOT / "opentraj_report.py",
        # M-004: the de-duplicated local density this payload publishes is
        # defined here (separation/min-frame thresholds and the pairing rule),
        # so two payloads with different de-duplicated values would otherwise
        # share an identical `analysis_code_config_hash`. Ordered to mirror
        # EXPECTED_INDICATOR_IMPLEMENTATION_HASH's tuple.
        SOURCE_ROOT / "duplicate_annotations.py",
        ROOT / "configs" / "task.json",
        Path(__file__),
    ):
        try:
            label = path.relative_to(ROOT).as_posix()
        except ValueError:
            label = path.name
        hasher.update(label.encode())
        hasher.update(path.read_bytes())
    return "sha256:" + hasher.hexdigest()


def _configured_cv_preprocessing_mode() -> str:
    task = json.loads((ROOT / "configs" / "task.json").read_text(encoding="utf-8"))
    mode = str(
        task.get("publication", {}).get(
            "cv_preprocessing", "raw_world_coordinates"
        )
    )
    if mode not in PUBLICATION_CV_PREPROCESSING_MODES:
        raise ValueError(f"Unknown publication CV preprocessing in configs/task.json: {mode}")
    return mode


def compute_paper_dataset_metrics(
    *,
    allow_partial: bool = False,
    cv_preprocessing_mode: str | None = None,
) -> dict:
    cv_preprocessing_mode = cv_preprocessing_mode or _configured_cv_preprocessing_mode()
    if cv_preprocessing_mode not in PUBLICATION_CV_PREPROCESSING_MODES:
        raise ValueError(f"Unknown publication CV preprocessing: {cv_preprocessing_mode}")
    rows = []
    missing: dict[str, str] = {}
    source_cache: dict[PaperSource, dict] = {}
    for dataset in load_paper_datasets():
        try:
            primitives = []
            for source in dataset.sources:
                if source not in source_cache:
                    source_cache[source] = _primitive_source(source, cv_preprocessing_mode)
                primitives.append(source_cache[source])
            rows.append(_aggregate_row(dataset, primitives))
        except (FileNotFoundError, ValueError) as error:
            missing[dataset.key] = str(error)
    if missing and not allow_partial:
        details = "\n".join(f"  - {key}: {value}" for key, value in missing.items())
        raise RuntimeError(f"Publication metrics are incomplete:\n{details}")
    payload = {
        "schema_version": 4,
        "definition": {
            "duration": "retained OpenTraj trajlet count * 4.8 s / 3600",
            "local_density": (
                "per-retained-trajlet PEAK local density (OpenTraj takes the maximum over each "
                "4.8 s trajlet), summarised as mean, median and maximum over trajlets. The mean "
                "of maxima is dominated by its upper tail, so the median is reported beside it. "
                "local_density_deduplicated_* excludes duplicate pedestrian annotations "
                f"(identities within {DUPLICATE_SEPARATION_M} m of another for at least "
                f"{DUPLICATE_MIN_FRAMES} canonical frames) from each frame before the density "
                "kernel runs, and is what the published table reports"
            ),
            "path_efficiency": "mean retained-trajlet OpenTraj path efficiency",
            "dca": (
                "mean over retained trajlets of the per-trajlet minimum constant-relative-velocity "
                "distance of closest approach: per canonical frame, the minimum strictly positive DCA "
                "over all pedestrians co-present in that frame, then the minimum over the trajlet's 12 "
                "frames; trajlets with no such frame are excluded (see finite_dca_count). Time to "
                "closest approach is clamped at 0, so already-separating pairs contribute their current "
                "separation. No pedestrian-radius gate applies to this column."
            ),
            "ttc": (
                "fraction of ALL retained trajlets (denominator = retained_opentraj_trajlet_count) whose "
                "per-trajlet minimum time to closest approach is below the threshold. Only neighbour "
                f"pairs whose constant-relative-velocity closest approach is under 2*Rp = "
                f"{2*PEDESTRIAN_RADIUS_M:.2f} m (Rp = {PEDESTRIAN_RADIUS_M} m, OpenTraj) with strictly "
                "positive time to closest approach qualify; trajlets with no qualifying pair are "
                "non-finite and count in the denominator only. This is time to closest approach, not "
                "time to first contact at 2*Rp."
            ),
            "cv_fde": (
                "mean deterministic last-observed-displacement FDE over independent "
                f"complete 8+12 windows using {cv_preprocessing_mode} positions"
            ),
        },
        "canonical_hz": CANONICAL_HZ,
        "publication_cv_preprocessing_mode": cv_preprocessing_mode,
        "opentraj_upstream_repo": OPENTRAJ_REPOSITORY,
        "opentraj_pinned_commit": OPENTRAJ_COMMIT,
        "analysis_code_config_hash": _provenance_hash(),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "metric_units": {
            "duration": "hours of retained 4.8 s OpenTraj trajlets",
            "local_density": "pedestrians_per_m2",
            "cv_fde": "metres",
            "path_efficiency": "unitless",
            "dca": "metres",
            "ttc_thresholds": "percent_of_all_retained_trajlets",
        },
        "population_contract": {
            "analysis": "complete unsampled retained OpenTraj trajlet population",
            "publication_cv": (
                "complete unsampled consecutive 8+12 single-track windows at stride 1; "
                f"positions={cv_preprocessing_mode}"
            ),
            "split_filtering": False,
            "model_index_filtering": False,
        },
        "complete": not missing,
        "missing": missing,
        "rows": rows,
    }
    return payload


def _latex_integer(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def render_table1(rows: list[dict]) -> str:
    lines = [
        "\\begin{tabular}{llrrrlr}",
        "\\hline",
        "Dataset & Setting & \\# Scenes & Dur. (h) & \\# Traj. & Annotation source & Local dens. (dedup.) mean / median \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['setting']} & {row['num_physical_scenes']} & "
            f"{row['retained_trajlet_duration_hours']:.2f} & "
            f"{_latex_integer(row['source_trajectory_count'])} & {row['annotation_source']} & "
            f"{row['local_density_deduplicated_mean']:.3f} / "
            f"{row['local_density_deduplicated_median']:.3f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_table2(rows: list[dict]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Dataset & CV FDE & Path efficiency & DCA & TTC $<1$ s (\\%) & TTC $<3$ s (\\%) \\\\",
        "\\hline",
    ]
    for row in rows:
        if row["dca_mean"] is None:
            raise ValueError(f"{row['label']}: DCA is undefined")
        lines.append(
            f"{row['label']} & {row['cv_fde_mean']:.3f} & "
            f"{row['path_efficiency_mean']:.3f} & {row['dca_mean']:.3f} & "
            f"{row['ttc_lt_1_pct']:.1f} & {row['ttc_lt_3_pct']:.1f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _validate_publication_payload(payload: dict, *, allow_partial: bool) -> None:
    rows = payload["rows"]
    keys = tuple(row["key"] for row in rows)
    expected = tuple(key for key in REQUIRED_ROW_KEYS if key not in payload["missing"])
    if keys != expected:
        raise ValueError(f"Publication row order/membership mismatch: expected {expected}, got {keys}")
    if not allow_partial and (not payload["complete"] or keys != REQUIRED_ROW_KEYS):
        raise ValueError("Strict publication output must contain every required row")
    numeric_fields = (
        "retained_trajlet_duration_hours", "local_density_mean", "local_density_median",
        "local_density_peak", "local_density_deduplicated_mean",
        "local_density_deduplicated_median", "local_density_deduplicated_peak",
        "cv_fde_mean", "path_efficiency_mean", "dca_mean", "ttc_lt_1_pct",
        "ttc_lt_3_pct",
    )
    for row in rows:
        missing_numeric = [
            field for field in numeric_fields
            if row.get(field) is None or not np.isfinite(float(row[field]))
        ]
        if missing_numeric:
            raise ValueError(f"{row['label']}: invalid publication values {missing_numeric}")
        if row["ttc_denominator"] != row["retained_opentraj_trajlet_count"]:
            raise ValueError(f"{row['label']}: TTC denominator is not all retained trajlets")
        if row["publication_cv_preprocessing_mode"] != payload.get(
            "publication_cv_preprocessing_mode"
        ):
            raise ValueError(f"{row['label']}: publication CV preprocessing provenance mismatch")
        if not row["source_artifacts"]:
            raise ValueError(f"{row['label']}: source provenance is empty")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple dataset preprocessing wrappers may finish together and refresh the
    # same deterministic publication tables.  A per-process temporary prevents
    # one writer from replacing another writer's in-progress file.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def generate_publication_tables(
    *,
    allow_partial: bool = False,
    cv_preprocessing_mode: str | None = None,
) -> dict:
    payload = compute_paper_dataset_metrics(
        allow_partial=allow_partial,
        cv_preprocessing_mode=cv_preprocessing_mode,
    )
    _validate_publication_payload(payload, allow_partial=allow_partial)
    table1 = render_table1(payload["rows"])
    table2 = render_table2(payload["rows"])
    stats_path = ROOT / "paper" / "stats" / "paper_dataset_metrics.json"
    # Preserve the previous primary result whenever its CV signal differs. This
    # keeps the raw Schöller-compatible result beside the explicitly requested
    # offline-smoothed analysis variant.
    if stats_path.exists() and (ROOT / "paper" / "table2.tex").exists():
        previous_text = stats_path.read_text(encoding="utf-8")
        previous = json.loads(previous_text)
        previous_mode = previous.get("publication_cv_preprocessing_mode")
        if previous_mode is None and previous.get("rows"):
            previous_mode = previous["rows"][0].get("publication_cv_preprocessing_mode")
        if previous_mode and previous_mode != payload["publication_cv_preprocessing_mode"]:
            _atomic_write_text(
                ROOT / "paper" / "stats" / f"paper_dataset_metrics_{previous_mode}.json",
                previous_text,
            )
            _atomic_write_text(
                ROOT / "paper" / f"table2_{previous_mode}.tex",
                (ROOT / "paper" / "table2.tex").read_text(encoding="utf-8"),
            )
    _atomic_write_text(stats_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(ROOT / "paper" / "table1.tex", table1)
    _atomic_write_text(ROOT / "paper" / "table2.tex", table2)
    mode = payload["publication_cv_preprocessing_mode"]
    _atomic_write_text(
        ROOT / "paper" / "stats" / f"paper_dataset_metrics_{mode}.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(ROOT / "paper" / f"table2_{mode}.tex", table2)
    logger.info("Publication CV preprocessing: %s", mode)
    for row in payload["rows"]:
        logger.info(
            "%s: tracks=%s retained_trajlets=%s cv_windows=%s PASS",
            row["label"], row["source_trajectory_count"],
            row["retained_opentraj_trajlet_count"], row["publication_cv_window_count"],
        )
    if payload["missing"]:
        logger.warning("Partial publication output; missing: " + ", ".join(payload["missing"]))
    return payload


def _configure_logging(log_name: str) -> None:
    """Log to console and to logs/<log_name> so unattended runs keep a record."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / log_name)],
    )


def main() -> None:
    _configure_logging("paper_dataset_metrics.log")
    parser = argparse.ArgumentParser(description="Generate STEPS publication dataset tables")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--cv-preprocessing",
        choices=PUBLICATION_CV_PREPROCESSING_MODES,
        default=None,
        help="Override configs/task.json publication.cv_preprocessing",
    )
    args = parser.parse_args()
    generate_publication_tables(
        allow_partial=args.allow_partial,
        cv_preprocessing_mode=args.cv_preprocessing,
    )


if __name__ == "__main__":
    main()
