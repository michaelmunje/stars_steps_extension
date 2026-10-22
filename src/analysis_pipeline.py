from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from dataclasses import replace
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from analysis_trajlets import AnalysisTrajlet, extract_opentraj_trajlets, index_rows_to_table
from artifact_provenance import implementation_hash, source_set_revision
from canonical import CANONICAL_DT, CANONICAL_HZ, decode_metadata, recording_identity, validate_canonical_file
from data import (
    PER_WINDOW_SPLIT_STRATEGIES,
    PerWindowSplitStrategyUnsupportedError,
    _analysis_index_path,
    _index_path,
    _recordings,
    _split_suffix,
    create_splits,
    prepare_splits,
)
from config import DatasetSpec
from preprocessing_views import OpenTrajKalmanModel
from split_manifests import update_sample_counts, write_manifest


ANALYSIS_VIEW_SCHEMA = pa.schema(
    [
        ("track_id", pa.int32()),
        ("step", pa.int32()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("smoothed_x", pa.float32()),
        ("smoothed_y", pa.float32()),
        ("smoothed_vx", pa.float32()),
        ("smoothed_vy", pa.float32()),
    ]
)
ANALYSIS_CACHE_CONTRACT = "opentraj-analysis-v2"
OPENTRAJ_REFERENCE = "crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a"
ROOT = Path(__file__).resolve().parent
ANALYSIS_INDEX_CONTRACT = "opentraj-analysis-trajlet-index-v1"
ANALYSIS_IMPLEMENTATION_HASH = implementation_hash(
    (
        ROOT / "analysis_pipeline.py",
        ROOT / "analysis_trajlets.py",
        ROOT / "preprocessing_views.py",
    ),
    base=ROOT,
)


def _smooth_track_positions(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pickle-safe deterministic worker for one continuous trajectory segment."""
    return OpenTrajKalmanModel(CANONICAL_DT, n_iter=7).smooth(positions)


def _with_index_metadata(
    table: pa.Table,
    *,
    sources: list[Path],
    dataset_dir: Path,
    purpose: str,
) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"analysis_index_contract": ANALYSIS_INDEX_CONTRACT.encode(),
            b"analysis_purpose": purpose.encode(),
            b"analysis_implementation_hash": ANALYSIS_IMPLEMENTATION_HASH.encode(),
            b"canonical_source_set_revision": source_set_revision(
                sources, base=dataset_dir
            ).encode(),
            b"canonical_source_count": str(len(sources)).encode(),
        }
    )
    return table.replace_schema_metadata(metadata)


def _analysis_index_table(
    trajlets: list[AnalysisTrajlet],
    *,
    sources: list[Path],
    dataset_dir: Path,
    purpose: str,
) -> pa.Table:
    return _with_index_metadata(
        index_rows_to_table([row.as_index_row() for row in trajlets]),
        sources=sources,
        dataset_dir=dataset_dir,
        purpose=purpose,
    )


def attach_trajnet_categories(
    trajlets: list[AnalysisTrajlet], forecast_rows: list[dict]
) -> list[AnalysisTrajlet]:
    """Join the independent categorizations without changing either extraction rule."""
    categories: dict[tuple[str, str, int, int], int] = {}
    for row in forecast_rows:
        key=(str(row["scene_sequence"]),str(row["recording"]),int(row["primary_track_id"]),int(row["start_step"]))
        category=int(row["scene_tag"])
        if key in categories and categories[key] != category:
            raise ValueError(f"Conflicting TrajNet++ categories for {key}: {categories[key]} and {category}")
        categories[key]=category
    return [
        replace(
            row,
            trajnet_category=categories.get(
                (row.scene_sequence,row.recording,row.agent_id,row.source_start_step)
            ),
        )
        for row in trajlets
    ]


def _track_segments(ids: np.ndarray, steps: np.ndarray):
    order = np.lexsort((steps, ids))
    ordered_ids = ids[order]
    for group in np.split(order, np.flatnonzero(np.diff(ordered_ids)) + 1):
        if not len(group):
            continue
        for segment in np.split(group, np.flatnonzero(np.diff(steps[group]) != 1) + 1):
            if len(segment):
                yield int(ids[segment[0]]), segment


def build_analysis_view(
    source: Path,
    destination: Path,
    *,
    progress_every: int = 0,
    track_workers: int = 1,
) -> tuple[Path, list[AnalysisTrajlet], dict]:
    """Create a non-destructive OpenTraj smoothing view and exact trajlet index rows."""
    identity = validate_canonical_file(source)
    source_stat=source.stat();source_revision=f"{source_stat.st_size}:{source_stat.st_mtime_ns}"
    if destination.exists():
        cached_metadata=decode_metadata(pq.ParquetFile(destination).schema_arrow)
        pinned_cache = (
            cached_metadata.get("canonical_source_revision") == source_revision
            and cached_metadata.get("analysis_preprocessing") == "opentraj_kalman_smooth"
            and cached_metadata.get("opentraj_reference") == OPENTRAJ_REFERENCE
        )
        contract = cached_metadata.get("analysis_cache_contract")
        if pinned_cache and contract in {None, ANALYSIS_CACHE_CONTRACT}:
            cached=pq.read_table(destination)
            if contract is None:
                # One-time provenance upgrade for analysis values already
                # verified numerically against the same pinned OpenTraj code.
                upgraded = dict(cached_metadata)
                upgraded["analysis_cache_contract"] = ANALYSIS_CACHE_CONTRACT
                cached = cached.replace_schema_metadata(
                    {key.encode(): value.encode() for key, value in upgraded.items()}
                )
                temporary = destination.with_suffix(".contract.tmp.parquet")
                pq.write_table(cached, temporary, compression="zstd")
                temporary.replace(destination)
            ids=cached["track_id"].combine_chunks().to_numpy();steps=cached["step"].combine_chunks().to_numpy()
            smooth_x=cached["smoothed_x"].combine_chunks().to_numpy();smooth_y=cached["smoothed_y"].combine_chunks().to_numpy()
            trajlets=[];n_segments=0
            for agent_id,segment in _track_segments(ids,steps):
                n_segments+=1
                if progress_every and n_segments % progress_every == 0:
                    print(f"  {source.name}: reconstructed {n_segments:,} cached track segments", flush=True)
                trajlets.extend(extract_opentraj_trajlets(dataset=identity.dataset,scene_id=identity.scene_id,scene_sequence=identity.scene_sequence,recording=identity.recording,agent_id=agent_id,steps=steps[segment],xy=np.c_[smooth_x[segment],smooth_y[segment]],sample_hz=CANONICAL_HZ))
            stats={"recording":identity.recording,"scene_id":identity.scene_id,"scene_sequence":identity.scene_sequence,"full_tracks":len(np.unique(ids)),"continuous_segments":n_segments,"valid_opentraj_trajlets":len(trajlets),"opentraj_static_filtered_trajlets":sum(row.opentraj_static for row in trajlets),"retained_analysis_trajlets":sum(row.retained_after_static_filter for row in trajlets),"analysis_view_cache_reused":True}
            return destination,trajlets,stats
    table = pq.read_table(source, columns=["track_id", "step", "x", "y"])
    ids = table["track_id"].combine_chunks().to_numpy()
    steps = table["step"].combine_chunks().to_numpy()
    x = table["x"].combine_chunks().to_numpy()
    y = table["y"].combine_chunks().to_numpy()
    smooth_xy = np.full((len(ids), 2), np.nan, dtype=np.float32)
    smooth_v = np.full((len(ids), 2), np.nan, dtype=np.float32)
    trajlets: list[AnalysisTrajlet] = []
    n_segments = 0
    segments = list(_track_segments(ids, steps))
    positions = [np.c_[x[segment], y[segment]].astype(float) for _, segment in segments]
    executor: ProcessPoolExecutor | None = None
    try:
        if track_workers > 1 and len(segments) > 1:
            context = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(
                max_workers=min(track_workers, len(segments)),
                mp_context=context,
            )
            smoothed_results = executor.map(
                _smooth_track_positions,
                positions,
                chunksize=16,
            )
        else:
            smoothed_results = map(_smooth_track_positions, positions)
        for (agent_id, segment), (smoothed, velocity) in zip(segments, smoothed_results):
            n_segments += 1
            if progress_every and n_segments % progress_every == 0:
                print(
                    f"  {source.name}: smoothed {n_segments:,}/{len(segments):,} track segments",
                    flush=True,
                )
            smooth_xy[segment] = smoothed
            smooth_v[segment] = velocity
            trajlets.extend(
                extract_opentraj_trajlets(
                    dataset=identity.dataset,
                    scene_id=identity.scene_id,
                    scene_sequence=identity.scene_sequence,
                    recording=identity.recording,
                    agent_id=agent_id,
                    steps=steps[segment],
                    xy=smoothed,
                    sample_hz=CANONICAL_HZ,
                )
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if progress_every:
        print(f"  {source.name}: smoothed {n_segments:,}/{len(segments):,} track segments", flush=True)
    metadata = decode_metadata(table.schema)
    metadata.update(
        {
            "analysis_preprocessing": "opentraj_kalman_smooth",
            "analysis_preprocessing_scope": "full continuous trajectory segment (non-causal)",
            "opentraj_reference": OPENTRAJ_REFERENCE,
            "analysis_cache_contract": ANALYSIS_CACHE_CONTRACT,
            "canonical_source": str(source),
            "canonical_source_revision":source_revision,
        }
    )
    schema = ANALYSIS_VIEW_SCHEMA.with_metadata(
        {key.encode(): value.encode() for key, value in metadata.items()}
    )
    result = pa.Table.from_arrays(
        [
            pa.array(ids, pa.int32()),
            pa.array(steps, pa.int32()),
            pa.array(x, pa.float32()),
            pa.array(y, pa.float32()),
            pa.array(smooth_xy[:, 0], pa.float32()),
            pa.array(smooth_xy[:, 1], pa.float32()),
            pa.array(smooth_v[:, 0], pa.float32()),
            pa.array(smooth_v[:, 1], pa.float32()),
        ],
        schema=schema,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    pq.write_table(result, temporary, compression="zstd")
    temporary.replace(destination)
    unique_tracks = len(np.unique(ids))
    stats = {
        "recording": identity.recording,
        "scene_id": identity.scene_id,
        "scene_sequence": identity.scene_sequence,
        "full_tracks": unique_tracks,
        "continuous_segments": n_segments,
        "valid_opentraj_trajlets": len(trajlets),
        "opentraj_static_filtered_trajlets": sum(row.opentraj_static for row in trajlets),
        "retained_analysis_trajlets": sum(row.retained_after_static_filter for row in trajlets),
        "analysis_view_cache_reused":False,
    }
    return destination, trajlets, stats


def build_analysis_dataset(dataset_dir: Path, *, data_seed: int = 42,
                           split_strategy: str = "grouped_scene_sequence",
                           held_out_scene: str | None = None,
                           workers: int = 1) -> tuple[Path, Path, dict]:
    files = _recordings(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No canonical recordings under {dataset_dir / 'converted'}")
    prepare_splits(dataset_dir, data_seed, split_strategy, held_out_scene)
    split_config=create_splits(dataset_dir,data_seed,split_strategy,held_out_scene)
    results: dict[int, tuple[Path, list[AnalysisTrajlet], dict]] = {}
    if workers == 1 or len(files) == 1:
        iterator = enumerate(files)
        for index, source in tqdm(
            iterator,
            total=len(files),
            desc=f"OpenTraj analysis views: {dataset_dir.name}",
            unit="recording",
        ):
            name = source.name.removesuffix(".traj.parquet")
            destination = dataset_dir / "analysis" / "opentraj_kalman_smooth" / f"{name}.analysis.parquet"
            results[index] = build_analysis_view(
                source, destination, progress_every=1_000,
                track_workers=workers if len(files) == 1 else 1,
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            pending = {}
            for index, source in enumerate(files):
                name = source.name.removesuffix(".traj.parquet")
                destination = dataset_dir / "analysis" / "opentraj_kalman_smooth" / f"{name}.analysis.parquet"
                pending[executor.submit(build_analysis_view, source, destination)] = index
            with tqdm(
                total=len(pending),
                desc=f"OpenTraj analysis views: {dataset_dir.name}",
                unit="recording",
            ) as progress:
                for future in as_completed(pending):
                    results[pending[future]] = future.result()
                    progress.update()

    if split_config.get("strategy") in PER_WINDOW_SPLIT_STRATEGIES:
        raise PerWindowSplitStrategyUnsupportedError(
            f"build_analysis_dataset cannot express split_strategy={split_config['strategy']!r}: "
            f"it assigns train/val/test per window within a recording, but "
            f"split_by_recording is empty here because every record carries "
            f"split={split_config['strategy']!r} rather than a per-recording train/val/test "
            f"label. Use split_strategy='grouped_scene_sequence' or 'leave_one_scene_out' for "
            f"OpenTraj analysis views instead."
        )
    all_trajlets: list[AnalysisTrajlet] = []
    recording_stats = []
    for index, source in enumerate(files):
        _, trajlets, stats = results[index]
        split=split_config["split_by_recording"][stats["recording"]]
        all_trajlets.extend(replace(row,split=split) for row in trajlets)
        recording_stats.append(stats)
    forecast_path=_index_path(DatasetSpec(dataset_dir.name,dataset_dir),data_seed,split_strategy,held_out_scene)
    forecast_rows=[]
    if forecast_path.exists():
        forecast_rows=pq.read_table(
            forecast_path,
            columns=["scene_sequence","recording","primary_track_id","start_step","scene_tag"],
        ).to_pylist()
        all_trajlets=attach_trajnet_categories(all_trajlets,forecast_rows)
    index_path = _analysis_index_path(dataset_dir,split_strategy,held_out_scene)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(".tmp.parquet")
    # Materialized once and reused for both the index table and the manifest counts.
    index_rows = [row.as_index_row() for row in all_trajlets]
    pq.write_table(
        _with_index_metadata(
            index_rows_to_table(index_rows),
            sources=files,
            dataset_dir=dataset_dir,
            purpose="forecast_split_analysis",
        ),
        temporary,
        compression="zstd",
    )
    temporary.replace(index_path)
    manifest_path=prepare_splits(dataset_dir,data_seed,split_strategy,held_out_scene);manifest=json.loads(manifest_path.read_text())
    manifest=update_sample_counts(manifest,index_rows,forecast_rows)
    write_manifest(manifest,manifest_path)
    totals = Counter()
    for row in recording_stats:
        for key in ("full_tracks", "continuous_segments", "valid_opentraj_trajlets", "opentraj_static_filtered_trajlets", "retained_analysis_trajlets"):
            totals[key] += row[key]
    # One pass over the trajlets supplies every per-split tally in the summary.
    split_tally: Counter[tuple[str | None, bool, bool]] = Counter()
    trajnet_categories_attached = 0
    for row in all_trajlets:
        split_tally[(row.split, row.opentraj_static, row.retained_after_static_filter)] += 1
        trajnet_categories_attached += row.trajnet_category is not None
    flags = (False, True)
    summary = {
        "dataset": dataset_dir.name,
        "data_seed": data_seed,
        "split_strategy": split_strategy,
        "held_out_scene": held_out_scene,
        "analysis_preprocessing": "opentraj_kalman_smooth",
        "canonical_hz": CANONICAL_HZ,
        "trajlet_length_s": 4.8,
        "trajlet_overlap_s": 2.0,
        "returned_samples": 12,
        "static_endpoint_offset_samples": 12,
        "candidate_stride_samples": 7,
        "trajnet_categories_attached": trajnet_categories_attached,
        "split_counts": {
            split: {
                "valid_opentraj_trajlets": sum(split_tally[(split,static,retained)] for static in flags for retained in flags),
                "opentraj_static_filtered_trajlets": sum(split_tally[(split,True,retained)] for retained in flags),
                "retained_analysis_trajlets": sum(split_tally[(split,static,True)] for static in flags),
            }
            for split in ("train","val","test")
        },
        **totals,
        "recordings": recording_stats,
    }
    summary_suffix = _split_suffix(split_strategy, held_out_scene)
    summary_path = dataset_dir / "analysis" / f"opentraj_summary{summary_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return index_path, summary_path, summary


def _publication_index_suffix(artifact_variant: str) -> str:
    return "" if artifact_variant == "default" else f"-{artifact_variant}"


def _forecasting_collision(artifact_variant: str) -> str | None:
    """Return the colliding forecasting split strategy, or ``None`` if disjoint.

    The publication writer and ``data._analysis_index_path`` (the forecasting
    analysis index) share the same ``indices/opentraj-analysis-trajlets{suffix}``
    stem. The forbidden suffix set is derived here by calling
    ``data._split_suffix`` directly rather than restated as a literal list, so a
    fifth split strategy (or a change to the sanitisation rule) cannot silently
    reopen the collision.
    """
    candidate = _publication_index_suffix(artifact_variant)
    for strategy in ("grouped_scene_sequence", "temporal", "held_out_recording"):
        if _split_suffix(strategy, None) == candidate:
            return strategy
    if candidate.startswith("-loso-"):
        held_out_scene = candidate[len("-loso-"):]
        if held_out_scene and _split_suffix("leave_one_scene_out", held_out_scene) == candidate:
            return f"leave_one_scene_out/{held_out_scene}"
    return None


def _validated_artifact_variant(value: str) -> str:
    """argparse ``type=`` for ``--artifact-variant``: reject values that would
    collide with the forecasting analysis index namespace (see
    ``_forecasting_collision``). Cannot be expressed as ``choices=`` because the
    ``leave_one_scene_out`` family (``loso-<scene>``) is unbounded.
    """
    collision = _forecasting_collision(value)
    if collision is not None:
        raise argparse.ArgumentTypeError(
            f"--artifact-variant {value!r} would collide with the forecasting analysis index "
            f"for split strategy {collision!r}; choose a variant name that does not overlap "
            "with data._split_suffix's output (e.g. 'publication')"
        )
    return value


def build_publication_analysis_dataset(
    dataset_dir: Path,
    *,
    scene_sequences: set[str] | None = None,
    artifact_variant: str = "publication",
    workers: int = 1,
) -> tuple[Path, Path, dict]:
    """Build exact analysis artifacts for publication rather than model membership."""
    collision = _forecasting_collision(artifact_variant)
    if collision is not None:
        raise ValueError(
            f"artifact_variant={artifact_variant!r} would write to the same analysis index path "
            f"as the forecasting split strategy {collision!r} (data._analysis_index_path); "
            "choose a variant whose suffix does not overlap with data._split_suffix's output"
        )
    files = sorted(
        path for path in (dataset_dir / "converted").rglob("*.traj.parquet")
        if "Test_set" not in path.parts and "Test_set" not in path.name
    )
    if scene_sequences is not None:
        files = [path for path in files if recording_identity(path).scene_sequence in scene_sequences]
    if not files:
        raise FileNotFoundError(f"No selected canonical publication recordings in {dataset_dir}")
    identities = [recording_identity(path) for path in files]
    selected = {identity.scene_sequence for identity in identities}
    recording_keys = [(identity.scene_sequence, identity.recording) for identity in identities]
    if len(recording_keys) != len(set(recording_keys)):
        raise ValueError("Canonical publication files contain duplicate recording identities")
    if scene_sequences is not None and selected != scene_sequences:
        raise FileNotFoundError(f"Missing publication sequences: {sorted(scene_sequences-selected)}")

    results: dict[int, tuple[Path, list[AnalysisTrajlet], dict]] = {}
    if workers == 1 or len(files) == 1:
        for index, source in tqdm(enumerate(files), total=len(files), desc=f"OpenTraj publication views: {dataset_dir.name}", unit="recording"):
            name = source.name.removesuffix(".traj.parquet")
            destination = dataset_dir / "analysis" / "opentraj_kalman_smooth" / f"{name}.analysis.parquet"
            results[index] = build_analysis_view(
                source, destination, progress_every=1_000,
                track_workers=workers if len(files) == 1 else 1,
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            pending = {}
            for index, source in enumerate(files):
                name = source.name.removesuffix(".traj.parquet")
                destination = dataset_dir / "analysis" / "opentraj_kalman_smooth" / f"{name}.analysis.parquet"
                pending[executor.submit(build_analysis_view, source, destination)] = index
            with tqdm(total=len(pending), desc=f"OpenTraj publication views: {dataset_dir.name}", unit="recording") as progress:
                for future in as_completed(pending):
                    results[pending[future]] = future.result()
                    progress.update()

    all_trajlets: list[AnalysisTrajlet] = []
    recording_stats = []
    for index in range(len(files)):
        _, trajlets, stats = results[index]
        all_trajlets.extend(replace(row, split="publication") for row in trajlets)
        recording_stats.append(stats)
    index_path = dataset_dir / "indices" / f"opentraj-analysis-trajlets{_publication_index_suffix(artifact_variant)}.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(".tmp.parquet")
    pq.write_table(
        _analysis_index_table(
            all_trajlets,
            sources=files,
            dataset_dir=dataset_dir,
            purpose="publication_dataset_characterization",
        ),
        temporary,
        compression="zstd",
    )
    temporary.replace(index_path)
    summary = {
        "dataset": dataset_dir.name,
        "purpose": "publication_dataset_characterization",
        "analysis_preprocessing": "opentraj_kalman_smooth",
        "canonical_hz": CANONICAL_HZ,
        "included_scene_sequences": sorted(selected),
        "valid_opentraj_trajlets": len(all_trajlets),
        "retained_analysis_trajlets": sum(row.retained_after_static_filter for row in all_trajlets),
        "recordings": recording_stats,
    }
    summary_path = dataset_dir / "analysis" / f"opentraj_summary_{artifact_variant}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return index_path, summary_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build non-causal OpenTraj analysis views and trajlets")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--data-seed", type=int, default=42)
    # Deliberately a two-element subset of config.SPLIT_STRATEGIES, not the full
    # four: build_analysis_dataset explicitly refuses "temporal" and
    # "held_out_recording" (see its guard around analysis_pipeline.py:321-325),
    # so rejecting them here at parse time instead of widening to match is correct.
    parser.add_argument("--split-strategy",choices=("grouped_scene_sequence","leave_one_scene_out"),default="grouped_scene_sequence")
    parser.add_argument("--held-out-scene")
    parser.add_argument("--publication", action="store_true", help="Use explicit publication membership, independent of forecasting folds")
    parser.add_argument("--scene-sequences", nargs="+")
    parser.add_argument("--artifact-variant", default="publication", type=_validated_artifact_variant)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Independent recording-level smoothing workers (default: capped at 8)",
    )
    args = parser.parse_args()
    if args.publication:
        index, summary, counts = build_publication_analysis_dataset(
            args.dataset_dir,
            scene_sequences=set(args.scene_sequences) if args.scene_sequences else None,
            artifact_variant=args.artifact_variant,
            workers=max(1,args.workers),
        )
    else:
        index, summary, counts = build_analysis_dataset(args.dataset_dir, data_seed=args.data_seed,split_strategy=args.split_strategy,held_out_scene=args.held_out_scene,workers=max(1,args.workers))
    print(index)
    print(summary)
    print(json.dumps({key: value for key, value in counts.items() if key != "recordings"}, indent=2))


if __name__ == "__main__":
    main()
