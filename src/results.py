from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


RUN_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("model", pa.string()),
        ("target_dataset", pa.string()),
        ("steps_fraction", pa.float64()),
        ("target_fraction", pa.float64()),
        ("data_seed", pa.int64()),
        ("model_seed", pa.int64()),
        ("pretrain_source", pa.string()),
        ("n_train_scenes", pa.int64()),
        ("n_train_tracks", pa.int64()),
        ("iterations", pa.int64()),
        ("ade", pa.float64()),
        ("fde", pa.float64()),
        ("top3_ade", pa.float64()),
        ("top3_fde", pa.float64()),
        ("k", pa.int64()),
        ("pred_col", pa.float64()),
        ("gt_col", pa.float64()),
        *((f"ade_type{i}", pa.float64()) for i in range(1, 5)),
        *((f"fde_type{i}", pa.float64()) for i in range(1, 5)),
        ("config_hash", pa.string()),
        ("canonical_release", pa.string()),
        ("canonical_schema_version", pa.int64()),
        ("analysis_preprocessing", pa.string()),
        ("forecast_preset", pa.string()),
        ("forecast_preset_modified", pa.bool_()),
        ("forecast_literature_exact", pa.bool_()),
        ("forecast_input_preprocessing", pa.string()),
        ("forecast_target_preprocessing", pa.string()),
        ("model_normalization", pa.string()),
        ("coordinate_frame", pa.string()),
        ("feature_representation", pa.string()),
        ("derivative_mode", pa.string()),
        ("standardization_mode", pa.string()),
        ("augmentation_rotation", pa.bool_()),
        ("augmentation_flip", pa.bool_()),
        ("augmentation_scale", pa.bool_()),
        ("scene_membership", pa.string()),
        ("min_complete_agents", pa.int64()),
        ("forecast_preprocessing_hash", pa.string()),
        ("upstream_repository", pa.string()),
        ("upstream_pinned_commit", pa.string()),
        ("split_manifest_hash", pa.string()),
        ("normalization_stats_path", pa.string()),
        ("history_steps", pa.int64()),
        ("future_steps", pa.int64()),
        ("sample_hz", pa.float64()),
        ("canonical_stride", pa.int64()),
        ("chunk_stride", pa.string()),
        ("min_length", pa.float64()),
        ("static_threshold", pa.float64()),
        ("linear_threshold", pa.float64()),
        ("inter_dist_thresh", pa.float64()),
        ("inter_pos_range", pa.float64()),
        ("grp_dist_thresh", pa.float64()),
        ("grp_std_thresh", pa.float64()),
        *((f"acceptance_type{i}", pa.float64()) for i in range(1, 5)),
        ("w_target", pa.float64()),
        ("n_target", pa.int64()),
        ("batch_size", pa.int64()),
        ("num_workers", pa.int64()),
        ("train_rotation", pa.bool_()),
        ("eval_stride", pa.int64()),
        # Git commit/uncommitted-change status captured by artifact_provenance.git_provenance()
        # (F-007); git_dirty is nullable because provenance resolution degrades to "unknown"
        # rather than raising outside a git checkout.
        ("git_commit", pa.string()),
        ("git_dirty", pa.bool_()),
    ]
)

# Derived so the frozen column list and the Arrow schema cannot drift: pa.Table.from_pylist
# silently drops keys absent from the schema and silently nulls schema fields absent from the
# record, so a duplicated literal list would hide either direction of divergence.
RUN_COLUMNS = tuple(RUN_SCHEMA.names)

# Derived the same way, for the same reason (M-015): every schema field typed as a floating
# column is a candidate for a silent NaN/inf write, not just the handful of ADE/FDE metrics
# named in the original report. A hand-copied list drifts from the schema exactly like
# RUN_COLUMNS would (see M-005, M-013).
FLOAT_COLUMNS = tuple(field.name for field in RUN_SCHEMA if pa.types.is_floating(field.type))


def append_run(path: Path, record: dict[str, Any]) -> None:
    """Append a single run record to the frozen-schema runs.parquet at ``path``.

    Concurrency note (M-016, partial fix only): the temp file used for the atomic
    write is now named uniquely per call (pid + a random uuid4 suffix), which removes
    the *shared*-temp-file hazard where two concurrent writers could clobber each
    other's temp file before either replace() ran. It does **not** make this function
    safe for concurrent use. The read-existing/concat/write-new sequence is still a
    read-modify-write race: two callers that both read the same existing runs.parquet
    will each append to that same snapshot, and whichever replace() lands second wins,
    silently discarding the run appended by the one that landed first. No locking is
    implemented here (that requires a decision that has not been made). Callers must
    still enforce single-writer discipline around this file.

    Nullable float columns (see ``FLOAT_COLUMNS``) encode "not measured" as ``None``.
    ``NaN``/``inf`` are rejected unconditionally, for every float column, with no
    per-column exemptions: this is deliberate (M-015) and is what stops a silently
    wrong metric from being recorded. A caller that has no value for a per-type metric
    (e.g. no scored test scenes of a given type) must write ``None``, not
    ``float("nan")``.

    Temp-file cleanup on failure is best-effort, not a guarantee: ``replace()`` is the
    atomic commit point, but a hard kill of the process between ``pq.write_table`` and
    ``replace`` can still leave an orphaned ``*.tmp.*.parquet`` file behind.
    """
    missing = set(RUN_COLUMNS) - record.keys()
    if missing:
        raise ValueError(f"Run record is missing columns: {sorted(missing)}")
    extra = record.keys() - set(RUN_COLUMNS)
    if extra:
        raise ValueError(f"Run record has unknown columns: {sorted(extra)}")
    for field in FLOAT_COLUMNS:
        value = record[field]
        if value is not None and not math.isfinite(value):
            raise ValueError(
                f"Run record has non-finite {field}={value!r} for run_id={record['run_id']}"
            )
    row = pa.Table.from_pylist([record], schema=RUN_SCHEMA)
    if path.exists():
        existing = pq.read_table(path)
        if existing.schema != RUN_SCHEMA:
            raise ValueError("Existing runs.parquet does not match the frozen run schema")
        if record["run_id"] in existing["run_id"].to_pylist():
            raise ValueError(f"run_id already exists: {record['run_id']}")
        row = pa.concat_tables([existing, row])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex}.parquet")
    try:
        pq.write_table(row, temporary, compression="zstd")
        temporary.replace(path)
    except BaseException:
        # Cleanup must never mask the failure that triggered it: if unlink() itself
        # raises (e.g. a permission error), swallowing that keeps the original
        # write/replace exception as the one the caller sees.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
