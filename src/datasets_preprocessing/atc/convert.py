from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
import subprocess
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from canonical import CANONICAL_HZ, CANONICAL_SCHEMA as OUTPUT_SCHEMA, canonical_metadata, schema_with_metadata
from artifact_provenance import git_provenance


SAMPLE_HZ = CANONICAL_HZ
POSITION_SCALE = 0.001
# Number of buffered rows (across all in-flight tracks) that triggers an
# intermediate flush() call to the temp Parquet file during conversion.
FLUSH_ROW_THRESHOLD = 100_000
# ATC uses one fixed map coordinate frame. This center was measured across all
# ten recordings in the first official archive and is reused for every date.
SCENE_CENTER_M = (3.7245000000000026, -2.5200000000000014)
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RawSource:
    container: Path
    member: str | None = None

    @property
    def name(self) -> str:
        return Path(self.member).name if self.member else self.container.name

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    def __str__(self) -> str:
        return f"{self.container}:{self.member}" if self.member else str(self.container)


def _archive_members(path: Path) -> list[str]:
    result = subprocess.run(
        ["7z", "l", "-slt", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    members = []
    for line in result.stdout.splitlines():
        if line.startswith("Path = "):
            member = line.removeprefix("Path = ")
            if Path(member).name.startswith("atc-") and member.lower().endswith(".csv"):
                members.append(member)
    return members


def raw_files(dataset_dir: Path, raw_dir: Path | None = None) -> list[RawSource]:
    root = raw_dir or dataset_dir / "raw"
    extracted = sorted(root.rglob("atc-*.csv"))
    sources = [RawSource(path) for path in extracted]
    names = {path.name for path in extracted}
    for archive in sorted(root.glob("*.7z")):
        for member in _archive_members(archive):
            if Path(member).name not in names:
                sources.append(RawSource(archive, member))
                names.add(Path(member).name)
    return sorted(sources, key=lambda source: source.name)


@contextmanager
def _open_source(source: RawSource):
    if source.member is None:
        with source.container.open(newline="", encoding="utf-8") as file:
            yield file
        return

    process = subprocess.Popen(
        ["7z", "x", "-so", str(source.container), source.member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    stream = io.TextIOWrapper(process.stdout, encoding="utf-8", newline="")
    try:
        yield stream
    finally:
        stream.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"7z failed while reading {source}: {stderr.strip()}")


def _rows(path: RawSource):
    with _open_source(path) as file:
        for line_number, row in enumerate(csv.reader(file), 1):
            if not row:
                continue
            timestamp = float(row[0])
            track_id = int(row[1])
            raw_x = float(row[2])
            raw_y = float(row[3])
            heading = float(row[6])
            if not (math.isfinite(raw_x) and math.isfinite(raw_y) and math.isfinite(heading)):
                raise ValueError(
                    f"{path}:{line_number}: raw_x, raw_y, and heading must be finite "
                    f"(got raw_x={raw_x}, raw_y={raw_y}, heading={heading})"
                )
            yield timestamp, track_id, raw_x, raw_y, heading


def _table(rows: list[tuple[int, int, float, float, float]]) -> pa.Table:
    rows.sort(key=lambda item: (item[0], item[1]))
    return pa.Table.from_arrays(
        [
            pa.array([row[1] for row in rows], type=pa.int32()),
            pa.array([row[0] for row in rows], type=pa.int32()),
            pa.array([row[2] for row in rows], type=pa.float32()),
            pa.array([row[3] for row in rows], type=pa.float32()),
            pa.array([row[4] for row in rows], type=pa.float32()),
        ],
        schema=OUTPUT_SCHEMA,
    )


def convert_file(
    path: RawSource,
    output: Path,
    center: tuple[float, float],
    *,
    show_progress: bool = True,
    provenance: dict[str, object] | None = None,
) -> None:
    print(f"Converting: {path} -> {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    metadata = canonical_metadata(dataset="atc",scene_id="atc_map",scene_sequence=path.stem,recording=path.stem,extra={
        "source_to_local_3x3": json.dumps(
            [[POSITION_SCALE, 0.0, -center[0]], [0.0, POSITION_SCALE, -center[1]], [0.0, 0.0, 1.0]]
        ),
        "heading_transform": "Add the planar rotation from source_to_local_3x3 and wrap to [-pi, pi]; this ATC transform is translation-only.",
    })
    schema = schema_with_metadata(metadata)
    first_time: float | None = None
    current_step: int | None = None
    current: dict[int, tuple[int, int, float, float, float]] = {}
    current_error: dict[int, float] = {}
    pending: list[tuple[int, int, float, float, float]] = []
    writer: pq.ParquetWriter | None = None
    delta_histogram: Counter[int] = Counter()
    last_step_by_track: dict[int, int] = {}
    tracks_with_gaps: set[int] = set()
    aligned_rows = emitted_rows = 0

    def flush() -> None:
        nonlocal writer, aligned_rows, emitted_rows
        if not pending:
            return
        if writer is None:
            writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        ordered = sorted(pending, key=lambda item: (item[0], item[1]))
        for step, track_id, *_ in ordered:
            emitted_rows += 1
            aligned_rows += 1
            previous = last_step_by_track.get(track_id)
            if previous is not None:
                delta = step - previous
                delta_histogram[delta] += 1
                if delta != 1:
                    tracks_with_gaps.add(track_id)
            last_step_by_track[track_id] = step
        writer.write_table(_table(ordered))
        pending.clear()

    progress = tqdm(
        desc=f"converting {path.name}",
        unit="rows",
        unit_scale=True,
        disable=not show_progress,
    )
    try:
        for row_count, (timestamp, track_id, raw_x, raw_y, heading) in enumerate(_rows(path), 1):
            if first_time is None:
                first_time = timestamp
            step = int(round((timestamp - first_time) * SAMPLE_HZ))
            if current_step is not None and step < current_step:
                raise ValueError(f"timestamps are not ordered in {path}")
            if current_step is not None and step != current_step:
                pending.extend(current.values())
                current.clear()
                current_error.clear()
                if len(pending) >= FLUSH_ROW_THRESHOLD:
                    flush()
            current_step = step
            elapsed=timestamp-first_time; alignment_error=abs(elapsed-step/SAMPLE_HZ)
            if track_id not in current_error or alignment_error < current_error[track_id]:
                current_error[track_id]=alignment_error
                current[track_id] = (
                    step,
                    track_id,
                    raw_x * POSITION_SCALE - center[0],
                    raw_y * POSITION_SCALE - center[1],
                    heading,
                )
            progress.update()
        if current:
            pending.extend(current.values())
        flush()
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()
            temporary.replace(output)
    finally:
        progress.close()

    if first_time is None:
        raise ValueError(f"ATC raw file contains no rows: {path}")
    diagnostics = {
        "recording": path.stem,
        "emitted_rows": emitted_rows,
        "fraction_stride_aligned": aligned_rows / emitted_rows,
        "step_delta_histogram": {str(key): value for key, value in sorted(delta_histogram.items())},
        "tracks": len(last_step_by_track),
        "tracks_with_gaps": len(tracks_with_gaps),
        **(provenance or git_provenance(REPO_ROOT)),
    }
    output.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def _convert_job(job: tuple[RawSource, Path, tuple[float, float], dict[str, object]]) -> Path:
    source, output, center, provenance = job
    # Per-row progress bars from concurrent workers cannot share a terminal
    # line usefully; rely on the aggregate "N/M recordings" bar instead.
    convert_file(source, output, center, show_progress=False, provenance=provenance)
    return output


def convert_dataset(dataset_dir: Path, raw_dir: Path | None = None, workers: int = 1) -> list[Path]:
    paths = raw_files(dataset_dir, raw_dir)
    if not paths:
        raise FileNotFoundError(f"No ATC CSV files found in {raw_dir or dataset_dir / 'raw'}")
    center = SCENE_CENTER_M
    # Resolved once per conversion run so every recording's diagnostics sidecar
    # records the same commit/dirty snapshot, rather than one git call per file.
    provenance = git_provenance(REPO_ROOT)
    jobs = [
        (path, dataset_dir / "converted" / f"{path.stem}.traj.parquet", center, provenance)
        for path in paths
    ]
    if workers <= 1:
        outputs = []
        with tqdm(total=len(jobs), desc="ATC recordings", unit="recording") as progress:
            for path, output, job_center, job_provenance in jobs:
                convert_file(path, output, job_center, show_progress=True, provenance=job_provenance)
                outputs.append(output)
                progress.update()
        return outputs
    outputs: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_convert_job, job) for job in jobs]
        with tqdm(total=len(futures), desc="ATC recordings", unit="recording") as progress:
            for future in as_completed(futures):
                try:
                    outputs.append(future.result())
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise
                progress.update()
    return sorted(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ATC CSV files to canonical parquet")
    parser.add_argument("dataset_dir", type=Path, default=Path(__file__).resolve().parents[3]/"datasets/atc", nargs="?")
    parser.add_argument("--raw-dir", type=Path, help="directory containing atc-*.csv files")
    parser.add_argument("--workers", type=int, default=1, help="recordings to convert concurrently")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    for output in convert_dataset(args.dataset_dir, args.raw_dir, args.workers):
        print(output)


if __name__ == "__main__":
    main()
