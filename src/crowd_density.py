from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from config import Config, DatasetSpec, SeedConfig
from data import SPLITS, SceneDataset, _scene_arrays
from dataset_paths import PAPER_STATS_ROOT


def window_agent_counts(dataset: SceneDataset) -> np.ndarray:
    """Agents present at each window's final observed frame, in ``dataset.rows`` order.

    Batched counterpart of ``dataset[index].valid[history_steps - 1].sum()``:
    each recording is read once and every window of that recording is cut out of
    the in-memory arrays, mirroring the idiom in ``stats.analyze``. ``Scene.valid``
    is ``_scene_arrays(...)["valid"]`` verbatim and ``prepare_forecast_representation``
    only reads it, so the counts are identical.

    Unlike ``SceneDataset.__getitem__`` this path never calls
    ``prepare_forecast_representation``, so representation errors that a pathological
    window would raise there do not surface here; only membership is counted.

    Datasets with generic (non-``target_complete``) scene membership keep using the
    per-scene path; see the comment below.
    """
    observation_index = dataset.history_steps - 1
    if dataset.generic_membership:
        # Generic membership resolves each window from its precomputed
        # ``window_record`` rather than from the raw recording. The recording still
        # holds non-member agents, so a batched count over it would exceed the
        # membership-filtered one; fall back to the per-scene path, which is the
        # only source of that filtering.
        return np.fromiter(
            (
                int(dataset[index].valid[observation_index].sum())
                for index in tqdm(
                    range(len(dataset)),
                    desc=f"crowd density: {dataset.spec.name}/{dataset.split}",
                    unit="scene",
                    mininterval=1.0,
                )
            ),
            dtype=np.int64,
            count=len(dataset),
        )
    window_span = (dataset.total_steps - 1) * dataset.canonical_stride
    rows_by_recording: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        rows_by_recording[row["recording"]].append(index)

    counts = np.zeros(len(dataset.rows), dtype=np.int64)
    progress = tqdm(
        total=len(dataset.rows),
        desc=f"crowd density: {dataset.spec.name}/{dataset.split}",
        unit="scene",
        mininterval=1.0,
    )
    for recording, indices in rows_by_recording.items():
        path = dataset.recordings[recording]
        table = pq.read_table(path, columns=["track_id", "step", "x", "y"])
        ids, steps, x, y = (table[c].combine_chunks().to_numpy() for c in ("track_id", "step", "x", "y"))
        # Windows are cut with np.searchsorted, which requires file-order steps to be
        # sorted. validate_canonical_file rejects unsorted files, and SceneDataset.__init__
        # runs it for every recording, so this is re-asserted here per read (rather than
        # assumed) in case a future caller bypasses SceneDataset and reaches this path
        # with an unvalidated file, which would otherwise silently mis-slice.
        if len(steps) > 1 and not np.all(np.diff(steps) >= 0):
            raise ValueError(f"{path} stores steps out of order; canonical recordings must be step-sorted")
        for index in indices:
            row = dataset.rows[index]
            start_step = int(row["start_step"])
            low = np.searchsorted(steps, start_step, side="left")
            high = np.searchsorted(steps, start_step + window_span, side="right")
            arrays = _scene_arrays(
                ids[low:high], steps[low:high], x[low:high], y[low:high], start_step,
                dataset.history_steps, dataset.future_steps, dataset.canonical_stride,
                int(row["primary_track_id"]),
            )
            counts[index] = int(arrays["valid"][observation_index].sum())
            progress.update()
    progress.close()
    return counts


def crowd_density(dataset_dir: Path, split: str, seed: int) -> dict:
    """Report agents visible to the model at the final observation timestep."""
    cfg = Config(seeds=SeedConfig(seed, seed), train_rotation=False)
    dataset = SceneDataset(DatasetSpec(dataset_dir.name, dataset_dir), split, cfg)
    counts = window_agent_counts(dataset)
    return {
        "definition": "agents per primary-agent window present at the final observed frame",
        "scenes": len(counts),
        "mean": float(counts.mean()) if len(counts) else 0.0,
        "std": float(counts.std()) if len(counts) else 0.0,
        "min": int(counts.min()) if len(counts) else 0,
        "max": int(counts.max()) if len(counts) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report model-visible pedestrians per scene window")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path holding all queried splits; defaults to "
        "paper/stats/{dataset_dir.name}_crowd_density.json",
    )
    args = parser.parse_args()
    results = {}
    for split in SPLITS if args.split == "all" else (args.split,):
        stats = crowd_density(args.dataset_dir, split, args.seed)
        print(f"{args.dataset_dir.name}/{split}: {stats}")
        results[split] = stats
    output = args.output or PAPER_STATS_ROOT / f"{args.dataset_dir.name}_crowd_density.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
