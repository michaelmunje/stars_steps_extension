"""Reproduce the PTDPES duration and gap audit from the distributed CSV files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _summarize(files: list[Path], *, raw: bool) -> dict[str, float | int]:
    durations: list[np.ndarray] = []
    point_counts: list[np.ndarray] = []
    tracks = gaps = extra_segments = 0
    for path in files:
        time_column = "frame_index" if raw else "time"
        frame = pd.read_csv(path, usecols=["tracker_id", time_column]).dropna()
        frame = frame.drop_duplicates(["tracker_id", time_column])
        for _, track in frame.groupby("tracker_id", sort=False):
            values = np.sort(track[time_column].to_numpy(dtype=float))
            if not len(values):
                continue
            seconds = values / 15.0 if raw else values
            durations.append(np.asarray([seconds[-1] - seconds[0]], dtype=float))
            point_counts.append(np.asarray([len(values)], dtype=float))
            delta = np.diff(seconds)
            boundaries = int(np.count_nonzero(delta > (1.0 / 15.0 + 1e-5 if raw else 0.5 + 1e-5)))
            gaps += int(boundaries > 0)
            extra_segments += boundaries
            tracks += 1
    values = np.concatenate(durations) if durations else np.empty(0)
    points = np.concatenate(point_counts) if point_counts else np.empty(0)
    duration_bins={
        "2_to_10_s":int(np.count_nonzero((values>=2)&(values<10))),
        "10_to_50_s":int(np.count_nonzero((values>=10)&(values<50))),
        "50_to_100_s":int(np.count_nonzero((values>=50)&(values<100))),
        "over_100_s":int(np.count_nonzero(values>=100)),
    }
    return {
        "files": len(files),
        "tracks": tracks,
        "median_whole_track_span_s": float(np.median(values)),
        "mean_whole_track_span_s": float(np.mean(values)),
        "median_data_points": float(np.median(points)),
        "mean_data_points": float(np.mean(points)),
        "tracks_with_gaps": gaps,
        "extra_contiguous_segments_from_gaps": extra_segments,
        "duration_bins_s":duration_bins,
    }


def _seasonal_key(path: Path) -> str:
    return path.stem.removesuffix("_speed")


def _comparative_key(path: Path, *, raw: bool) -> tuple[str, str] | None:
    if raw:
        match = re.match(r"(\d+)_(\d{4}-\d{2}-\d{2})-(\d{2})h", path.name)
        if not match:
            return None
        place, date_text, hour_text = match.groups()
        date = datetime.strptime(date_text, "%Y-%m-%d")
        hour = int(hour_text)
        period = "saturday" if date.weekday() == 5 else "morning" if hour < 11 else "noon" if hour < 16 else "evening"
        return place, period
    match = re.match(r"(\d+)_(morning|noon|evening|saturday)_", path.name)
    return (match.group(1), match.group(2)) if match else None


def _mapping_rows(raw_files: list[Path], modified_files: list[Path], raw_root: Path, modified_root: Path) -> list[dict[str, str]]:
    normal_modified = [path for path in modified_files if "Test_set" not in path.parts]
    modified_lookup: dict[tuple[str, str, str], Path] = {}
    for path in normal_modified:
        if "01.Trajectories_comparative" in path.parts:
            key = _comparative_key(path, raw=False)
            if key:
                modified_lookup[("comparative", *key)] = path
        else:
            modified_lookup[("seasonal", _seasonal_key(path), "")] = path
    used: set[Path] = set()
    rows: list[dict[str, str]] = []
    for raw_path in raw_files:
        if "01.Trajectories_comparative" in raw_path.parts:
            key = _comparative_key(raw_path, raw=True)
            lookup = ("comparative", *key) if key else None
            subset = "comparative"
        else:
            lookup = ("seasonal", _seasonal_key(raw_path), "")
            subset = "seasonal"
        modified_path = modified_lookup.get(lookup) if lookup else None
        if modified_path:
            used.add(modified_path)
        rows.append({"raw_path": str(raw_path.relative_to(raw_root)), "modified_path": str(modified_path.relative_to(modified_root)) if modified_path else "", "subset": subset, "status": "matched" if modified_path else "raw_only"})
    for path in normal_modified:
        if path not in used:
            rows.append({"raw_path": "", "modified_path": str(path.relative_to(modified_root)), "subset": "comparative" if "01.Trajectories_comparative" in path.parts else "seasonal", "status": "modified_only"})
    for path in modified_files:
        if "Test_set" not in path.parts:
            continue
        counterpart = next((candidate for candidate in normal_modified if candidate.name == path.name), None)
        duplicate = bool(counterpart and hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(counterpart.read_bytes()).digest())
        rows.append({"raw_path": "", "modified_path": str(path.relative_to(modified_root)), "subset": "test_set", "status": "byte_identical_duplicate" if duplicate else "unresolved_test_set"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=Path(__file__).resolve().parents[3]/"datasets/PTDPES")
    args = parser.parse_args()
    root = args.dataset_dir / "raw/01.Trajectories"
    raw_files = sorted((root / "01.Trajectories_raw").rglob("*.csv"))
    modified_root = root / "02.Trajectories_modified"
    raw_root = root / "01.Trajectories_raw"
    modified_files = sorted(modified_root.rglob("*.csv"))
    production_modified = [path for path in modified_files if "Test_set" not in path.parts]
    mapping = _mapping_rows(raw_files, modified_files, raw_root, modified_root)
    result = {
        "raw": _summarize(raw_files, raw=True),
        "modified_excluding_test_set": _summarize(production_modified, raw=False),
        "recording_mapping": {
            "matched": sum(row["status"] == "matched" for row in mapping),
            "raw_only": sum(row["status"] == "raw_only" for row in mapping),
            "modified_only": sum(row["status"] == "modified_only" for row in mapping),
            "test_set_byte_identical_duplicates": sum(row["status"] == "byte_identical_duplicate" for row in mapping),
        },
        "published_reference": {
            "median_s": 32,
            "mean_s": 54,
            "article": "https://www.nature.com/articles/s41597-026-06686-6",
            "duration_bin_claims":{"50_to_100_s":"more than 39,000","over_100_s":"more than 17,000"},
        },
        "point_count_hypothesis": (
            "The modified-release point-count median/mean (29/58.5) are much closer to the paper's "
            "32/54 than the measured durations (16.0/31.1 s), suggesting a mislabeled unit."
        ),
        "interpretation": (
            "The modified release reproduces the paper's decisive duration-bin claims: 39,085 tracks "
            "in 50-100 s and 17,378 over 100 s satisfy the stated >39,000 and >17,000 counts. The "
            "converter is therefore not losing the long-duration population. The paper's 32/54-second "
            "summary is internally inconsistent with the distributed release and is most plausibly a "
            "point-count/unit-label error; report the measured release values with this caveat."
        ),
    }
    output = Path(__file__).resolve().parents[3] / "paper/stats/ptdpes_duration_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    mapping_output = output.with_name("ptdpes_recording_map.csv")
    with mapping_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("raw_path", "modified_path", "subset", "status"))
        writer.writeheader()
        writer.writerows(mapping)
    print(output)
    print(mapping_output)


if __name__ == "__main__":
    main()
