#!/usr/bin/env python3
"""Build an anonymised STEPS release from the cleaned recordings.

    build_anonymous_release.py --source <dir> --dest <dir> [--limit N] [--dry-run]

Layout produced:

    Atrium_Data/MM/DD/STEPS_MM_DD_HHMMSS_trajectory_data.jsonl
    Atrium_Data/MM/DD/STEPS_MM_DD_HHMMSS_cleanup_report.json
    Atrium_Data/manifest.json

Each frame gains `model_detections_by_camera`: the per-detection body pose grouped by the
camera that produced it, matching the v1 schema. With --pose-layout move (default) the pose
is relocated there and `projected_detections` keeps `detection_id` to reference it; with
duplicate it appears in both, as v1 did, which costs ~76% more disk for no extra
information. Every line is scanned for forbidden patterns and the recording FAILS if one
appears, so anonymity is verified rather than assumed. Sidecars are rebuilt field by field
from an allow-list, which drops absolute paths and personal directories by construction.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import sys

try:
    import orjson  # same library the upstream cleanup used; ~5x faster than json here
except ModuleNotFoundError:  # pragma: no cover - fall back rather than fail
    orjson = None
import time
from datetime import datetime
from pathlib import Path

# Anything matching these must never reach the release.
FORBIDDEN = re.compile(rb"gdc|speedway|austin|utexas|zhengpengen|/home/|/robodata|/Users/", re.I)
CHUNK = 8 << 20
RECORDING_NAME = re.compile(r"^[a-z]+_(\d{4})(\d{2})(\d{2})_(\d{6})$")
# Calibration / validation bags that must never enter the release.
DEFAULT_EXCLUDED = ("atrium_20260824_120001",)

# Sidecar fields kept verbatim. Every path-valued field is deliberately absent.
REPORT_KEEP = (
    "format", "config", "parser", "input_bytes", "output_bytes", "recording_seconds",
    "frames", "empty_frames", "input_fused_people", "output_fused_people",
    "input_projected_detections", "output_projected_detections", "input_track_records",
    "output_track_records", "removed_track_records", "flipped_output_records",
    "flipped_fused_records", "invalid_position_records", "flipped_records_by_track",
    "orientation_events", "peak_buffered_frames", "peak_active_tracks",
)
RECEIPT_KEEP = ("receipt_version", "config", "frames")


def released_name(recording: str) -> tuple[str, str, str]:
    """atrium_20260818_094501 -> ('08', '18', 'STEPS_08_18_094501')"""
    match = RECORDING_NAME.match(recording)
    if not match:
        raise ValueError(f"unexpected recording name: {recording}")
    _, month, day, hhmmss = match.groups()
    return month, day, f"STEPS_{month}_{day}_{hhmmss}"


def _loads(raw: bytes):
    return orjson.loads(raw) if orjson else json.loads(raw)


def _dumps(obj) -> bytes:
    return orjson.dumps(obj) if orjson else json.dumps(obj, separators=(",", ":")).encode()


# Fields that describe the detection in its own camera's frame. Everything else in a
# projected detection describes the world-frame result and stays in the global list.
CAMERA_FRAME_FIELDS = (
    "bbox_xyxy", "confidence", "foot_pixel_uv", "foot_quality", "foot_source",
    "image_width_px", "image_height_px", "suppressed_duplicate", "duplicate_of_detection_id",
    "camera_track_id", "camera_track_confirmed", "camera_track_observed",
    "camera_track_consecutive_hits", "camera_track_consistent_seconds",
    "camera_position_smoothed_xy", "camera_position_covariance",
    "camera_measurement_covariance", "camera_velocity_ground_xy_mps",
)


def split_by_frame_of_reference(frame: dict, duplicate: bool) -> None:
    """Per-camera detections go to `model_detections_by_camera`; `projected_detections`
    keeps one world-frame record per detection, linked by `detection_id`."""
    by_camera: dict[str, list] = {}
    global_records = []
    for det in frame.get("projected_detections", []):
        camera = str(det.get("camera_index"))
        entry = {"detection_id": det.get("detection_id"), "camera_index": det.get("camera_index")}
        for key in CAMERA_FRAME_FIELDS:
            if key in det:
                entry[key] = det[key] if duplicate else det.pop(key)
        pose = det.get("sam3d") if duplicate else det.pop("sam3d", None)
        if pose is not None:
            entry["sam3d"] = pose
        by_camera.setdefault(camera, []).append(entry)
        det.pop("camera_index", None) if not duplicate else None
        global_records.append(det)
    for cam in frame.get("present_cameras", []):
        by_camera.setdefault(str(cam), [])
    frame["model_detections_by_camera"] = dict(sorted(by_camera.items()))
    frame["projected_detections"] = global_records


def convert_and_scan(src: Path, dst: Path, duplicate: bool) -> tuple[int, str, int]:
    """Rewrite the payload frame by frame, refusing to finish on any forbidden text."""
    digest = hashlib.sha256()
    total = frames = 0
    tmp = dst.with_suffix(dst.suffix + ".partial")
    with src.open("rb", buffering=1 << 22) as fh, tmp.open("wb", buffering=1 << 22) as out:
        for raw in fh:
            if FORBIDDEN.search(raw):
                tmp.unlink(missing_ok=True)
                raise ValueError(f"forbidden text found inside {src}")
            frame = _loads(raw)
            split_by_frame_of_reference(frame, duplicate)
            line = _dumps(frame) + b"\n"
            out.write(line)
            digest.update(line)
            total += len(line)
            frames += 1
    tmp.replace(dst)
    return total, digest.hexdigest(), frames


def rebuild_sidecar(data: dict, keep: tuple[str, ...], recording_label: str) -> dict:
    out = {k: data[k] for k in keep if k in data}
    out["recording"] = recording_label
    leaked = FORBIDDEN.search(json.dumps(out).encode())
    if leaked:
        raise ValueError(f"sidecar for {recording_label} still contains {leaked.group()!r}")
    return out


def process_recording(job: tuple) -> dict:
    """Convert one recording. Runs in a worker process; returns a result record."""
    rec_str, dest_str, duplicate, overwrite = job
    rec, dest = Path(rec_str), Path(dest_str)
    try:
        month, day, label = released_name(rec.name)
    except ValueError as exc:
        return {"error": str(exc), "recording": rec.name}
    frames = next(rec.rglob("frames.cleaned.jsonl"), None)
    if frames is None:
        return {"error": "no frames.cleaned.jsonl", "recording": rec.name}
    out_dir = dest / "Atrium_Data" / month / day
    report_dir = dest / "Cleanup_Reports" / month / day
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{label}_trajectory_data.jsonl"
    try:
        if target.exists() and not overwrite:
            entry = {"recording": label, "bytes": target.stat().st_size, "sha256": "skipped-existing"}
        else:
            size, sha, count = convert_and_scan(frames, target, duplicate)
            entry = {"recording": label, "bytes": size, "sha256": sha, "frames_written": count}
        report = next(rec.rglob("cleanup.report.json"), None)
        if report:
            clean = rebuild_sidecar(json.loads(report.read_text()), REPORT_KEEP, label)
            receipt = next(rec.rglob("cleanup.receipt.json"), None)
            if receipt:
                clean["receipt"] = rebuild_sidecar(
                    json.loads(receipt.read_text()), RECEIPT_KEEP, label)
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"{label}_cleanup_report.json").write_text(json.dumps(clean, indent=2) + "\n")
            entry["frames"] = clean.get("frames")
            entry["recording_seconds"] = clean.get("recording_seconds")
        return entry
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"error": str(exc), "recording": rec.name}


def build(source: Path, dest: Path, limit: int | None, dry_run: bool,
          duplicate: bool, overwrite: bool, workers: int, exclude: tuple[str, ...]) -> int:
    recordings = sorted(p for p in source.iterdir()
                        if p.is_dir() and p.name not in set(exclude))
    if exclude:
        print(f"excluding {len(exclude)} recording(s): {', '.join(sorted(exclude))}")
    if limit:
        recordings = recordings[:limit]
    if dry_run:
        for rec in recordings:
            month, day, label = released_name(rec.name)
            print(f"  {rec.name} -> Atrium_Data/{month}/{day}/{label}_trajectory_data.jsonl")
        print(f"\n{len(recordings)} recording(s) would be written by {workers} worker(s)")
        return 0

    # Largest first, so one long recording cannot become the tail that everything waits on.
    def source_size(rec: Path) -> int:
        f = next(rec.rglob("frames.cleaned.jsonl"), None)
        return f.stat().st_size if f else 0
    recordings.sort(key=source_size, reverse=True)

    jobs = [(str(r), str(dest), duplicate, overwrite) for r in recordings]
    entries, failures = [], []
    started = time.monotonic()
    done_bytes = 0
    total_bytes = sum(source_size(r) for r in recordings) or 1
    # as_completed, not map: map yields in submission order, so with largest-first
    # scheduling nothing would print until the biggest recording finished.
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(process_recording, job) for job in jobs]
        for index, future in enumerate(futures.as_completed(pending), 1):
            result = future.result()
            if "error" in result:
                failures.append(f"{result['recording']}: {result['error']}")
                print(f"[{index}/{len(jobs)}] FAILED {result['recording']}: {result['error']}",
                      file=sys.stderr, flush=True)
                continue
            entries.append(result)
            done_bytes += result.get("bytes", 0)
            elapsed = time.monotonic() - started
            rate = done_bytes / elapsed / 1e6 if elapsed else 0
            remaining = (total_bytes - done_bytes) / (done_bytes / elapsed) if done_bytes else 0
            print(f"[{index}/{len(jobs)}] {result['recording']} "
                  f"({result.get('bytes', 0)/1e6:.0f} MB) | {rate:.0f} MB/s | "
                  f"eta {remaining/60:.0f} min", flush=True)

    root = dest / "Atrium_Data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "dataset": "STEPS",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_hz": 20.0,
        "layout": "Atrium_Data/MM/DD/STEPS_MM_DD_HHMMSS_trajectory_data.jsonl",
        "cleanup_reports": "Cleanup_Reports/MM/DD/STEPS_MM_DD_HHMMSS_cleanup_report.json - "
                           "provenance for the upstream cleanup pass; not required to use the data",
        "excluded_recordings": list(exclude),
        "frame_arrays": ["model_detections_by_camera", "projected_detections",
                         "fused_people", "tracked_people"],
        "model_detections_by_camera": "mapping from camera id to that camera's detections, "
                                      "with image-frame geometry and sam3d body pose",
        "projected_detections": "flat global list, one world-frame record per detection, "
                                "linked to its camera record by detection_id",
        "includes_body_pose": True,
        "includes_source_imagery": False,
        "timestamp_ns": "nanoseconds from an arbitrary monotonic origin, consistent across "
                        "recordings; wall-clock date and time are in the filename",
        "recordings": len(entries),
        "entries": sorted(entries, key=lambda e: e["recording"]),
    }, indent=2) + "\n")

    elapsed = time.monotonic() - started
    if failures:
        print(f"\n{len(failures)} FAILED:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
    print(f"\n{len(entries)} recording(s) written, {len(failures)} failed "
          f"in {elapsed/60:.1f} min ({done_bytes/elapsed/1e6:.0f} MB/s)")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pose-layout", choices=("move", "duplicate"), default="move",
                        help="move (default) relocates pose into model_detections_by_camera; "
                             "duplicate also keeps it in projected_detections[].sam3d, as v1 did")
    parser.add_argument("--overwrite", action="store_true", help="rebuild files that already exist")
    parser.add_argument("--exclude", action="append", default=list(DEFAULT_EXCLUDED),
                        help="source recording name to leave out; repeatable")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help="parallel worker processes (default: half the logical cores)")
    args = parser.parse_args()
    if not args.source.is_dir():
        print(f"error: no such source directory: {args.source}", file=sys.stderr)
        return 2
    return build(args.source, args.dest, args.limit, args.dry_run,
                 args.pose_layout == "duplicate", args.overwrite, args.workers,
                 tuple(args.exclude))


if __name__ == "__main__":
    raise SystemExit(main())
