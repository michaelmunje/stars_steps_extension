from __future__ import annotations

from pathlib import Path

SPLITS = ("train", "val", "test")


def _recording_key(recording: str) -> str:
    """Canonicalize a recording identity to its stem, independent of whether
    the caller passed the bare recording name or its ``.traj.parquet`` file
    name."""
    return recording.removesuffix(".traj.parquet")


def assign_split(config: dict, *, recording: str, start: int, first: int,
                 last: int, window_span: int) -> str | None:
    """Assign one complete window according to an explicit split strategy."""
    # The "recording" default selects a branch nothing can request: every config reaching
    # here comes from create_splits (preprocess.py:228, the only production caller), whose
    # strategy is validated against the four names in config.py:82. Verified 2026-09-15
    # (AG-005): the sole repo-wide producer of strategy="recording" is a test constructing
    # its own dict, and 0 of 10 persisted manifests use it. Retained rather than deleted
    # because data.py's _validate_splits still carries a matching "recording" branch, so
    # the format was once real; removing it is a coordinated change, not a local tidy-up.
    strategy = config.get("strategy", "recording")
    if strategy == "temporal":
        ratios = config["ratios"]; guard = int(config["guard_steps"])
        end = last + window_span
        train_boundary = first + (end - first) * ratios["train"]
        val_boundary = train_boundary + (end - first) * ratios["val"]
        if start + window_span <= train_boundary: return "train"
        if start >= train_boundary + guard and start + window_span <= val_boundary: return "val"
        if start >= val_boundary + guard: return "test"
        return None
    if strategy == "held_out_recording":
        # One named recording is the whole test split; every other recording is
        # split train/val along its own timeline, with the same guard interval
        # the temporal strategy uses so no window straddles the boundary.
        key = _recording_key(recording)
        role = config.get("split_by_recording", {}).get(key)
        if role is None:
            raise ValueError(
                f"held_out_recording: unrecognized recording {recording!r} is absent from "
                f"split_by_recording (held-out recording: {config.get('held_out_recording')!r}); "
                "refusing to silently default it to train"
            )
        if role == "test":
            return "test"
        ratios = config["ratios"]; guard = int(config["guard_steps"])
        denominator = ratios["train"] + ratios["val"]
        end = last + window_span
        boundary = first + (end - first) * (ratios["train"] / denominator)
        if start + window_span <= boundary: return "train"
        if start >= boundary + guard: return "val"
        return None
    if strategy == "recording":
        name = _recording_key(recording) + ".traj.parquet"
        for split in SPLITS:
            # `Path(name).name` is unconditionally a no-op: `name` is built above from
            # `_recording_key(...)` plus a suffix and contains no path separator.
            if any(Path(item).name == Path(name).name for item in config.get(split, [])):
                return split
        return None
    if strategy == "grouped_scene_sequence":
        mapping = config.get("split_by_recording", {})
        return mapping.get(_recording_key(recording))
    if strategy == "leave_one_scene_out":
        key = _recording_key(recording)
        mapping = config.get("split_by_recording", {})
        if key in mapping:
            return mapping[key]
        scene_by_recording = config.get("scene_by_recording", {})
        if key not in scene_by_recording:
            raise ValueError(
                f"leave_one_scene_out: unrecognized recording {recording!r} "
                f"is absent from both split_by_recording and scene_by_recording "
                f"(configured held-out scene: {config.get('test_scene')!r}); "
                "refusing to silently default it to train"
            )
        scene = scene_by_recording[key]
        if scene == config.get("test_scene"): return "test"
        if scene in set(config.get("val_scenes", [])): return "val"
        return "train"
    raise ValueError(f"Unknown split strategy: {strategy}")
