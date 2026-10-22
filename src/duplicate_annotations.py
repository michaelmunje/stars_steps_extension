"""Detection of duplicate pedestrian annotations (one person, two track identities).

Some upstream releases annotate the same pedestrian twice under different track
identities, persistently co-located for the whole overlap. Grand Central is the
clear case: its pixel-space annotations put pairs such as ``14740``/``14747``
2-3 px apart across a track spanning 54->844 px, so the duplication exists in
the CVPR2015 release itself rather than in this repository's homography or
gap-splitting. MADRAS shows the same signature more weakly.

This matters for `opentraj_indicators.frame_local_density`, whose adaptive
bandwidth for a pedestrian is its nearest neighbour distance *strictly above*
0.01 m (0.01 m is only a fallback when no such distance exists in the frame)
and whose contributions scale as ``1/b**2``. An exactly coincident pair is
therefore excluded from each other's bandwidth search, so in any realistic
frame it does *not* spike the density: a coincident pair with one more
pedestrian 1 m away gives only ~0.41 ped/m**2 to each member, because the 1 m
neighbour sets the bandwidth. The fallback figure of ~3,183 ped/m**2 arises
only when the pair is *alone* in the frame, which never occurs here -- Grand
Central averages 76.2 pedestrians per frame.

The real driver is a *near*-duplicate a few centimetres away: that distance
becomes its twin's nearest admissible neighbour, driving the bandwidth down to
~0.02-0.06 m and the density very high (e.g. 0.02 m away with a third
pedestrian 1 m off gives ~639 ped/m**2 to each of the pair), so a duplicate
rate under 2% of tracks dominated the published Grand Central figure.

Detection is deliberately conservative. It reports pairs and the identities to
exclude; it never edits canonical data, and it is not used by the
parity-pinned kernel itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# Two pedestrian centroids cannot stay 5 cm apart: at that separation the
# annotations are the same person. The threshold is deliberately far below any
# plausible real spacing, and is calibrated against the manually annotated
# releases -- at 0.05 m it flags nothing whatsoever in ETH or UCY, whereas
# 0.3 m already removes 45 UCY tracks that are genuine close companions.
# Preferring false negatives keeps a crowding statistic honest: an undetected
# duplicate inflates it, but a wrongly discarded companion deflates it and is
# indistinguishable from the effect being measured.
DUPLICATE_SEPARATION_M = 0.05
# One coincident frame is a crossing or a tracker glitch. Three canonical
# frames (1.2 s at 2.5 Hz) is what separates a duplicated identity from a
# close pass.
DUPLICATE_MIN_FRAMES = 3


@dataclass(frozen=True)
class DuplicatePair:
    """Two track identities that behave like one annotated pedestrian."""

    track_a: int
    track_b: int
    co_occurring_frames: int
    frames_within_threshold: int
    min_separation_m: float
    median_separation_m: float


def _positions_by_track(
    track_id: np.ndarray, step: np.ndarray, xy: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    order = np.lexsort((step, track_id))
    ids, steps, positions = track_id[order], step[order], xy[order]
    boundaries = np.flatnonzero(np.diff(ids)) + 1
    return {
        int(group[0]): (group_steps, group_xy)
        for group, group_steps, group_xy in zip(
            np.split(ids, boundaries), np.split(steps, boundaries), np.split(positions, boundaries)
        )
    }


def duplicate_pairs(
    track_id: np.ndarray,
    step: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    separation_m: float = DUPLICATE_SEPARATION_M,
    min_frames: int = DUPLICATE_MIN_FRAMES,
) -> tuple[DuplicatePair, ...]:
    """Return track pairs kept within `separation_m` for at least `min_frames` frames.

    Rows are one (track, step) observation each, as stored in the canonical
    converted recordings and the analysis views. Pairs are returned sorted by
    identity so the result is reproducible.
    """
    track_id = np.asarray(track_id, dtype=np.int64)
    step = np.asarray(step, dtype=np.int64)
    xy = np.c_[np.asarray(x, dtype=float), np.asarray(y, dtype=float)]
    if len(track_id) < 2:
        return ()

    order = np.argsort(step, kind="stable")
    ids, positions = track_id[order], xy[order]
    boundaries = np.flatnonzero(np.diff(step[order])) + 1
    close_frames: Counter[tuple[int, int]] = Counter()
    for frame_ids, frame_xy in zip(np.split(ids, boundaries), np.split(positions, boundaries)):
        if len(frame_ids) < 2 or not np.isfinite(frame_xy).all():
            continue
        for first, second in cKDTree(frame_xy).query_pairs(separation_m):
            a, b = int(frame_ids[first]), int(frame_ids[second])
            close_frames[(a, b) if a < b else (b, a)] += 1

    candidates = {pair: count for pair, count in close_frames.items() if count >= min_frames}
    if not candidates:
        return ()
    by_track = _positions_by_track(track_id, step, xy)
    pairs = []
    for (a, b), frames_within in sorted(candidates.items()):
        steps_a, xy_a = by_track[a]
        steps_b, xy_b = by_track[b]
        _, index_a, index_b = np.intersect1d(steps_a, steps_b, return_indices=True)
        separations = np.linalg.norm(xy_a[index_a] - xy_b[index_b], axis=1)
        # A non-finite position poisons its own separation, so skip those rather than
        # letting one NaN decide the reported minimum. `np.errstate` is the wrong tool
        # here -- numpy signals an all-NaN reduction via warnings.warn(), not the
        # floating-point error state -- so branch explicitly instead of appearing to
        # suppress a warning that errstate never reached.
        finite = np.isfinite(separations)
        if finite.any():
            min_separation_m = float(separations[finite].min())
            median_separation_m = float(np.median(separations[finite]))
        else:
            min_separation_m = float("nan")
            median_separation_m = float("nan")
        pairs.append(
            DuplicatePair(
                track_a=a,
                track_b=b,
                co_occurring_frames=int(len(separations)),
                frames_within_threshold=int(frames_within),
                min_separation_m=min_separation_m,
                median_separation_m=median_separation_m,
            )
        )
    return tuple(pairs)


def duplicate_tracks(pairs: tuple[DuplicatePair, ...]) -> frozenset[int]:
    """Track identities to exclude, keeping the lowest id of each duplicate group.

    Duplicate pairs are treated as edges: a pedestrian annotated three times
    forms one connected component and contributes one surviving identity, not
    one per pair. Keeping the lowest id makes the choice deterministic and
    independent of pair ordering.
    """
    parent: dict[int, int] = {}

    def find(item: int) -> int:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for pair in pairs:
        root_a, root_b = find(pair.track_a), find(pair.track_b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return frozenset(track for track in parent if find(track) != track)


def duplicate_summary(
    pairs: tuple[DuplicatePair, ...],
    excluded: frozenset[int],
    *,
    separation_m: float = DUPLICATE_SEPARATION_M,
    min_frames: int = DUPLICATE_MIN_FRAMES,
) -> dict:
    """Provenance record for a duplicate-annotation filter pass."""
    duplicate_min_separation_m = None
    if pairs:
        with np.errstate(invalid="ignore"):
            duplicate_min_separation_m = float(
                np.nanmin([pair.min_separation_m for pair in pairs])
            )
    return {
        "duplicate_separation_m": float(separation_m),
        "duplicate_min_frames": int(min_frames),
        "duplicate_pairs": len(pairs),
        "duplicate_tracks_excluded": len(excluded),
        "duplicate_min_separation_m": duplicate_min_separation_m,
    }
