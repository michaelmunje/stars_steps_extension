"""Central-difference heading for contiguous canonical tracks.

The canonical schema stores a per-row ``heading`` in radians (see
:mod:`canonical`).  Adapters whose tracks are contiguous on the canonical
lattice by construction -- UCY and Grand Central -- all derive it from the same
convention, which lives here so that a correction cannot silently apply to one
dataset only:

* interior rows use the central difference ``xy[i + 1] - xy[i - 1]``;
* the first and last row fall back to the one-sided difference against their
  single neighbour;
* a displacement at or below :data:`HEADING_SPEED_EPSILON` world units leaves
  the heading undefined (``NaN``), because ``arctan2`` of a numerically stationary
  step carries no direction.

Adapters that do **not** use this helper fall into two other groups:

* tracks that may contain gaps (madras, PTDPES) deliberately apply an extra
  per-dataset contiguity guard before differencing instead of using this
  helper;
* tracks whose canonical ``heading`` is not differenced at all, but taken
  from a source-provided convention: ATC (``heading`` is a source column),
  STEPS (``yaw_ground_rad``), and ETH (derived from source velocity
  columns).
"""

from __future__ import annotations

import numpy as np


# Displacement floor, in canonical world units, below which heading is undefined.
# This is a fixed scientific convention shared by the adapters above; do not
# retune it for one dataset.
HEADING_SPEED_EPSILON = 1e-8


def central_difference_heading(xy: np.ndarray) -> np.ndarray:
    """Return the heading in radians for each row of a contiguous track.

    ``xy`` is an ``(n, 2)`` array of positions ordered by step, with no gaps on
    the canonical lattice.  The result is a length-``n`` float array holding
    ``arctan2(dy, dx)`` in ``[-pi, pi]``, or ``NaN`` where the differencing
    stencil yields a displacement of at most :data:`HEADING_SPEED_EPSILON`.

    A single-position track has no neighbour to difference against, so its
    heading is ``NaN`` by the same convention.  An empty track is rejected.
    """
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected an (n, 2) array of positions, got shape {points.shape}")
    if len(points) == 0:
        raise ValueError("cannot derive heading for a track with no positions")
    if len(points) == 1:
        return np.array([np.nan])
    delta = np.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    if len(points) > 2:
        delta[1:-1] = points[2:] - points[:-2]
    speed = np.linalg.norm(delta, axis=1)
    return np.where(speed > HEADING_SPEED_EPSILON, np.arctan2(delta[:, 1], delta[:, 0]), np.nan)
