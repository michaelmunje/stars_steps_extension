from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pykalman import KalmanFilter


ForecastInputPreprocessing = Literal[
    "raw",
    "causal_kalman",
    "observation_only_smooth",
    "opentraj_offline_smooth_compat",
]
ForecastTargetPreprocessing = Literal["raw", "opentraj_offline_smooth_compat"]


@dataclass(frozen=True)
class ForecastPreprocessingResult:
    input_xy: np.ndarray
    target_xy: np.ndarray
    input_mode: str
    target_mode: str
    uses_future_information_for_input: bool


class OpenTrajKalmanModel:
    """Modern port of OpenTraj's constant-acceleration Kalman smoother
    (crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a,
    toolkit/utils/kalman_smoother.py; references/opentraj.json pins the commit).

    Deliberate deviations from that reference, recorded here rather than left
    to look like drift (BC-016):

    - `n_iter` defaults to 7 here, not upstream's class default of 4
      (kalman_smoother.py:9). This follows upstream's own production call
      site, `trajdataset.py:123` (`KalmanModel(dt, n_dim=2, n_iter=7)`), which
      is what upstream actually runs, not its unused class default.
    - The transition matrix's top-right entry uses `dt**2`, not the textbook
      `dt**2/2` for a constant-acceleration model. This is retained
      deliberately to match `kalman_smoother.py:14` byte-for-byte, not a typo;
      "correcting" it would break parity with `tests/test_opentraj_compat.py`
      and `src/scripts/validate_opentraj_parity.py::validate_kalman`.
    - `_filter` builds a fresh `pykalman.KalmanFilter` on every call instead of
      caching one on `self` as upstream's `__init__` does. This is equivalent:
      upstream itself constructs a fresh `KalmanModel` per trajectory group at
      `trajdataset.py:123`, so building fresh filter state per `_filter` call
      here reproduces that per-group freshness even though a single
      `OpenTrajKalmanModel` instance is reused across all agents in
      `_process_agents`.
    """

    def __init__(self, dt: float, n_dim: int = 2, n_iter: int = 7) -> None:
        self.n_iter = n_iter
        self.n_dim = n_dim
        self.transition = np.array(
            [[1, dt, dt**2], [0, 1, dt], [0, 0, 1]], dtype=float
        )
        self.observation = np.array([[1, 0, 0]], dtype=float)
        self.transition_covariance = 0.5 * np.array(
            [
                [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                [dt**3 / 6, dt**2 / 2, dt],
            ],
            dtype=float,
        )
        self.observation_covariance = np.array([[1.0]])

    def _filter(self, measurement: np.ndarray, *, smooth: bool) -> tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=float)
        if measurement.ndim != 2 or measurement.shape[1] != self.n_dim:
            raise ValueError(f"measurement must have shape (T, {self.n_dim})")
        if len(measurement) == 1:
            # Upstream (kalman_smoother.py:52) guards only `smooth`; its
            # `filter` has no length-1 branch and would run EM on a single
            # sample. Applying the guard to both here (BC-016) is deliberate
            # hardening beyond upstream, not an accidental extension: the
            # user's decision was to keep behaviour as-is and document it.
            return measurement.copy(), np.zeros_like(measurement)
        states = []
        for dim in range(self.n_dim):
            kalman = KalmanFilter(
                transition_matrices=self.transition,
                observation_matrices=self.observation,
                transition_covariance=self.transition_covariance,
                observation_covariance=self.observation_covariance,
            ).em(measurement[:, dim], n_iter=self.n_iter)
            means, _ = kalman.smooth(measurement[:, dim]) if smooth else kalman.filter(measurement[:, dim])
            states.append(means)
        stacked = np.stack(states)
        return stacked[:, :, 0].T, stacked[:, :, 1].T

    def smooth(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._filter(measurement, smooth=True)

    def filter(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._filter(measurement, smooth=False)


def _segments(valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    return [part for part in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(part)]


def _process_agents(
    xy: np.ndarray,
    valid: np.ndarray,
    dt: float,
    *,
    smooth: bool,
    stop: int | None = None,
) -> np.ndarray:
    output = np.array(xy, dtype=np.float32, copy=True)
    limit = len(output) if stop is None else min(stop, len(output))
    model = OpenTrajKalmanModel(dt=dt, n_dim=2, n_iter=7)
    for agent in range(output.shape[1]):
        mask = np.asarray(valid[:limit, agent], dtype=bool)
        for segment in _segments(mask):
            positions = output[segment, agent].astype(float)
            processed, _ = model.smooth(positions) if smooth else model.filter(positions)
            output[segment, agent] = processed
    return output


def preprocess_forecast_scene(
    xy: np.ndarray,
    valid: np.ndarray,
    obs_len: int = 8,
    dt: float = 0.4,
    input_mode: ForecastInputPreprocessing = "raw",
    target_mode: ForecastTargetPreprocessing = "raw",
) -> ForecastPreprocessingResult:
    """Build explicitly separated inputs and targets without silent future leakage."""
    xy = np.asarray(xy, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if xy.ndim != 3 or valid.shape != xy.shape[:2]:
        raise ValueError("xy/valid must have shapes (T,N,2)/(T,N)")
    if not 1 < obs_len < len(xy):
        raise ValueError("obs_len must split the observed and future samples")

    input_xy = np.array(xy, copy=True)
    target_xy = np.array(xy[obs_len:], copy=True)
    leaks = False
    # Result of the full-sequence offline smoother, kept only when the input mode already ran it
    # over the whole window. observation_only_smooth must not populate this: it passes
    # stop=obs_len, which is a different computation.
    offline_smoothed: np.ndarray | None = None
    if input_mode == "causal_kalman":
        processed = _process_agents(xy, valid, dt, smooth=False, stop=obs_len)
        input_xy[:obs_len] = processed[:obs_len]
    elif input_mode == "observation_only_smooth":
        processed = _process_agents(xy, valid, dt, smooth=True, stop=obs_len)
        input_xy[:obs_len] = processed[:obs_len]
    elif input_mode == "opentraj_offline_smooth_compat":
        offline_smoothed = _process_agents(xy, valid, dt, smooth=True)
        input_xy = offline_smoothed
        leaks = True
    elif input_mode != "raw":
        raise ValueError(f"Unknown forecast input preprocessing: {input_mode}")

    if target_mode == "opentraj_offline_smooth_compat":
        # _process_agents is a pure function of (xy, valid, dt, smooth, stop) and pykalman's EM is
        # deterministic, so when the input mode produced the same full-sequence smoothing the
        # result is bit-identical and the (expensive) second fit is skipped.
        if offline_smoothed is None:
            offline_smoothed = _process_agents(xy, valid, dt, smooth=True)
        # Copy so target_xy never becomes a view into input_xy when the array is shared.
        target_xy = np.array(offline_smoothed[obs_len:], copy=True)
    elif target_mode != "raw":
        raise ValueError(f"Unknown forecast target preprocessing: {target_mode}")
    return ForecastPreprocessingResult(input_xy, target_xy, input_mode, target_mode, leaks)
