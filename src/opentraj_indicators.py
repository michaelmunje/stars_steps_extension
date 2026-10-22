"""Ports of upstream OpenTraj toolkit indicators, pinned at
crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a (references/opentraj.json).

Per-function upstream originals (opentraj/toolkit/indicators/ at the pinned commit):
  motion_properties      <- motion_properties.py (speed_avg_of_trajs, speed_var_of_trajs,
                             acceleration_of_tarjs)
  path_efficiency_index  <- path_efficiency.py::path_efficiency_index
  dca_ttc_matrices       <- collision_energy.py::DCA_MTX
  frame_local_density    <- crowd_density.py::local_density
  angular_deviation,
  aligned_angular_sample <- traj_deviation.py

Parity against the vendored checkout is checked in
src/scripts/validate_opentraj_parity.py for motion_properties, path_efficiency_index,
dca_ttc_matrices, frame_local_density, collision_energy, and global_density.
angular_deviation and aligned_angular_sample are NOT parity-checked: upstream's
traj_deviation.py (:13) is plot-coupled and references an undefined `ds_name` (:52),
so it is not directly callable, and parity for it would need a replicated expectation
rather than a direct call against upstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


PEDESTRIAN_RADIUS_M = 0.33  # metres; opentraj/toolkit/indicators/collision_energy.py:53's
# Rp = 0.33 at the pinned commit above. Upstream states this as a bare assumption
# ("# assume pedestrians radius is 0.33") with no cited source of its own; we
# document its provenance as "upstream OpenTraj" rather than inventing one. See
# K-001 for other, differing radius conventions used elsewhere in this repo for
# nominally similar quantities.


def motion_properties(trajlets: np.ndarray) -> dict[str, np.ndarray]:
    """Exact OpenTraj speed and scalar-speed acceleration definitions
    (opentraj/toolkit/indicators/motion_properties.py at the pinned commit above).

    trajlets: array of shape (N, T, 5) = (x, y, vx, vy, t), t in seconds
    (opentraj_report.py builds this as step * CANONICAL_DT). acceleration_* is
    d(speed)/dt and is in m/s^2 when x, y are in metres.
    """
    values = np.asarray(trajlets, dtype=float)
    speed = np.linalg.norm(values[:, :, 2:4], axis=2)
    dt = np.diff(values[:, :, 4], axis=1)
    acceleration = np.diff(speed, axis=1) / dt
    return {
        "speed_mean": np.mean(speed, axis=1),
        "speed_range": np.max(speed, axis=1) - np.min(speed, axis=1),
        "acceleration_mean_abs": np.mean(np.abs(acceleration), axis=1),
        "acceleration_max_abs": np.max(np.abs(acceleration), axis=1),
    }


def path_efficiency_index(trajlets: np.ndarray) -> np.ndarray:
    values = np.asarray(trajlets, dtype=float)
    numerator = np.linalg.norm(values[:, -1, :2] - values[:, 0, :2], axis=1)
    denominator = np.linalg.norm(np.diff(values[:, :, :2], axis=1), axis=2).sum(axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def angular_deviation(trajlets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port OpenTraj's trajectory-alignment angular-deviation computation."""
    values = np.asarray(trajlets, dtype=float)
    displacement = values[:, :, :2] - values[:, 0:1, :2]
    significant = np.linalg.norm(displacement, axis=2) > 0.25
    first_index = significant.argmax(axis=1)
    first = np.stack([displacement[index, first_index[index]] for index in range(len(values))])
    theta = np.arctan2(first[:, 0], first[:, 1])
    rotations = np.stack(
        [np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) for angle in theta]
    )
    aligned = np.matmul(rotations, displacement.transpose((0, 2, 1))).transpose((0, 2, 1))
    aligned = aligned[~np.any(np.isnan(aligned), axis=(1, 2))]
    keypoints = np.mean(aligned, axis=0)
    mean_degrees = np.rad2deg(np.arctan2(keypoints[:, 0], keypoints[:, 1]))
    std_degrees = np.std(np.rad2deg(np.arctan2(aligned[:, :, 0], aligned[:, :, 1])), axis=0)
    return mean_degrees, std_degrees


def dca_ttc_matrices(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact OpenTraj constant-relative-velocity DCA/TTCA equations."""
    state = np.asarray(state, dtype=float)
    differences = np.tile(state, (len(state), 1, 1))
    differences = differences - differences.transpose((1, 0, 2))
    dp, dv = differences[:, :, :2], differences[:, :, 2:]
    dot = dp[:, :, 0] * dv[:, :, 0] + dp[:, :, 1] * dv[:, :, 1]
    velocity_sq = dv[:, :, 0] ** 2 + dv[:, :, 1] ** 2 + 1e-6
    ttc = -dot / velocity_sq
    ttc[ttc < 0] = 0
    dca = np.linalg.norm(dp + ttc[:, :, None] * dv, axis=2)
    return dca, ttc


def frame_interactions(state: np.ndarray, radius: float = PEDESTRIAN_RADIUS_M) -> tuple[np.ndarray, np.ndarray]:
    if len(state) < 2 or not np.isfinite(state).all():
        return np.full(len(state), np.nan), np.full(len(state), np.nan)
    dca, ttc = dca_ttc_matrices(state)
    minimum_ttc = np.full(len(state), np.nan)
    minimum_dca = np.full(len(state), np.nan)
    for index in range(len(state)):
        collisions = ttc[index][(dca[index] < 2 * radius) & (ttc[index] > 0)]
        if len(collisions):
            minimum_ttc[index] = collisions.min()
        positive_dca = dca[index][dca[index] > 0]
        if len(positive_dca):
            minimum_dca[index] = positive_dca.min()
    return minimum_ttc, minimum_dca


def frame_local_density(xy: np.ndarray, a: float = 1.0) -> np.ndarray:
    """OpenTraj adaptive nearest-neighbour Gaussian local density."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return np.full(len(xy), np.nan)
    distances = cdist(xy, xy)
    bandwidth = np.array([
        np.min(row[row > 0.01]) if np.any(row > 0.01) else 0.01 for row in distances
    ])
    scaled_bandwidth_sq = (a * bandwidth) ** 2
    # OpenTraj weights contribution i using pedestrian i's nearest-neighbour
    # bandwidth, not the target pedestrian's scalar bandwidth.
    return np.array([
        np.sum(np.exp(-(distances[index] ** 2) / (2 * scaled_bandwidth_sq)) / scaled_bandwidth_sq)
        / (2 * np.pi)
        for index in range(len(xy))
    ])


def aligned_angular_sample(trajlet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one aligned path and its angles using OpenTraj's exact convention."""
    values=np.asarray(trajlet,dtype=float)
    displacement=values[:,:2]-values[0:1,:2]
    first_index=int((np.linalg.norm(displacement,axis=1)>0.25).argmax())
    first=displacement[first_index]
    theta=float(np.arctan2(first[0],first[1]))
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    aligned=(rotation@displacement.T).T
    angles=np.rad2deg(np.arctan2(aligned[:,0],aligned[:,1]))
    return aligned,angles


def global_density(count: int, scene_xy_bounds: tuple[float, float, float, float]) -> float:
    minimum_x, maximum_x, minimum_y, maximum_y = scene_xy_bounds
    area = (maximum_x - minimum_x) * (maximum_y - minimum_y)
    return count / area if area > 0 else float("nan")


def collision_energy(ttc: np.ndarray, upper: float = 3.0, lower: float = 0.2) -> np.ndarray:
    values = np.asarray(ttc, dtype=float)
    values = values[(values < upper) & (values > lower)]
    return (1.0 / values**2) * np.exp(-values / upper)


def gaussian_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    dimensions = len(x)
    return math.exp(-np.linalg.norm(x - y) ** 2 / (2 * bandwidth**2)) / (2 * math.pi * bandwidth**2) ** dimensions


def conditional_weights(observations: list[np.ndarray], index: int, bandwidth: float = 0.5) -> np.ndarray:
    values = np.array([gaussian_kernel(observations[index], item, bandwidth) for item in observations])
    return values / values.sum()


def conditional_trajectory_entropy(
    observations: list[np.ndarray],
    futures: list[np.ndarray],
    *,
    samples: int = 30,
    bandwidth: float = 0.5,
    seed: int = 0,
    replace: bool = True,
) -> np.ndarray:
    """OpenTraj's stochastic KDE entropy estimator with explicit deterministic seed."""
    rng = np.random.default_rng(seed)
    output = []
    covariance = bandwidth * np.eye(futures[0].size)
    for index in range(len(observations)):
        weights = conditional_weights(observations, index, bandwidth)
        chosen = rng.choice(len(futures), samples, p=weights, replace=replace or np.count_nonzero(weights) <= samples)
        sampled = [rng.multivariate_normal(futures[item].reshape(-1), covariance).reshape(futures[item].shape) for item in chosen]
        entropy = 0.0
        for sample in sampled:
            density = sum(weights[item] * gaussian_kernel(sample, futures[item], bandwidth) for item in range(len(futures)))
            entropy += -math.log(max(density, 1e-320))
            # This division inside the loop is intentionally retained from upstream.
            entropy /= samples
        output.append(entropy)
    return np.asarray(output)


def global_multimodality(endpoint_samples: np.ndarray, max_clusters: int = 31) -> int:
    from sklearn.mixture import GaussianMixture
    values = np.asarray(endpoint_samples, dtype=float)
    components = np.arange(1, min(len(values), max_clusters))
    if not len(components):
        return 0
    models = [GaussianMixture(int(n), covariance_type="full", random_state=0).fit(values) for n in components]
    return int(components[np.argmin([model.bic(values) for model in models])])
