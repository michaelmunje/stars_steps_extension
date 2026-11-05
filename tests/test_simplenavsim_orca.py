"""ORCA pedestrian model: collision-free crossing, non-reciprocal robot avoidance, the 2-D LP."""
from __future__ import annotations

import math

import numpy as np
import torch

from simplenavsim.core.scenarios import Scenario, _walker, collate
from simplenavsim.core.state import RobotState
from simplenavsim.pedestrians.orca import OrcaModel, linear_program2

DT = 0.1


def _scene(starts, goals, *, duration=40.0, speed=1.0, radius=0.25, t0=None, robot=(0.0, -50.0, 0.0)) -> Scenario:
    steps = int(round(duration / DT)) + 1
    t0 = t0 or [0.0] * len(starts)
    walkers = [_walker(s, g, speed, t, DT, steps) for s, g, t in zip(starts, goals, t0)]
    n = len(starts)
    return Scenario("orca", "custom", 0, DT, duration, np.array(robot, np.float32), np.array([0.0, -49.0], np.float32),
                    np.stack([w[0] for w in walkers]), np.stack([w[1] for w in walkers]),
                    np.full(n, radius, np.float32), np.full(n, speed, np.float32))


def _simulate(model, batch, steps=None, robot_pos=(0.0, -50.0), robot_vel=(0.0, 0.0), present=False):
    """Step ``model`` directly with a robot moving at constant velocity; returns pos, vel, active [B, P, K, ...]."""
    B = batch.size
    steps = steps or batch.steps - 1
    state = model.reset(batch)
    start = torch.tensor(robot_pos).expand(B, 2)
    velocity = torch.tensor(robot_vel).expand(B, 2)
    speed, heading = torch.linalg.norm(velocity, dim=-1), torch.atan2(velocity[:, 1], velocity[:, 0])
    present = torch.as_tensor(present).expand(B)
    pos, vel, active, robots = [state.pos], [state.vel], [state.active], [start]
    for k in range(1, steps + 1):
        robot = RobotState(torch.cat([start + velocity * (k - 1) * DT, heading[:, None]], -1), speed, torch.zeros(B))
        state = model.step(state, batch, k, robot, 0.3, present)
        pos.append(state.pos), vel.append(state.vel), active.append(state.active), robots.append(robot.pos)
    return torch.stack(pos, 2), torch.stack(vel, 2), torch.stack(active, 2), torch.stack(robots, 1), state


def _min_gap(pos, active, radius):
    """Smallest surface gap between two simultaneously active pedestrians."""
    gap = torch.linalg.norm(pos[:, :, None] - pos[:, None], dim=-1) - (radius[:, :, None] + radius[:, None])[..., None]
    P = pos.shape[1]
    pair = active[:, :, None] & active[:, None] & ~torch.eye(P, dtype=torch.bool)[None, :, :, None]
    return float(gap[pair].min())


def test_eight_pedestrian_antipodal_swap_is_collision_free():
    angles = np.arange(8) * 2 * math.pi / 8
    circle = 4.0 * np.stack([np.cos(angles), np.sin(angles)], -1)
    jitter = np.random.default_rng(1).uniform(-0.02, 0.02, (8, 2))  # exact symmetry deadlocks ORCA (also RVO2)
    batch = collate([_scene(circle + jitter, -circle, duration=50.0)])
    pos, vel, active, _, state = _simulate(OrcaModel(mode="goal"), batch)
    assert _min_gap(pos, active, batch.ped_radius) >= -1e-3
    assert bool(state.done.all())


def test_pedestrians_avoid_a_robot_moving_through_them_and_ignore_an_absent_one():
    standing = [(0.0, 0.1), (1.5, -0.35), (3.0, 0.2), (5.0, -0.1)]  # all in the robot's path
    batch = collate([_scene(standing, standing, duration=12.0)])
    model = OrcaModel(mode="reactive_replay", tracking_gain=0.5)
    pos, _, active, robot, _ = _simulate(model, batch, robot_pos=(-4.0, 0.0), robot_vel=(1.0, 0.0), present=True)
    gap = torch.linalg.norm(pos - robot[:, None], dim=-1) - batch.ped_radius[..., None] - 0.3
    assert float(gap[active].min()) >= -1e-3            # the robot never yields, the pedestrians do
    moved = torch.linalg.norm(pos - pos[:, :, :1], dim=-1).amax(-1)
    assert bool((moved[0] > 0.2).all())                 # each stepped aside
    pos_absent, *_ = _simulate(model, batch, robot_pos=(-4.0, 0.0), robot_vel=(1.0, 0.0), present=False)
    torch.testing.assert_close(pos_absent, pos_absent[:, :, :1].expand_as(pos_absent))


def test_linear_program2_matches_a_brute_force_optimum():
    torch.manual_seed(0)
    N, L = 64, 5
    direction = torch.nn.functional.normalize(torch.randn(N, L, 2), dim=-1)
    point = 0.3 * torch.randn(N, L, 2)
    valid = torch.ones(N, L, dtype=torch.bool)
    opt, radius = 2.0 * torch.randn(N, 2), torch.full((N,), 1.5)
    fail, result = linear_program2(point, direction, valid, radius, opt)
    grid = torch.stack(torch.meshgrid(torch.linspace(-1.5, 1.5, 301), torch.linspace(-1.5, 1.5, 301), indexing="ij"), -1)
    grid = grid.reshape(-1, 2)
    inside = torch.linalg.norm(grid, dim=-1) <= 1.5
    for n in torch.nonzero(fail == L).squeeze(1).tolist():
        det = direction[n, :, 0] * (point[n, :, 1] - grid[:, None, 1]) - direction[n, :, 1] * (point[n, :, 0] - grid[:, None, 0])
        feasible = inside & (det <= 0).all(-1)
        if feasible.any():
            best = torch.linalg.norm(grid[feasible] - opt[n], dim=-1).min()
            assert float(torch.linalg.norm(result[n] - opt[n])) <= float(best) + 1e-6
    assert bool((fail == L).any())

