"""ORCA pedestrian model (simplenavsim.pedestrians.orca): collision-free crossing, robot avoidance, batching."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from simplenavsim.core.config import make_pedestrian_model
from simplenavsim.core.geometry import obstacle_clearance
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import Scenario, _walker, collate, scenario_set
from simplenavsim.core.state import RobotState
from simplenavsim.environments.lightweight.env import LightweightEnv
from simplenavsim.pedestrians.orca import OrcaModel, OrcaParams, linear_program2, orca_velocities
from simplenavsim.policies.goal_seeking import GoalSeekingPolicy

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


def test_two_pedestrians_head_on_pass_without_overlap_and_arrive():
    batch = collate([_scene([(-4.0, 0.05), (4.0, 0.0)], [(4.0, 0.05), (-4.0, 0.0)], duration=20.0)])
    pos, vel, active, _, state = _simulate(OrcaModel(mode="goal"), batch)
    assert _min_gap(pos, active, batch.ped_radius) >= -1e-3
    assert bool(state.done.all())
    lateral = pos[0, :, :, 1].abs().max()
    assert float(lateral) > 0.2  # they sidestepped rather than passing through each other


def test_eight_pedestrian_antipodal_swap_is_collision_free():
    angles = np.arange(8) * 2 * math.pi / 8
    circle = 4.0 * np.stack([np.cos(angles), np.sin(angles)], -1)
    jitter = np.random.default_rng(1).uniform(-0.02, 0.02, (8, 2))  # exact symmetry deadlocks ORCA (also RVO2)
    batch = collate([_scene(circle + jitter, -circle, duration=50.0)])
    pos, vel, active, _, state = _simulate(OrcaModel(mode="goal"), batch)
    assert _min_gap(pos, active, batch.ped_radius) >= -1e-3
    assert bool(state.done.all())


def test_speeds_never_exceed_the_limit():
    batch = collate(scenario_set("train", ["dense_flow"], 2, duration=15.0))
    for mode in ("goal", "reactive_replay"):
        model = OrcaModel(mode=mode)
        _, vel, active, _, _ = _simulate(model, batch, robot_pos=(0.0, 0.0), robot_vel=(0.8, 0.0), present=True)
        speed = torch.linalg.norm(vel, dim=-1)
        ref_speed = torch.linalg.norm(batch.ped_ref[:, :, 1:] - batch.ped_ref[:, :, :-1], dim=-1).amax(-1) / DT
        base = batch.ped_speed if mode == "goal" else torch.maximum(batch.ped_speed, ref_speed)
        assert bool((speed <= model.params.max_speed_factor * base[..., None] + 1e-4).all())


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


def test_robot_avoidance_is_not_reciprocal():
    """A walker facing an oncoming robot takes the whole avoidance; facing an identical pedestrian, half."""
    p = OrcaParams()
    pos = torch.tensor([[[0.0, 0.0], [2.0, 0.05]]])
    vel = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    radius, limit = torch.full((1, 2), 0.25), torch.full((1, 2), 1.3)
    both = torch.tensor([[True, True]])
    reciprocal = orca_velocities(pos, vel, vel, both, radius, limit, DT, p)[0, 0]
    only = torch.tensor([[True, False]])
    against_robot = orca_velocities(pos, vel, vel, only, radius, limit, DT, p, robot_pos=pos[:, 1], robot_vel=vel[:, 1],
                                    robot_radius=0.25, robot_present=torch.tensor([True]))[0, 0]
    deviation = lambda v: float(torch.linalg.norm(v - vel[0, 0]))
    assert deviation(against_robot) == pytest.approx(2 * deviation(reciprocal), rel=0.05)


def test_inactive_and_padding_pedestrians_are_ignored():
    starts, goals = [(-4.0, 0.05), (4.0, 0.0)], [(4.0, 0.05), (-4.0, 0.0)]
    alone = collate([_scene(starts, goals, duration=12.0)])
    # A third pedestrian standing right in the way, but only appearing after the run ends.
    late = collate([_scene(starts + [(0.0, 0.0)], goals + [(0.0, 1.0)], duration=12.0, t0=[0.0, 0.0, 11.5])])
    crowd = _scene([(x, 3.0) for x in range(-3, 4)], [(x, -3.0) for x in range(-3, 4)], duration=12.0)
    padded = collate([_scene(starts, goals, duration=12.0), crowd])             # row 0 padded to 7 pedestrians
    model = OrcaModel(mode="goal")
    reference, *_ = _simulate(model, alone, steps=100)
    with_late, *_ = _simulate(model, late, steps=100)
    batched, *_ = _simulate(model, padded, steps=100)
    torch.testing.assert_close(with_late[:, :2], reference)
    torch.testing.assert_close(batched[:1, :2], reference)


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


@pytest.mark.parametrize("mode", ["reactive_replay", "goal"])
def test_orca_runs_in_lightweight_rollouts(mode):
    batch = collate(scenario_set("train", ["dense_flow", "narrow_passage"], 2, duration=15.0))
    env = LightweightEnv(make_pedestrian_model({"model": "orca", "orca": {"mode": mode}}))
    assert isinstance(env.pedestrians, OrcaModel) and env.pedestrians.params.mode == mode
    policy = GoalSeekingPolicy()
    run = rollout(env, batch, policy, policy.parameter_tensor({}, batch.size))
    assert bool(torch.isfinite(run.ped_pos).all())
    cf_a, cf_b = rollout(env, batch, robot_present=False), rollout(env, batch, robot_present=False)
    torch.testing.assert_close(cf_a.ped_pos, cf_b.ped_pos)
    assert _min_gap(cf_a.ped_pos, cf_a.ped_active, batch.ped_radius) >= -0.05
    B, P, K, _ = run.ped_pos.shape
    clearance, _ = obstacle_clearance(run.ped_pos.reshape(B, P * K, 2), batch.circles, batch.circle_mask,
                                      batch.segments, batch.segment_mask)
    clearance = clearance.reshape(B, P, K) - batch.ped_radius[..., None]
    assert float(clearance[run.ped_active].min()) >= -0.05                  # walls are respected
    step = torch.linalg.norm(run.ped_pos[:, :, 1:] - run.ped_pos[:, :, :-1], dim=-1)
    both = run.ped_active[:, :, 1:] & run.ped_active[:, :, :-1]
    assert float(step[both].max()) < 2.0 * 1.4 * 1.3 * DT                     # no teleporting
