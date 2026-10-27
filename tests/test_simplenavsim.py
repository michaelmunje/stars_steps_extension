"""SimpleNavSim lightweight simulator: policies, pedestrian models, scenarios, metrics, data, CMA-ES."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from simplenavsim.core.metrics import all_metrics, summarize
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import KINDS, SPLIT_OFFSETS, SYNTHETIC_KINDS, Scenario, collate, make_scenario, scenario_set
from simplenavsim.core.state import Observation
from simplenavsim.datasets.trajectory_loader import RecordingPool, Recording, Track, load_csv, resample, write_csv
from simplenavsim.environments.lightweight.env import LightweightEnv, integrate_unicycle
from simplenavsim.optimization.cmaes import CMAESConfig, CMAESTrainer
from simplenavsim.optimization.evaluator import Evaluator
from simplenavsim.pedestrians.reactive_replay import ReactiveReplayModel
from simplenavsim.pedestrians.replay import ReplayModel
from simplenavsim.pedestrians.social_force import SocialForceModel
from simplenavsim.policies.goal_seeking import GoalSeekingPolicy
from simplenavsim.policies.social_navigation import ControllerConfig, SocialNavigationPolicy


def _obs(robot=(0.0, 0.0, 0.0), goal=(5.0, 0.0), peds=(), vels=None, segments=(), v=0.0):
    P = max(1, len(peds))
    ped_pos = torch.zeros(1, P, 2)
    ped_vel = torch.zeros(1, P, 2)
    visible = torch.zeros(1, P, dtype=torch.bool)
    for i, p in enumerate(peds):
        ped_pos[0, i] = torch.tensor(p)
        ped_vel[0, i] = torch.tensor((vels or [(0.0, 0.0)] * len(peds))[i])
        visible[0, i] = True
    seg = torch.tensor(segments, dtype=torch.float32).reshape(1, -1, 4)
    return Observation(torch.tensor([robot]), torch.tensor([v]), torch.zeros(1), torch.tensor([goal]), ped_pos, ped_vel,
                       visible, torch.full((1, P), 0.25), torch.zeros(1, 0, 3), torch.zeros(1, 0, dtype=torch.bool),
                       seg, torch.ones(1, seg.shape[1], dtype=torch.bool), torch.zeros(1, 2))


def _act(policy, obs, **values):
    return policy.act(obs, policy.parameter_tensor({**policy.default_parameters(), **values}, 1))[0]


# ----------------------------------------------------------------------------- policies


@pytest.mark.parametrize("policy_cls", [SocialNavigationPolicy, GoalSeekingPolicy])
def test_policy_outputs_are_within_robot_limits(policy_cls):
    policy = policy_cls()
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = _obs(robot=tuple(rng.uniform(-3, 3, 3)), goal=tuple(rng.uniform(-5, 5, 2)),
                   peds=[tuple(rng.uniform(-4, 4, 2)) for _ in range(3)], v=float(rng.uniform(0, 1)))
        v, w = _act(policy, obs)
        c = policy.config
        assert 0.0 <= float(v) <= c.max_speed + 1e-6 and abs(float(w)) <= c.max_angular_speed + 1e-6


def test_policy_drives_toward_an_unobstructed_goal():
    v, w = _act(SocialNavigationPolicy(), _obs(goal=(5.0, 0.0)))
    assert float(v) > 0.5 and abs(float(w)) < 0.4


def test_policy_turns_away_from_a_wall_in_front():
    wall = [(1.0, -2.0, 1.0, 2.0)]  # blocks the straight line to the goal at x=1
    for policy in (SocialNavigationPolicy(), GoalSeekingPolicy()):
        obs = _obs(goal=(5.0, 0.0), segments=wall, v=0.6)
        v, w = _act(policy, obs)
        # Must not keep driving straight into the wall within the horizon.
        pose = obs.robot_pose
        for _ in range(10):
            pose = integrate_unicycle(pose, v[None], w[None], 0.1)
        assert float(pose[0, 0]) < 1.0 - 0.3


def test_social_policy_yields_to_an_approaching_pedestrian_and_goal_seeking_does_not():
    obs = _obs(goal=(6.0, 0.0), peds=[(2.0, 0.0)], vels=[(-1.2, 0.0)], v=0.8)
    v_social, w_social = _act(SocialNavigationPolicy(), obs)
    v_goal, w_goal = _act(GoalSeekingPolicy(), obs)
    assert float(v_goal) > 0.8 and abs(float(w_goal)) < 0.2       # blind baseline keeps going
    assert float(v_social) < float(v_goal) or abs(float(w_social)) > 0.3


def test_emergency_stop_when_no_command_is_safe():
    # Surrounded closely by pedestrians closing in from every side.
    ring = [(0.62 * math.cos(a), 0.62 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    vels = [(-0.5 * x / 0.62, -0.5 * y / 0.62) for x, y in ring]
    policy = SocialNavigationPolicy()
    v, w = _act(policy, _obs(peds=ring, vels=vels))
    assert float(v) == 0.0 and float(w) == 0.0 and bool(policy.last_info["emergency_stop"][0])


def test_parameter_specs_round_trip_and_bounds():
    for spec in SocialNavigationPolicy.parameters:
        for value in (spec.lower, spec.default, spec.upper):
            assert spec.from_unit(spec.to_unit(value)) == pytest.approx(value, rel=1e-6)
        assert spec.from_unit(-1.0) == pytest.approx(spec.lower) and spec.from_unit(2.0) == pytest.approx(spec.upper)


# ----------------------------------------------------------------------------- pedestrians


def _single(kind="head_on", seed=0, **kwargs):
    return collate([make_scenario(kind, seed, duration=20.0, **kwargs)])


def test_replay_reproduces_the_reference_and_ignores_the_robot():
    batch = _single("crossing")
    env = LightweightEnv(ReplayModel())
    run = rollout(env, batch, SocialNavigationPolicy(), SocialNavigationPolicy().parameter_tensor({}, 1))
    active = run.ped_active[0, 0]
    torch.testing.assert_close(run.ped_pos[0, 0][active], batch.ped_ref[0, 0, : run.steps][active])
    cf = rollout(env, batch, robot_present=False)
    m = all_metrics(run, cf, batch, env.robot.radius)
    assert float(m["disruption_displacement"][0]) == 0.0


@pytest.mark.parametrize("model", [SocialForceModel(), ReactiveReplayModel()])
def test_reactive_pedestrians_avoid_a_stationary_robot_smoothly(model):
    batch = _single("head_on", 1)
    env = LightweightEnv(model)
    run = rollout(env, batch, GoalSeekingPolicy(), torch.zeros(1, len(GoalSeekingPolicy().parameter_names)))  # parameters irrelevant: stays still
    stay = rollout(env, batch, policy=None, robot_present=True)                  # robot never moves
    gap = torch.linalg.norm(stay.ped_pos[0, 0] - batch.robot_start[0, :2], dim=-1)[stay.ped_active[0, 0]]
    assert float(gap.min()) > 0.25 + 0.3 - 0.1  # walks around the robot instead of through it
    step = torch.linalg.norm(stay.ped_pos[0, 0, 1:] - stay.ped_pos[0, 0, :-1], dim=-1)
    both = stay.ped_active[0, 0, 1:] & stay.ped_active[0, 0, :-1]
    assert float(step[both].max()) < 1.3 * float(batch.ped_speed[0, 0]) * batch.dt * 1.5  # no teleporting
    del run


def test_reactive_replay_tracks_the_reference_without_a_robot_and_recovers_after_a_detour():
    batch = _single("head_on", 2)
    env = LightweightEnv(ReactiveReplayModel())
    absent = rollout(env, batch, robot_present=False)
    active = absent.ped_active[0, 0] & batch.ped_valid[0, 0, : absent.steps]
    error = torch.linalg.norm(absent.ped_pos[0, 0] - batch.ped_ref[0, 0, : absent.steps], dim=-1)[active]
    assert float(error.max()) < 0.05
    # A stationary robot in the way of a long straight walk: detour, then return to the path.
    t = np.arange(201) * 0.1
    walk = np.stack([6.0 - 1.0 * t, np.full_like(t, 0.05)], -1).astype(np.float32)[None]
    scenario = Scenario("detour", "custom", 0, 0.1, 20.0, np.array([0, 0, 0], np.float32), np.array([0, 9], np.float32),
                        walk, (t <= 14.0)[None], np.array([0.25], np.float32), np.array([1.0], np.float32))
    batch = collate([scenario])
    present = rollout(env, batch, policy=None)
    deviation = torch.linalg.norm(present.ped_pos[0, 0] - batch.ped_ref[0, 0, : present.steps], dim=-1)
    valid = present.ped_active[0, 0] & batch.ped_valid[0, 0, : present.steps]
    assert float(deviation[valid].max()) > 0.2                       # it detoured
    last = int(torch.nonzero(valid).max())
    lateral = (present.ped_pos[0, 0, :, 1] - batch.ped_ref[0, 0, : present.steps, 1]).abs()
    assert float(lateral[last]) < 0.1                                # back on its path (any lag is ped_delay)
    lag = (batch.ped_ref[0, 0, : present.steps, 0] - present.ped_pos[0, 0, :, 0]).abs()
    assert float(lag[last]) < float(lag[valid].max())                # and catching up


def test_counterfactual_is_independent_of_the_robot_and_deterministic():
    batch = collate(scenario_set("train", ["dense_flow", "crossing"], 2, duration=15.0))
    env = LightweightEnv(ReactiveReplayModel())
    a, b = rollout(env, batch, robot_present=False), rollout(env, batch, robot_present=False)
    torch.testing.assert_close(a.ped_pos, b.ped_pos)


# ----------------------------------------------------------------------------- scenarios


def test_every_synthetic_kind_builds_and_is_seed_deterministic():
    for kind in SYNTHETIC_KINDS:
        a, b = make_scenario(kind, 7), make_scenario(kind, 7)
        np.testing.assert_array_equal(a.ped_ref, b.ped_ref)
        np.testing.assert_array_equal(a.robot_goal, b.robot_goal)
        assert a.ped_ref.shape[1] == a.steps if len(a.ped_radius) else True
    assert make_scenario("crossing", 1).ped_ref.tolist() != make_scenario("crossing", 2).ped_ref.tolist()
    assert len(KINDS) == 10


def test_splits_use_disjoint_seeds():
    names = {split: {s.name for s in scenario_set(split, ["crossing"], 50)} for split in SPLIT_OFFSETS}
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"]) and not (names["val"] & names["test"])


def test_goal_seeking_reaches_the_goal_in_an_empty_world():
    batch = collate(scenario_set("test", ["empty"], 3))
    env = LightweightEnv(ReplayModel())
    policy = GoalSeekingPolicy()
    run = rollout(env, batch, policy, policy.parameter_tensor({}, batch.size))
    metrics = summarize(all_metrics(run, None, batch, env.robot.radius))
    assert metrics["success"] == 1.0 and metrics["collision"] == 0.0
    assert metrics["path_length"] == pytest.approx(float(torch.linalg.norm(batch.robot_goal - batch.robot_start[:, :2], dim=-1).mean()), rel=0.15)


def test_time_efficiency_is_near_one_driving_straight_and_zero_on_failure():
    batch = collate(scenario_set("test", ["empty"], 3))
    env = LightweightEnv(ReplayModel())
    policy = GoalSeekingPolicy()
    run = rollout(env, batch, policy, policy.parameter_tensor({}, batch.size))
    m = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed)
    # Even in an empty world, turning to face the goal, accelerating from rest and the slow final
    # approach cost 2-3 s of a ~10 m run (about 0.75-0.8).
    assert bool((m["time_efficiency"] > 0.7).all()) and bool((m["time_efficiency"] <= 1.0).all())
    faster_robot = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed * 2.0)
    assert torch.allclose(faster_robot["time_efficiency"], m["time_efficiency"] / 2.0)
    run.success[:] = False
    assert float(all_metrics(run, None, batch, env.robot.radius)["time_efficiency"].abs().max()) == 0.0


def test_time_to_goal_objective_term_charges_slow_arrivals_only_when_weighted():
    from simplenavsim.optimization.evaluator import Objective
    metrics = {key: torch.zeros(2) for key in ("collision", "normalized_time", "personal_violation_rate", "command_smoothness")}
    metrics.update(success=torch.ones(2), path_efficiency=torch.ones(2), time_efficiency=torch.tensor([1.0, 0.6]))
    assert torch.equal(Objective()(metrics), torch.ones(2))
    assert torch.allclose(Objective(time_to_goal=0.25)(metrics), torch.tensor([1.0, 0.9]))


def test_default_arc_grid_is_dense_and_includes_driving_straight():
    commands = SocialNavigationPolicy().candidates("cpu")
    forward = commands[commands[:, 0] > 0]
    assert commands.shape[0] == 12 * 31                                     # 11 speeds + reverse, 31 turn rates
    assert bool(((forward[:, 0] - 1.0).abs() < 1e-6).any() & (forward[:, 1].abs() < 1e-6).any())
    turn_rates = commands[:, 1].unique()
    assert float((turn_rates[1:] - turn_rates[:-1]).max()) == pytest.approx(0.1, abs=1e-5)


def test_collision_is_detected_and_ends_the_episode():
    walker = np.repeat(np.array([[3.0, 0.0]], np.float32), 101, 0)[None]
    scenario = Scenario("block", "custom", 0, 0.1, 10.0, np.array([0, 0, 0], np.float32), np.array([6, 0], np.float32),
                        walker, np.ones((1, 101), bool), np.array([0.25], np.float32), np.array([1.0], np.float32))
    batch = collate([scenario])
    env = LightweightEnv(ReplayModel())
    policy = GoalSeekingPolicy()  # blind to pedestrians: drives into the standing person
    run = rollout(env, batch, policy, policy.parameter_tensor({}, 1))
    assert bool(run.ped_collision[0]) and not bool(run.success[0])
    assert float(run.robot_pose[0, -1, 0]) < 3.0


# ----------------------------------------------------------------------------- data


def test_csv_round_trip_and_resampling(tmp_path):
    t = np.arange(0, 10, 0.4)
    recording = Recording("rec", {"a": Track(t, np.stack([t, 0 * t], -1)), "b": Track(t[5:], np.stack([0 * t[5:], t[5:]], -1))})
    write_csv([recording], tmp_path / "x.csv")
    loaded = load_csv(tmp_path / "x.csv")[0]
    np.testing.assert_allclose(loaded.tracks["a"].xy, recording.tracks["a"].xy, atol=1e-4)
    xy, inside = resample(loaded.tracks["b"], 0.0, 0.1, 101)
    assert not inside[:20].any() and inside[21:96].all()
    np.testing.assert_allclose(xy[50], [0.0, 5.0], atol=1e-3)


def test_recording_pool_splits_by_recording_and_builds_scenarios():
    t = np.arange(0, 60, 0.4)
    recordings = [Recording(f"r{i}", {str(j): Track(t, np.stack([t * 0.5 + j, np.full_like(t, j * 0.7)], -1))
                                      for j in range(5)}) for i in range(6)]
    splits = {"r0": "train", "r1": "train", "r2": "val", "r3": "val", "r4": "test", "r5": "test"}
    pool = RecordingPool(recordings, splits)
    scenario = make_scenario("recorded", 3, pool=pool, split="val", duration=20.0)
    assert scenario.meta["recording"] in {"r2", "r3"} and len(scenario.ped_radius) >= 3


# ----------------------------------------------------------------------------- optimisation


def test_cmaes_runs_resumes_and_never_selects_worse_than_default(tmp_path):
    env = LightweightEnv(ReactiveReplayModel())
    policy = SocialNavigationPolicy(ControllerConfig(num_speeds=4, num_turn_rates=7))
    evaluator = Evaluator(env, policy)
    config = CMAESConfig(optimize=["goal_weight", "preferred_speed"], population_size=4, num_generations=2,
                         kinds=["crossing"], train_per_kind=1, val_per_kind=1, validate_every=1, duration=12.0)
    trainer = CMAESTrainer(config, evaluator, tmp_path)
    summary = trainer.run()
    default = [line for line in (tmp_path / "generations.jsonl").read_text().splitlines() if '"default"' in line]
    assert default and summary["best"]["val_objective"] >= __import__("json").loads(default[0])["val_objective"]
    config.num_generations = 3
    resumed = CMAESTrainer(config, evaluator, tmp_path).run(resume=True)
    assert resumed["generations"] == 3


def test_stars_crowd_model_runs_reacts_to_the_robot_and_is_deterministic(tmp_path):
    from baselines.stars_ma_model import StarsMultiAgentForecaster
    from simplenavsim.pedestrians.stars_crowd import StarsCrowdModel

    torch.manual_seed(0)
    model = StarsMultiAgentForecaster(8, 12, dt=0.4, encoder_recipe={"hidden_dim": 32, "latent_dim": 16,
                                      "edge_transformer_layers": 1, "node_transformer_layers": 1},
                                      decoder_recipe={"hidden_dim": 32, "latent_dim": 4})
    path = tmp_path / "forecaster.pt"
    torch.save({"model": "stars_ma", "phase": "forecast", "state_dict": model.state_dict(),
                "encoder_recipe": model.encoder.recipe, "decoder_recipe": model.decoder_recipe,
                "graph": {"knn": 16, "radius": 10.0}, "window": {"observation_steps": 8, "prediction_steps": 12, "dt": 0.4},
                "run_config": {"data_seed": 42}}, path)
    batch = collate(scenario_set("test", ["dense_flow", "head_on"], 1, duration=12.0))
    env = LightweightEnv(StarsCrowdModel(path, seed=3))
    absent_a = rollout(env, batch, robot_present=False)
    absent_b = rollout(env, batch, robot_present=False)
    torch.testing.assert_close(absent_a.ped_pos, absent_b.ped_pos)               # seeded intents
    present = rollout(env, batch, policy=None)                                   # a stationary robot
    gap = torch.linalg.norm(present.ped_pos - batch.robot_start[:, None, None, :2], dim=-1) - batch.ped_radius[..., None] - 0.3
    assert float(gap[present.ped_active].min()) >= -1e-4                         # never inside the robot
    model_driven = present.ped_active.clone()
    model_driven[:, :, :40] = False                                              # after the 3.2 s warm-up
    assert bool(model_driven.any())


def test_wall_crossings_slide_along_the_wall_or_stop_at_corners():
    from simplenavsim.core.geometry import block_wall_crossings

    wall = torch.tensor([[[0.0, -1.0, 0.0, 1.0], [-1.0, 1.0, 1.0, 1.0]]])     # x = 0 and y = 1
    mask = torch.tensor([[True, True]])
    old = torch.tensor([[[-0.1, 0.0], [-0.1, 0.0], [-0.1, 0.9], [-2.0, 0.0]]])
    new = torch.tensor([[[0.1, 0.1], [-0.2, 0.1], [0.1, 1.1], [-1.9, 0.0]]])
    out = block_wall_crossings(old, new, wall, mask)
    assert torch.allclose(out[0, 0], torch.tensor([-0.1, 0.1]))                # slides along x = 0
    assert torch.allclose(out[0, 1], new[0, 1])                                # no crossing: unchanged
    assert torch.allclose(out[0, 2], old[0, 2])                                # corner: stays
    assert torch.allclose(out[0, 3], new[0, 3])
    assert torch.equal(block_wall_crossings(old, new, wall, torch.zeros_like(mask)), new)


def test_learned_policy_shapes_weights_and_shield():
    from simplenavsim.core.state import Observation
    from simplenavsim.policies.learned import OBS_DIM, LearnedPolicy

    policy = LearnedPolicy()
    theta = policy.init_weights(0)
    assert theta.size == policy.dim
    B = 3
    obs = Observation(robot_pose=torch.zeros(B, 3), robot_v=torch.zeros(B), robot_w=torch.zeros(B),
                      goal=torch.tensor([[5.0, 0.0]] * B), ped_pos=torch.tensor([[[2.0, 0.0]]] * B),
                      ped_vel=torch.zeros(B, 1, 2), ped_visible=torch.ones(B, 1, dtype=torch.bool),
                      ped_radius=torch.full((B, 1), 0.25), circles=torch.zeros(B, 0, 3),
                      circle_mask=torch.zeros(B, 0, dtype=torch.bool),
                      segments=torch.tensor([[[0.5, -1.0, 0.5, 1.0]]] * B), segment_mask=torch.ones(B, 1, dtype=torch.bool),
                      last_command=torch.zeros(B, 2))
    features, _ = policy.features(obs)
    assert features.shape == (B, OBS_DIM) and torch.isfinite(features).all()
    params = policy.parameter_tensor([theta, theta * 0, theta], B)
    command = policy.act(obs, params)
    assert command.shape == (B, 2)
    # Zero weights -> raw output 0 -> v = midpoint (0.35 m/s) straight ahead, into the wall 0.5 m away.
    full_speed = torch.tensor([[1.0, 0.0]] * B)
    assert policy._shield(obs, full_speed, None).all()
    assert torch.allclose(policy.to_command(policy.from_command(full_speed * 0.5)), full_speed * 0.5, atol=1e-4)


def test_learned_shield_fallback_avoids_instead_of_stopping():
    from simplenavsim.core.state import Observation
    from simplenavsim.policies.learned import LearnedConfig, LearnedPolicy

    policy = LearnedPolicy(LearnedConfig(shield_fallback=True))
    obs = Observation(robot_pose=torch.zeros(1, 3), robot_v=torch.ones(1), robot_w=torch.zeros(1),
                      goal=torch.tensor([[5.0, 0.0]]), ped_pos=torch.tensor([[[1.2, 0.0]]]),
                      ped_vel=torch.zeros(1, 1, 2), ped_visible=torch.ones(1, 1, dtype=torch.bool),
                      ped_radius=torch.full((1, 1), 0.25), circles=torch.zeros(1, 0, 3),
                      circle_mask=torch.zeros(1, 0, dtype=torch.bool), segments=torch.zeros(1, 0, 4),
                      segment_mask=torch.zeros(1, 0, dtype=torch.bool), last_command=torch.zeros(1, 2))
    future = obs.ped_pos[:, :, None].expand(1, 1, 10, 2)
    straight = torch.tensor([[1.0, 0.0]])
    assert policy._shield(obs, straight, future).all()
    safe, none_safe = policy._fallback(obs, straight, future)
    assert not none_safe.any() and not policy._shield(obs, safe, future).any()


def test_contextual_policy_starts_exactly_at_its_base_controller():
    from simplenavsim.core.state import Observation
    from simplenavsim.policies.contextual import ContextualPolicy
    from simplenavsim.policies.social_navigation import SocialNavigationPolicy

    controller = SocialNavigationPolicy()
    base = {"goal_weight": 3.0, "pedestrian_weight": 5.0, "preferred_speed": 0.9}
    policy = ContextualPolicy(controller, base)
    B = 2
    obs = Observation(robot_pose=torch.zeros(B, 3), robot_v=torch.full((B,), 0.5), robot_w=torch.zeros(B),
                      goal=torch.tensor([[6.0, 1.0], [4.0, -2.0]]), ped_pos=torch.tensor([[[2.0, 0.3]], [[1.5, -0.5]]]),
                      ped_vel=torch.tensor([[[-1.0, 0.0]], [[0.0, 1.0]]]), ped_visible=torch.ones(B, 1, dtype=torch.bool),
                      ped_radius=torch.full((B, 1), 0.25), circles=torch.zeros(B, 0, 3),
                      circle_mask=torch.zeros(B, 0, dtype=torch.bool), segments=torch.zeros(B, 0, 4),
                      segment_mask=torch.zeros(B, 0, dtype=torch.bool), last_command=torch.zeros(B, 2))
    theta = policy.parameter_tensor(policy.init_weights(0), B)
    expected = controller.act(obs, controller.parameter_tensor({**controller.default_parameters(), **base}, B))
    assert torch.allclose(policy.act(obs, theta), expected)
    nudged = theta.clone()
    nudged[:, -len(policy.tuned):] = 3.0                      # output biases: every tuned setting shifted up
    params = policy.controller_parameters(obs, nudged)
    assert (params[:, controller.parameter_names.index("goal_weight")] > 3.0).all()
