"""SimpleNavSim: controller safety, pedestrian models, splits, metrics, CMA-ES, learned crowds."""
from __future__ import annotations

import math

import numpy as np
import torch

from simplenavsim.core.metrics import all_metrics
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import SPLIT_OFFSETS, Scenario, collate, make_scenario, scenario_set
from simplenavsim.core.state import Observation
from simplenavsim.environments.lightweight.env import LightweightEnv
from simplenavsim.optimization.cmaes import CMAESConfig, CMAESTrainer
from simplenavsim.optimization.evaluator import Evaluator
from simplenavsim.pedestrians.reactive_replay import ReactiveReplayModel
from simplenavsim.pedestrians.replay import ReplayModel
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


# ----------------------------------------------------------------------------- pedestrians


def _single(kind="head_on", seed=0, **kwargs):
    return collate([make_scenario(kind, seed, duration=20.0, **kwargs)])


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
    assert float(lateral[last]) < 0.1                                # back on its path
    lag = (batch.ped_ref[0, 0, : present.steps, 0] - present.ped_pos[0, 0, :, 0]).abs()
    assert float(lag[last]) < float(lag[valid].max())                # and catching up


def test_counterfactual_is_independent_of_the_robot_and_deterministic():
    batch = collate(scenario_set("train", ["dense_flow", "crossing"], 2, duration=15.0))
    env = LightweightEnv(ReactiveReplayModel())
    a, b = rollout(env, batch, robot_present=False), rollout(env, batch, robot_present=False)
    torch.testing.assert_close(a.ped_pos, b.ped_pos)


# ----------------------------------------------------------------------------- scenarios


def test_splits_use_disjoint_seeds():
    names = {split: {s.name for s in scenario_set(split, ["crossing"], 50)} for split in SPLIT_OFFSETS}
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"]) and not (names["val"] & names["test"])


def test_goal_seeking_reaches_the_goal_in_an_empty_world_efficiently():
    batch = collate(scenario_set("test", ["empty"], 3))
    env = LightweightEnv(ReplayModel())
    policy = GoalSeekingPolicy()
    run = rollout(env, batch, policy, policy.parameter_tensor({}, batch.size))
    m = all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance)
    assert bool(m["success"].all()) and float(m["path_efficiency"].min()) > 0.95
    # Turning to face the goal, accelerating and the slow final approach cost 2-3 s of a ~10 m run.
    assert 0.7 < float(m["time_efficiency"].min()) and float(m["time_efficiency"].max()) <= 1.0


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


def test_stars_crowd_runs_never_enters_the_robot_and_is_deterministic(tmp_path):
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

