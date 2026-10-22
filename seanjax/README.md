# SEAN-JAX: social navigation simulation and policy tuning

SEAN-JAX is the successor to [SEAN 2.0](https://github.com/yale-sean/social_sim_unity): a robot
navigates among pedestrians, and a parameterised navigation policy is tuned with CMA-ES. The aim is
to reach the goal while minimising disruption to nearby pedestrians.

This is **Phase 1–2: a lightweight, batched 2-D backend** in PyTorch (the name notwithstanding;
JAX is optional in the design). Every environment is one row of a tensor batch, on CPU or GPU, so a
whole CMA-ES population × scenario set runs as a single simulation. Isaac Sim / Isaac Lab
(Phase 3) and ROS 2 (Phase 4) are not implemented yet; see [Status](#status).

Code lives in `src/seanjax/` (installed with the repository: `uv sync`), configs in
`seanjax/configs/`, command-line tools in `seanjax/scripts/`, tests in `tests/test_seanjax.py`.

## Quick start

From the repository root, after `uv sync` (and `source .venv/bin/activate`):

```bash
# One scenario, one policy: prints metrics, writes a PNG and a GIF
python seanjax/scripts/run_sim.py --backend lightweight --scenario crossing --policy social

# Tune the social policy with CMA-ES (writes checkpoints/seanjax/cmaes/)
python seanjax/scripts/train_cmaes.py --backend lightweight --config seanjax/configs/cmaes.yaml \
  --num-generations 100 --population-size 24 --seed 42

# Compare baselines and the tuned policy on the untouched test scenarios
python seanjax/scripts/evaluate.py --policy checkpoints/seanjax/cmaes/best_policy.yaml --split test

# Plot / animate a policy on one scenario, with the robot-absent counterfactual pedestrians
python seanjax/scripts/visualize_rollout.py --checkpoint checkpoints/seanjax/cmaes/best_policy.yaml \
  --scenario crossing --run-dir checkpoints/seanjax/cmaes
```

All scripts take `--device` (`cuda`, `cpu`; default: CUDA if available) and `--pedestrians`
(`replay`, `social_force`, `reactive_replay`). `seanjax/configs/cmaes_recorded.yaml` adds recorded
STEPS traffic to the scenario mix.

On TACC Horizon, run tuning on a GPU node, for example:
`sbatch scripts/horizon/gpu_jobs.slurm "python seanjax/scripts/train_cmaes.py --config seanjax/configs/cmaes.yaml --device cuda"`.
A generation of 24 candidates × 32 scenarios (768 environments, 30 s episodes) takes about 3 s on
one GB200.

## Components

| Module | Contents |
| --- | --- |
| `core/state.py` | Batched `ScenarioBatch`, `RobotState`, `PedestrianState`, backend-independent `Observation` |
| `core/scenarios.py` | Ten seeded, randomised scenario kinds; split-disjoint seeds; batching |
| `core/geometry.py` | Distances to circles and segments (walls, polygons) |
| `core/rollout.py` | Recorded episodes; robot-absent counterfactual |
| `core/metrics.py` | Task, comfort/safety and counterfactual disruption metrics |
| `environments/lightweight/env.py` | Differential-drive robot (acceleration limits, exact arcs), collisions, success, timeout |
| `pedestrians/` | `replay`, `social_force`, `reactive_replay` (ORCA: Phase 2, see Status) |
| `policies/` | `social_navigation` (trajectory scoring, 10 parameters, hard safety constraint), `goal_seeking` baseline, parameter specs |
| `predictors/stars.py` | Multi-agent STARS forecaster as the policy's pedestrian predictor |
| `optimization/` | `evaluator` (batched candidates, objective), `cmaes` (pycma; checkpoint/resume) |
| `datasets/trajectory_loader.py` | CSV schema, STEPS converter, recorded-traffic scenarios split by recording |
| `visualization/plots.py` | Rollout PNG/GIF, CMA-ES convergence, policy comparison |

### Scenarios

| Kind | What happens (randomised) |
| --- | --- |
| `empty` | No pedestrians |
| `crossing` | One pedestrian crosses the robot's path, timed to meet it |
| `head_on` | One pedestrian walks toward the robot |
| `overtaking` | A slow pedestrian walks ahead in the same direction |
| `following` | A pedestrian at about the robot's speed walks ahead |
| `group_toward` | A group of 2–4 walks abreast toward the robot |
| `narrow_passage` | Corridor of width 1.4–2.2 m; a pedestrian comes the other way |
| `stationary_group` | A conversational group of 3–5 stands on the path |
| `dense_flow` | 10–20 pedestrians walking both ways along a corridor |
| `recorded` | A busy 30 s window of real STEPS traffic (recording-level splits) |

Train, validation and test scenarios use disjoint seed ranges (`SPLIT_OFFSETS`). Recorded scenes
draw from recordings assigned to the split by STEPS's grouped split manifest, or by a seeded hash
for CSV data.

### Pedestrian models

- **Replay**: recorded/scripted positions exactly, independent of the robot.
- **Social force**: Helbing–Molnar forces from other pedestrians, the robot and walls, with
  anisotropy and a sideways component. Bounded acceleration and speed; walks to the end of its
  reference.
- **Reactive replay**: tracks its reference trajectory (`tracking_gain`), avoids with the same
  forces, and returns smoothly to its path. An undisturbed walker reproduces its reference exactly
  (tested).

The robot is a non-reciprocal obstacle for every model, and reactive pedestrians cannot
interpenetrate it. The **counterfactual** run uses identical inputs with the robot removed, and the
disruption metrics compare the two.

### Navigation policy

At each step the controller rolls out a grid of constant `(v, w)` commands (6 × 11) over
`prediction_horizon` with the robot's acceleration limits. It predicts pedestrians (constant
velocity, or the STARS forecaster) and scores each command on:

- goal progress;
- predicted collisions;
- personal-space intrusion;
- disruption (blocking moving pedestrians' path ahead);
- obstacle clearance;
- speed, turning and command smoothness.

Commands that come within `safety_margin` of a pedestrian within `safety_horizon` seconds, or touch a
wall, are discarded (a hard constraint). If none remain, the robot stops (emergency stop). Tunable
parameters: `goal_weight`, `collision_weight`, `pedestrian_weight`, `disruption_weight`,
`obstacle_weight`, `preferred_speed`, `personal_space_radius`, `prediction_horizon`, `turn_penalty`,
`smoothness_weight`. `goal_seeking` is the same controller with every pedestrian term off.

### Metrics

| Group | Metrics |
| --- | --- |
| Task | `success`, `collision` (`ped_`/`obstacle_`), `timeout`, `time_to_goal`, `normalized_time`, `path_length` |
| Comfort and safety | `min_ped_clearance`, `intimate_violation_rate` (< 0.45 m), `personal_violation_rate` (< 1.2 m, SEAN 2.0 zones), `stops`, `emergency_stops`, `command_smoothness`, `jerk` |
| Disruption (vs. robot absent) | `disruption_displacement`, `disruption_max_displacement`, `velocity_deviation`, `ped_delay`, `peds_affected`, `reference_deviation` |

Disruption, comfort and goal achievement are reported separately. The CMA-ES objective (`objective:`
in the config) weights `success`, `collision`, a disruption metric, `normalized_time` and a comfort
metric; all are configurable.

### Trajectory data

The CSV schema is `recording_id,pedestrian_id,t,x,y[,vx,vy,theta]` (seconds, metres; optional
columns are derived when missing). `convert_steps(datasets/steps, out.csv)` exports STEPS. Any
dataset converts by writing those columns. Tracks are resampled to the simulation grid, and gaps
longer than 1 s split a track.

## Extending

**A new navigation policy.** Subclass `seanjax.policies.base.Policy`:
- declare `parameters` (`ParameterSpec(name, default, lower, upper, log)`);
- implement `act(obs: Observation, params: Tensor[B, n]) -> Tensor[B, 2]`;
- register it in `policies/goal_seeking.POLICIES`.

The batch dimension is environments, so policies must be batched. They see only `Observation`, so
they also run behind ROS 2 later.

**A new pedestrian model.** Subclass `seanjax.pedestrians.base.PedestrianModel`:
- implement `step(state, batch, k, robot, robot_radius, robot_present) -> PedestrianState`;
- use `batch.ped_ref` / `ped_valid` as the reference;
- honour `robot_present` so counterfactuals work;
- add it to `core/config.make_pedestrian_model`.

**A new predictor.** Subclass `policies.base.PedestrianPredictor` (`predict(obs, steps, dt)`) and
select it under `predictor:` in `policy.yaml`.

**A new scenario kind.** Add a branch to `core/scenarios.make_scenario` and the name to `KINDS`.

## SEAN 2.0 feature mapping

| SEAN 2.0 | SEAN-JAX |
| --- | --- |
| Unity + ROS 1 Noetic | Batched PyTorch simulator, no Unity/ROS dependency (preserved behaviour, replaced stack) |
| Social-force agents (`SFAgent.cs`) | `pedestrians/social_force.py`: same structure (goal relaxation, exponential agent/wall/robot repulsion, side step), batched |
| ORCA (stub in SEAN 2.0) | Planned (Phase 2); `pyrvo` builds as a reference |
| Trajectory playback (ETH/Zara CSV) | `replay` and `reactive_replay` from a documented CSV schema and STEPS |
| Social situations (cross_path, down_path, join/leave group) | Scenario kinds `crossing`, `head_on`/`following`/`overtaking`, `stationary_group`, `group_toward` (online situation classifier not yet ported) |
| Metrics (intimate/personal violations, collisions, path length, min distance) | Preserved with per-step distances; adds counterfactual disruption, stops, smoothness, time-to-goal |
| Robots (Jackal, Fetch, TurtleBot3, …), URDF | One differential-drive disc robot (Jackal-like limits); URDF robots come with Isaac (Phase 3) |
| Maps / occupancy from Unity scenes | Walls and polygons as segments; no occupancy-map import yet |
| `move_base` navigation baseline | `goal_seeking` and `social` controllers; ROS 2 / Nav2 node planned (Phase 4) |
| Cameras, LiDAR | Not supported in the lightweight backend (Phase 3) |

## Status

- **Done (Phases 1–2a).** The lightweight backend, three pedestrian models, two policies and the
  STARS predictor. Also CMA-ES with validation selection and resume, scenarios, recorded STEPS
  traffic, metrics, plots and tests.
- **Phase 2b, ORCA.** Not yet implemented.
- **Phase 3, Isaac Sim / Isaac Lab.** Not implemented. Officially, Isaac Sim supports Ubuntu
  22.04/24.04 and aarch64 only on DGX Spark. GB200 likely lacks the RT cores rendering needs. A
  headless physics test inside an Apptainer container on a compute node is the next step. See
  `docs/decisions.md`.
- **Phase 4, ROS 2.** Not implemented. RoboStack provides ROS 2 Jazzy for linux-aarch64 without
  sudo.
- **Performance limits.** Memory grows as environments × candidate commands × horizon steps ×
  pedestrians (`max_envs` bounds a batch). Pedestrian interactions are all pairs (fine up to ~100
  per environment). The STARS predictor costs one forecaster forward pass per control step
  (`predict_every` reduces that).
