# SimpleNavSim: social navigation simulation and policy tuning

SimpleNavSim is a lightweight, batched 2-D social navigation simulator in PyTorch: a
differential-drive robot navigates among pedestrians, and a parameterised navigation controller is
tuned with CMA-ES (or PPO) to reach its goal while disturbing pedestrians as little as possible.
Every environment is one row of a tensor batch, so a whole CMA-ES population × scenario set runs
as one simulation on a GPU. It is new code; [SEAN 2.0](https://github.com/yale-sean/social_sim_unity)
served only as a reference for behaviours and metric definitions.

Code: `src/simplenavsim/`. Configs: `simplenavsim/configs/`. Tools: `simplenavsim/scripts/`.
Tests: `tests/test_simplenavsim.py`.

## Usage

From the repository root after `uv sync` (`source .venv/bin/activate`); the STARS crowd and
predictor need a GPU (`sbatch scripts/horizon/gpu_jobs.slurm "<command>"` on TACC Horizon).

```bash
# Tune the controller in the STARS crowd (v4 recipe; --optimizer ppo for PPO on the same parameters)
python simplenavsim/scripts/train_cmaes.py --config simplenavsim/configs/cmaes_crowd_v4.yaml --device cuda \
  --policy checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml --out-dir checkpoints/simplenavsim/my_run

# Learned subgoals for the tuned controller (PPO, decision 99)
python simplenavsim/scripts/train_ppo.py --config simplenavsim/configs/cmaes_crowd_v4.yaml --device cuda \
  --policy checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml --out-dir checkpoints/simplenavsim/subgoal

# Evaluate policies on the test split under every pedestrian condition, with GIF grids
python simplenavsim/scripts/evaluate.py --help
```

Results: [`docs/results/README.md`](../docs/results/README.md); reasoning: `docs/decisions.md`.

## Components

| Module | Contents |
| --- | --- |
| `core/scenarios.py` | Nine seeded, randomised scenario kinds plus recorded STEPS windows; split-disjoint seeds; batching |
| `core/rollout.py`, `core/metrics.py` | Recorded episodes, the robot-absent counterfactual, task / comfort / disruption metrics |
| `environments/lightweight/env.py` | Differential-drive robot (acceleration limits, exact arcs), collisions, success, timeout |
| `pedestrians/` | `replay`, `social_force`, `reactive_replay`, `stars_crowd` |
| `policies/` | `social_navigation` (arc scoring with a hard safety constraint), `goal_seeking`, `ppo` (learned subgoals) |
| `predictors/stars.py` | The multi-agent STARS forecaster as the controller's pedestrian predictor |
| `optimization/` | `evaluator` (batched candidates, objective), `cmaes` (pycma, resume), `ppo_search` |
| `datasets/trajectory_loader.py` | Recorded STEPS traffic as scenarios, split by recording |
| `visualization/plots.py` | GIF grids, CMA-ES convergence |

**Scenarios.** `empty`, `crossing`, `head_on`, `overtaking`, `following`, `group_toward`,
`narrow_passage`, `stationary_group`, `dense_flow` (straight-line scripted walkers, randomised per
seed) and `recorded` (a busy 30 s STEPS window). Train, validation and test use disjoint seeds.

**Pedestrian models.** Every pedestrian has a scripted or recorded reference trajectory:
- `replay` follows it exactly; `social_force` walks to its end with Helbing–Molnar forces;
  `reactive_replay` tracks it while avoiding others with the same forces.
- `stars_crowd`: after a 3.2 s warm-up on the reference, the STARS forecaster moves everyone
  (re-forecast every 0.4 s, seeded AR(1) CVAE intents); with `include_robot` the robot is one more
  node in its graph.

The counterfactual run has identical inputs with the robot removed; disruption metrics compare the two.

**Controller.** Every 0.1 s it rolls out a grid of constant `(v, w)` arcs (6 × 11 plus reverse
in the v4 config), predicts pedestrians (constant velocity or STARS), discards arcs that come
within `safety_margin` of a pedestrian within `safety_horizon` or touch a wall, and picks the
lowest weighted cost (goal progress, collision, personal space, disruption, clearance, speed,
smoothness, waiting terms). CMA-ES tunes the weights (`cmaes.optimize`).

**Objective** (`objective:` in the config): success, collisions, a disruption metric
(`robot_influence` in the STARS crowd), time, a comfort metric, detour, smoothness, loops and
time to goal. Evaluation in the STARS crowd is not bit-for-bit reproducible (bf16 on GPU);
objective differences below about 0.01–0.02 are noise.

## Status

Reference policy: `checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml` (v4). Neither
re-tuning with CMA-ES nor PPO (on the same parameters, or as learned subgoals) improves on it
(decision 99). Isaac Sim / Isaac Lab and ROS 2 backends are not implemented.
