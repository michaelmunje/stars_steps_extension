# SimpleNavSim: social navigation simulation and policy tuning

A batched 2-D social-navigation simulator in PyTorch. A differential-drive robot drives to a goal
among pedestrians; a navigation controller with 11 tunable weights is tuned with CMA-ES (or PPO).
Pedestrians can be scripted, driven by social forces, or driven by the multi-agent STARS
trajectory forecaster. Every environment is one row of a tensor batch, so whole populations of
candidate policies run as one simulation. New code; [SEAN 2.0](https://github.com/yale-sean/social_sim_unity)
was only a reference for behaviours and metrics.

## Quick start

```bash
uv sync && source .venv/bin/activate          # from the repository root (Python 3.12)

# Trained weights: two STARS forecasters and the tuned policies, unpacked in place
gh release download weights-v1 --repo michaelmunje/stars_steps_extension
tar xzf simplenavsim_weights.tar.gz

# Evaluate the tuned policies under every pedestrian model (needs a GPU for the STARS crowd)
python simplenavsim/scripts/evaluate.py
python simplenavsim/scripts/evaluate.py --render
```

The bundle contains:

| File | What it is |
| --- | --- |
| `experiments/stars_ma/runs/cvae_kl01_finetune/checkpoints/best.pt` | STARS forecaster trained on STEPS: moves the simulated crowd and is the robot's pedestrian predictor |
| `experiments/stars_ma/runs/cvae_all_lr3e-4_seed42/checkpoints/best.pt` | STARS forecaster trained on six datasets (not used by the default configs) |
| `checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml` | **v4**: the CMA-ES-tuned controller, the reference policy |
| `checkpoints/simplenavsim/ppo_params_s42/best_policy.yaml` | PPO tuning the same 11 weights (a check on CMA-ES; not in the figures) |
| `checkpoints/simplenavsim/subgoal_s1/` | PPO network choosing subgoals for v4 |

**STEPS data is optional.** With the processed dataset at `datasets/steps`, scenarios include
recorded STEPS crowds and the evaluation measures crowd realism; without it both are skipped.
On TACC Horizon, wrap any command in `sbatch scripts/horizon/gpu_jobs.slurm "<command>"`
(see [`docs/horizon.md`](../docs/horizon.md)). `--device cpu` works for small runs.

## Evaluate

```bash
python simplenavsim/scripts/evaluate.py                                    # all conditions, 30 test scenes per kind
python simplenavsim/scripts/evaluate.py --conditions stars_reactive --per-kind 5
python simplenavsim/scripts/evaluate.py --policy mine=checkpoints/simplenavsim/my_run/best_policy.yaml
python simplenavsim/scripts/evaluate.py --render                           # tables and GIFs from the saved runs
```

Each run rolls out every policy in every pedestrian condition on the test split, saves
`runs/simplenavsim/evaluation/condition_<name>.pkl`, and `--render` writes to `docs/results/evaluation/`:

- `original.md`: the policies in the crowd they were trained in (`stars_reactive`), scored by the training objective;
- `conditions.md`: the same metrics under every condition, per scene kind, and crowd realism;
- `results.json`: every metric, per condition, policy and scene kind;
- GIF grids (with a PNG of the last frame) of the first test scene of each kind.

| Condition | Pedestrians |
| --- | --- |
| `replay` | walk their scripted paths and ignore the robot |
| `social_force` | walk to their scripted goal with social forces (avoid each other and the robot) |
| `reactive_replay` | track their scripted paths, pushed off them by social forces |
| `stars_blind` | moved by the STARS forecaster, which does not see the robot |
| `stars_reactive` | moved by the STARS forecaster, with the robot as one more person in its graph (the training crowd) |

STARS-driven pedestrians follow their scripted path for 3.2 s (the forecaster's history), then
the forecaster takes over, re-forecasting every 0.4 s with seeded, slowly varying CVAE intents.

## Train

```bash
# CMA-ES over the controller's 11 weights in the STARS crowd (the v4 recipe)
python simplenavsim/scripts/train_cmaes.py --config simplenavsim/configs/cmaes_crowd_v4.yaml \
    --policy checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml --device cuda \
    --out-dir checkpoints/simplenavsim/my_run
# The same search with PPO instead of CMA-ES
python simplenavsim/scripts/train_cmaes.py ... --optimizer ppo
# A PPO network that picks subgoals for a tuned controller
python simplenavsim/scripts/train_ppo.py --config simplenavsim/configs/cmaes_crowd_v4.yaml \
    --policy checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml --device cuda \
    --out-dir checkpoints/simplenavsim/my_subgoals
```

`--policy` sets the starting point (omit it to start from the defaults). Training writes
`best_policy.yaml` (selected on the validation split), `generations.jsonl` and
`convergence.png`; `--resume` continues an interrupted CMA-ES run. Population, generations,
scene counts and objective weights are in the config. Train, validation and test scenes never
overlap (disjoint seeds; recorded scenes split by recording).

| | CMA-ES | PPO on the weights | PPO subgoals |
| --- | --- | --- | --- |
| Learns | the 11 weights, fixed per episode | the same 11 weights | a network: observation -> subgoal offset (up to 3 m) every 1 s; weights fixed at v4 |
| Reward | episode objective J, averaged over training scenes | the same J per sampled weight set | J at episode end plus progress shaping that sums to a constant |

**Objective** (`objective:` in the config):
J = success - 2 collision - 1 robot_influence - 0.3 normalised time - 1 intimate-zone rate
- 1 (1 - path efficiency) - 0.1 command jerkiness - 0.2 loops.
`robot_influence` is how much the robot changes the STARS forecasts of the pedestrians (it
exists only in `stars_reactive`). Scores in the STARS crowd vary by about 0.01-0.02 between
identical runs (bf16 on GPU).

## The controller

Every 0.1 s the robot scores a grid of constant-speed, constant-turn arcs (6 speeds x 11 turn
rates plus reversing) over the next ~2 s against forecasts of the pedestrians (STARS or constant
velocity), removes arcs that come within 0.05 m of a pedestrian within 1 s or touch a wall, and
drives the cheapest remaining arc. If none remains it stops.

| Weight | Role in an arc's cost | Default | v4 |
| --- | --- | --- | --- |
| `goal_weight` | reward for getting closer to the goal | 2.0 | 0.5 (lower bound) |
| `alignment_weight` | extra reward for ending the arc facing the goal | 0.1 | 0.075 |
| `pedestrian_weight` | penalty for intruding into personal space, `(1 - gap / radius)^2` | 1.0 | 27.4 |
| `personal_space_radius` | gap (m, edge to edge) where that penalty starts | 0.8 | 0.55 |
| `disruption_weight` | penalty for being in the space ahead of a moving pedestrian | 1.0 | 0.01 (lower bound) |
| `prediction_horizon` | how far ahead (s) arcs are scored | 2.0 | 2.27 |
| `preferred_speed` | cruise speed (m/s) | 0.8 | 1.0 (upper bound) |
| `speed_scale` | pull towards that speed; off when a pedestrian is within 2x personal space | 1.0 | 5.0 |
| `retreat_scale` | penalty for arcs ending farther from the goal (wait rather than back off) | 1.0 | 1.59 |
| `smoothness_weight` | penalty for changing the command | 0.2 | 0.39 |
| `turn_penalty` | penalty for turning | 0.1 | 0.0 (lower bound) |

## Code

| Path | Contents |
| --- | --- |
| `src/simplenavsim/core/` | scenarios (`crossing`, `head_on`, `overtaking`, `following`, `group_toward`, `narrow_passage`, `stationary_group`, `dense_flow`, `recorded`), rollout, metrics, config |
| `src/simplenavsim/environments/lightweight/env.py` | robot dynamics, collisions, success (within 1 m of the goal), timeout (30 s) |
| `src/simplenavsim/pedestrians/` | `replay`, `social_force`, `reactive_replay`, `stars_crowd` |
| `src/simplenavsim/policies/` | `social_navigation` (the controller), `goal_seeking` (baseline), `ppo` (subgoals) |
| `src/simplenavsim/predictors/stars.py` | STARS as the controller's pedestrian predictor |
| `src/simplenavsim/optimization/` | objective and batched evaluator, CMA-ES, PPO search |
| `simplenavsim/configs/` | `cmaes_crowd_v4.yaml` (everything above) and the files it includes |
| `simplenavsim/scripts/` | `train_cmaes.py`, `train_ppo.py`, `evaluate.py` |

Results: [`docs/results/README.md`](../docs/results/README.md). Reasoning: the last section of
[`docs/decisions.md`](../docs/decisions.md).
