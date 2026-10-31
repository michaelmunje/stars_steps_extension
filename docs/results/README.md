# Results overview

Living summary of the multi-agent STARS forecaster and the SimpleNavSim navigation experiments. The
numbers here are copied from the files linked below. The reasoning behind each choice is in
[`../decisions.md`](../decisions.md); entry numbers are given in brackets.

**Latest navigation evaluation, with animations:** [`evaluation/`](evaluation/) (all policies x pedestrian conditions) [100].
**Weights** (forecasters and policies): GitHub release [`weights-v1`](https://github.com/michaelmunje/stars_steps_extension/releases/tag/weights-v1).
**Latest forecaster, with per-dataset results:** [`../latest_forecaster/`](../latest_forecaster/README.md).

## Current references

| Use | Artifact | Headline (test split) |
| --- | --- | --- |
| STEPS forecaster, and the SimpleNavSim crowd and predictor | `experiments/stars_ma/runs/cvae_kl01_finetune/checkpoints/best.pt` | ADE / FDE 0.080 / 0.126 (constant velocity 0.276 / 0.518); realistic 20 s rollouts [48] |
| All-dataset forecaster | `experiments/stars_ma/runs/cvae_all_lr3e-4_seed42/checkpoints/best.pt` | 2-3.5x better than constant velocity on all six datasets. Better than or level with a per-dataset model on every dataset, robust across seeds (pooled LR 3e-4) [86] |
| Navigation policy (v4) | `checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml` | In the STARS crowd: 98.5-98.9% success, 0.7-1.1% collisions [99, 100] |

Closed questions: pooled training beats per-dataset training once its LR is lowered to 3e-4 [86];
hyperparameter search has converged [61, 62]; closed-loop training does not
help [47, 65, 73]; controller cost terms cannot remove the remaining dense-flow loops [79].

## Trajectory forecasting (STEPS test split, 12,500 windows)

Scores are best-of-20 for the CVAE rows, in metres. *Moving* means agents whose true future mean
speed is above 0.2 m/s; about 90% of STEPS pedestrians are slower. The full table is
[`stars_ma_steps_test.md`](stars_ma_steps_test.md).

| Model | Agent ADE / FDE | Moving ADE / FDE | Joint ADE |
| --- | --- | --- | --- |
| Constant velocity | 0.276 / 0.518 | 0.894 / 1.771 | 0.279 |
| Deterministic decoder, fine-tuned phase-1 encoder | 0.115 / 0.205 | 0.559 / 1.107 | 0.119 |
| CVAE, KL 1.0, fine-tuned (collapsed latent, [32]) | 0.110 / 0.195 | 0.532 / 1.049 | — |
| **CVAE, KL 0.1, fine-tuned (`cvae_kl01_finetune`)** | **0.080 / 0.126** | **0.336 / 0.588** | **0.097** |
| CVAE, KL 0.1, from scratch with search settings | 0.081 / 0.129 | 0.342 / 0.600 | 0.097 |
| CVAE, best Optuna trial 53 at full length (2 seeds) | 0.079 / 0.125 | 0.333 / 0.587 | 0.097 |

- **KL weight is the key setting** [32–35, 41]. At 1.0 the latent collapsed: samples ended about
  9 mm apart. At about 0.1 the samples are diverse (0.16 m apart) and error falls 30–36%. Optuna
  attributes 51–58% of the variance in results to it; batch size accounts for 2–4%.
- **Seed noise** is about ±0.0007 m of validation ADE [44].
- **Pretraining helps a little** once the KL weight is right: fine-tuned beats from scratch,
  which beats frozen. Pretraining on all six datasets gives no extra gain on STEPS [46].
- **Most accurate vs most realistic** [48]: trial 53 is marginally more accurate, but its simulations are 25% too slow, so `cvae_kl01_finetune` drives the crowd and the planner.
- **Closed-loop (multi-step) training does not help** [47]. Rollouts are stable without it.
- **Simulation realism:** the best model's 20 s rollouts move at 0.115 m/s, matching smoothed real
  tracks. They are smoother than reality: acceleration is 0.03 vs 0.12 m/s² [38–39].
- **Figures:**
  - [`forecasts_cvae_frozen.png`](forecasts_cvae_frozen.png): KL 1.0, nearly identical samples.
  - [`forecasts_cvae_hpo0_scratch_full.png`](forecasts_cvae_hpo0_scratch_full.png): KL ≈ 0.1,
    diverse samples.
- **Searches:** [`hpo_steps_cvae_scratch.md`](hpo_steps_cvae_scratch.md) and
  [`hpo_steps_cvae_ssl.md`](hpo_steps_cvae_ssl.md).

### All six datasets (test split) [59, 86]

Reference pooled model `cvae_all_lr3e-4_seed42` (trial-53 architecture, pooled LR 3e-4), agent
ADE / FDE, best-of-20, against constant velocity. The earlier pooled model and its rollout
analysis are in [`stars_ma_all_test.md`](stars_ma_all_test.md).

| Dataset | Constant velocity | STARS (`cvae_all_lr3e-4_seed42`) |
| --- | --- | --- |
| STEPS | 0.276 / 0.518 | 0.079 / 0.124 |
| PTDPES | 0.963 / 1.800 | 0.282 / 0.408 |
| Madras | 0.627 / 1.180 | 0.209 / 0.315 |
| Grand Central | 0.703 / 1.427 | 0.243 / 0.403 |
| UCY | 0.397 / 0.891 | 0.203 / 0.406 |
| ETH | 0.727 / 1.428 | 0.235 / 0.377 |

![Per-dataset test error](stars_ma_all_test.png)

- SSL pretraining plus fine-tuning is 0.002–0.010 m worse on every dataset.
- **One model per dataset vs one for all** [86]: with the pooled LR lowered to 3e-4, the pooled
  model is better than or level with each dataset's own model on every dataset, across four runs:
  STEPS within ±1%, PTDPES, Madras and Grand Central 1-3% better, UCY 22-24% and ETH 40-42%
  better. (At the STEPS-tuned LR of 9.6e-4 pooling was seed-sensitive and on average worse on the
  large datasets [84, 85].)

![One model per dataset vs one for all](stars_ma_single_vs_all.png)
- Feeding back its own predictions every step costs little on STEPS but 14–20% elsewhere.

## Navigation (SimpleNavSim, 270 test scenes: 9 kinds x 30) [94-100]

How to run it: [`../../simplenavsim/README.md`](../../simplenavsim/README.md). Full tables and
GIFs: [`evaluation/`](evaluation/). Objective J as in that README; it has a disruption term only
in the STARS crowd that sees the robot, so compare scores within a row, not across rows.

Objective / success / collisions:

| Pedestrians | Goal-seeking | Untuned controller | **v4 (CMA-ES)** | PPO, same weights | PPO subgoals |
| --- | --- | --- | --- | --- | --- |
| Scripted paths (ignore robot) | -1.89 / 37% / 63% | 0.28 / 89% / 7.4% | 0.66 / 96% / 0.4% | 0.63 / 96% / 1.1% | 0.62 / 96% / 0.7% |
| Social forces | -0.63 / 67% / 33% | 0.71 / 99% / 0% | 0.63 / 95% / 0.7% | 0.64 / 96% / 0.4% | 0.62 / 96% / 0.7% |
| Scripted paths + social forces | -0.49 / 70% / 30% | 0.70 / 100% / 0.4% | 0.63 / 95% / 0.4% | 0.64 / 96% / 0.4% | 0.65 / 97% / 0% |
| STARS, ignores robot | -1.86 / 38% / 62% | 0.39 / 92% / 4.4% | 0.72 / 99% / 0.4% | 0.73 / 99% / 0.4% | 0.68 / 98% / 1.1% |
| **STARS, sees robot (training crowd)** | -2.07 / 38% / 62% | 0.08 / 90% / 4.1% | 0.42 / 99% / 1.1% | 0.42 / 99% / 0.7% | 0.37 / 97% / 1.9% |

- **v4 is the reference.** Re-tuning with CMA-ES or PPO lands on v4 or ties with it; learned
  subgoals are slightly worse [99, 100]. Differences under 0.02 are evaluation noise.
- **v4's tuning is specific to the STARS crowd:** with social-force pedestrians the untuned
  controller scores higher [100].
- **The STARS crowd that sees the robot tends to walk with it** (twice as much following as when
  it ignores the robot): it treats the robot as a companion, as walking groups do in STEPS.
  Not a frame bug (forecasts are rotation-, translation- and order-equivariant to 1e-5 m).
- **Weak spots:** head-on encounters in the STARS crowd (6.7% collisions) and narrow passages
  with pedestrians that ignore the robot (27% timeouts) [100].
- **Did not beat v4** [68-98]: extra controller cost terms and memory, wait-then-go plans,
  tuning in a pooled-data crowd, a behaviour-cloned steering network, per-step parameter
  adjustment, and PPO choosing arcs directly (4-5% collisions).
