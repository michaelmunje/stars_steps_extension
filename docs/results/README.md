# Results

Multi-agent STARS forecaster and SimpleNavSim navigation. Reasoning: the last section of
[`../decisions.md`](../decisions.md). Trained weights: GitHub release
[`weights-v1`](https://github.com/michaelmunje/stars_steps_extension/releases/tag/weights-v1).
How to run the simulator: [`../../simplenavsim/README.md`](../../simplenavsim/README.md).

## Forecasting (test split, best-of-20 agent ADE / FDE in metres)

| Model | STEPS | PTDPES | Madras | Grand Central | UCY | ETH |
| --- | --- | --- | --- | --- | --- | --- |
| Constant velocity | 0.276 / 0.518 | 0.963 / 1.800 | 0.627 / 1.180 | 0.703 / 1.427 | 0.397 / 0.891 | 0.727 / 1.428 |
| `cvae_kl01_finetune` (STEPS only) | **0.080 / 0.126** | | | | | |
| `cvae_all_lr3e-4_seed42` (six datasets) | 0.079 / 0.124 | 0.282 / 0.408 | 0.209 / 0.315 | 0.243 / 0.403 | 0.203 / 0.406 | 0.235 / 0.377 |

- The CVAE's KL weight is the key setting: at 1.0 the latent collapses (agent ADE 0.110 on
  STEPS); at 0.1 samples are diverse and error falls by about 30%.
- The pooled model is as good as or better than a per-dataset model on every dataset once its
  learning rate is 3e-4.
- `cvae_kl01_finetune` drives the simulated crowd and is the robot's predictor: its 20 s
  rollouts keep the real walking speed, while the slightly more accurate search winner was 25% too slow.

## Navigation (SimpleNavSim, 270 test scenes: 9 kinds x 30)

GIFs and full tables: [`evaluation/`](evaluation/). The objective has a disruption term only in
the STARS crowd that sees the robot, so compare scores within a row, not across rows.

Objective / success / collisions:

| Pedestrians | Random weights | Default weights | **CMA-ES (v4)** | PPO subgoals |
| --- | --- | --- | --- | --- |
| Scripted paths (ignore robot) | -0.72 / 57% / 21.5% | 0.28 / 89% / 7.4% | 0.66 / 96% / 0.4% | 0.62 / 96% / 0.7% |
| Social forces | 0.17 / 77% / 0.7% | 0.71 / 99% / 0.0% | 0.63 / 95% / 0.7% | 0.62 / 96% / 0.7% |
| Scripted paths + social forces | 0.23 / 80% / 0.7% | 0.70 / 100% / 0.4% | 0.63 / 95% / 0.4% | 0.65 / 97% / 0.0% |
| STARS, ignores robot | -0.52 / 59% / 13.0% | 0.40 / 92% / 4.1% | 0.73 / 99% / 0.0% | 0.68 / 98% / 1.1% |
| **STARS, sees robot (training crowd)** | -0.72 / 61% / 11.5% | 0.08 / 90% / 3.7% | 0.43 / 99% / 0.7% | 0.36 / 97% / 1.9% |

All four use the same trajectory-rollout planner; only the cost-function weights (or, for PPO
subgoals, where it aims) differ. Random weights are drawn per scene from the CMA-ES search range.
Not shown: goal-seeking (-2.07 / 39% / 61.5% in the training crowd) and PPO tuning the same
weights as CMA-ES (0.42 / 99% / 0.7%).

- v4 is the reference policy. Learned subgoals on top of it are slightly worse, PPO on the same
  weights ties with it, and random weights fail often (61% success in the training crowd). Differences under 0.02 are evaluation noise.
- v4's tuning is specific to the STARS crowd: with social-force pedestrians the untuned
  controller scores higher.
- The STARS crowd that sees the robot follows it about twice as often as one that ignores it.
- Weak spots: head-on encounters in the STARS crowd (6.7% collisions) and narrow passages with
  pedestrians that ignore the robot (27% timeouts).
