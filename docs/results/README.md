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
| Scripted paths (ignore robot) | -0.65 / 60% / 20.4% | 0.39 / 92% / 4.8% | 0.71 / 98% / 0.4% | 0.69 / 99% / 0.7% |
| Social forces | 0.19 / 78% / 0.7% | 0.71 / 99% / 0.0% | 0.68 / 97% / 0.7% | 0.69 / 98% / 0.7% |
| Scripted paths + social forces | 0.25 / 80% / 0.7% | 0.71 / 100% / 0.4% | 0.68 / 97% / 0.4% | 0.68 / 98% / 0.0% |
| STARS, ignores robot | -0.49 / 61% / 13.0% | 0.43 / 93% / 3.3% | 0.73 / 99% / 0.4% | 0.70 / 99% / 0.7% |
| **STARS, sees robot (training crowd)** | -0.72 / 61% / 11.1% | 0.08 / 90% / 4.1% | 0.42 / 99% / 1.1% | 0.38 / 97% / 1.5% |

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
  pedestrians that ignore the robot (10% timeouts; 27% before the passage was widened to 2.0-2.8 m).

## Random vs CMA-ES-tuned weights, 10 paired seeds ([`seed_study/`](seed_study/))

Objective J = SPL - 2 C + SC - 0.3 time / 30 s (Principles metrics, Francis et al. Table I). CMA-ES seed s
starts from random seed s's weights (200 generations). Test split, STARS crowd, mean ± standard error over seeds:

| | Success | Collisions | SPL | Personal space (>= 0.5 m) | Time to goal |
| --- | --- | --- | --- | --- | --- |
| Random weights | 58% ± 9% | 9.0% ± 1.4% | 0.55 ± 0.09 | 0.88 ± 0.03 | 15.8 ± 1.4 s |
| CMA-ES from those weights | 97.7% ± 0.3% | 1.2% ± 0.2% | 0.89 ± 0.00 | 0.97 ± 0.00 | 11.2 ± 0.1 s |
| Paired change | +39 ± 9 points | -7.7 ± 1.3 points | +0.34 ± 0.09 | +0.09 ± 0.03 | -4.6 ± 1.5 s |

## Forecasting decoders vs official baselines (STEPS test, 12,500 samples, best-of-20, m)

All on the same `steps_default` test samples, frame and best-of-20 definition; one seed each,
trained from scratch on STEPS (MoFlow, SocialVAE: official recipes; ours: STEPS encoder fine-tuned
as `cvae_kl01_finetune`). *Joint ADE* scores whole scenes from one sample (the baselines predict one
target per sample, so it does not apply). ± is the standard error over test samples.

| Model | Target ADE | Target FDE | Joint ADE |
| --- | --- | --- | --- |
| Constant velocity | 0.279 | 0.527 | 0.279 |
| MoFlow (official, CVPR 2025) | 0.085 ± 0.0014 | 0.126 ± 0.0026 | — |
| SocialVAE (official) | 0.078 ± 0.0014 | 0.116 ± 0.0028 | — |
| CVAE (`cvae_kl01_finetune`) | 0.081 | 0.128 | 0.097 |
| Per-agent diffusion (50 DDIM steps) | 0.079 | 0.106 | 0.137 |
| Joint diffusion (50 DDIM steps) | 0.079 | 0.107 | 0.124 |

- Target accuracy is a near tie: SocialVAE 0.078, both diffusion decoders 0.079, CVAE 0.081;
  MoFlow 0.085 (its best validation came at epoch 13 of 150).
- The diffusion decoders have the best final positions (0.106-0.107 vs 0.116 SocialVAE, 0.126 MoFlow).
- Joint consistency: the CVAE is best (0.097); social layers inside the diffusion denoiser help
  (0.137 -> 0.124) but do not catch up. Diffusion needs ~50 DDIM steps (10 steps: ADE 0.268).
