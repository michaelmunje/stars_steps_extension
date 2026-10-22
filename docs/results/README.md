# Results overview

Living summary of the multi-agent STARS forecaster and the SEAN-JAX navigation experiments. The
numbers here are copied from the files linked below. The reasoning behind each choice is in
[`../decisions.md`](../decisions.md); entry numbers are given in brackets.

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

## Navigation (SEAN-JAX lightweight backend, 80 scripted test scenarios)

Objective v1 = success − 2·collision − disruption − 0.3·time − 0.5·personal-zone rate.

**Same parameters, different pedestrian predictor** [37]:

| Policy | Constant velocity | STARS forecaster |
| --- | --- | --- |
| Goal-seeking baseline | −0.572 (30% collisions) | — |
| Social controller, default | 0.439 (2.5% collisions) | 0.536 (0 collisions) |
| CMA-ES tuned v1 (`checkpoints/seanjax/cmaes/best_policy.yaml`) | 0.612 | **0.647** |

**Robustness to the pedestrian model:** the v1 policy, tuned against reactive replay, has 0
collisions and beats the default under ORCA, social force and replay pedestrians [29].

**Controller fixes**, prompted by the rollout figures:
- [43] Tuned policies orbited instead of waiting. A retreat penalty, a speed cost that fades near
  pedestrians and slow reversing fixed this. Results above this point used the old controller.
- [45] A warm-started policy scored 0.707 by circling pedestrians and driving around walls. This is
  reward hacking. Objective v2 adds detour (path efficiency) and smoothness terms; reruns are in
  progress.

**Figures:**
- [`navigation_compare_stars.png`](navigation_compare_stars.png): goal-seeking vs default vs tuned,
  five scenarios.
- [`navigation_compare_warm.png`](navigation_compare_warm.png): the reward-hacking policy.
- Per-run GIFs are under `runs/seanjax/*/visualize/` (not in git).

## Forecaster as the crowd

`stars_crowd` moves the simulated pedestrians with the forecaster itself, with the robot as one
more person in the graph [46]. Tuning the navigation policy inside that crowd (`cmaes_crowd.yaml`)
is the current experiment.
