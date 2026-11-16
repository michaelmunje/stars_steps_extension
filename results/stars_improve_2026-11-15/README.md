# STARS improvement: collisions and speed (2026-11-15/16)

Branch `stars-improve` (from `goal-conditioned`). Main evaluation: **open-loop** forecasting, 8 observed / 12
predicted steps at 0.4 s, every agent of a scene window with a fully observed future.

**Tool:** `python -m baselines.stars_ma_openloop` (`src/baselines/stars_ma_openloop.py`) reports:
- **accuracy:** single-sample (sample 0 of K), mean-of-K, prior-mean ("mode"), min-of-20 and joint min-of-20 ADE/FDE;
- **collisions:** close-pair rates, i.e. the fraction of agent pairs closer than 0.2 / 0.3 / 0.4 m at some
  future step. They are reported per sample (averaged over K), on the sample mean, on a *shuffled* joint
  sample (each agent takes its own sample index), and excluding observed groups (within 1 m and co-moving
  at the last observed step);
- **speed:** path speed and net speed (|x12 - x0| / 4.8 s), the predicted/actual ratio for moving agents,
  speed W1 and turning;
- **ground-truth and window statistics.**

`--oversample M` and the decoder setting `oversample` choose collision-light joint samples (see Fixes).

## Diagnosis (validation split; `diagnosis/*.json`)

### Does the training data reflect the collisions? Mostly no.
- **Ground-truth close pairs are rare:** 0.03% of pairs at 0.2 m and 0.15% at 0.4 m, pooled. By dataset at
  0.4 m: MADRAS 1.9%, UCY 0.6%, STEPS 0.45%, PTDPES 0.12%, Grand Central 0.18%, ATC 0.01%.
- **Most of them are groups:** 50-88% of close pairs are walking together (within 1 m and co-moving at
  the last observed step).
- **No duplicate tracks** (< 0.05 m) of note.
- **Preprocessing doesn't create overlaps.** Canonical data is 2.5 Hz with linear resampling only for
  PTDPES (2 Hz). Nodes are agents visible at the last observed step with at least 2 observations. 12% of
  nodes are partially observed (gaps filled linearly) and 20% lack a full future; those are masked in the
  loss and not scored.
- **Constant velocity already gives 2-6x the ground-truth rate.**
- **Conclusion:** the data has some close pairs (groups, MADRAS, UCY), but far fewer than the model predicts.

### Causes of the collisions
- **Old pooled model** (`cvae_all_lr3e-4_seed42`): 0.14% of pairs at 0.2 m and 0.42% at 0.4 m per sample.
  That is 2-5x ground truth (UCY 2.1% vs 0.6%) and about 70% of constant velocity's rate.
- **Samples are not coordinated across agents.** Shuffling sample indices between agents gives the same
  rate (0.46% vs 0.42%). Each agent's latent is drawn independently. The single social layer mixes latents
  over the *observed* history only. The GRU decodes each agent alone, never sees the neighbours' predicted
  futures, and no loss penalises overlap.
- **Not a padding or masking bug.** The sample mean and the mode collide less than single samples.
- **Diffusion decoders (STEPS) collide more:** 0.86% for per-agent and 0.75% for joint diffusion vs 0.39%
  for the CVAE at 0.4 m (ground truth 0.45%). They reproduce tracker jitter (turning 1.05 rad/step) and
  have worse single-sample ADE. Joint denoising helps only a little.

### Causes of the "slow" speeds
- **Open-loop speeds are mostly fine.** Net speed is 0.42 vs ground truth 0.43 m/s. For moving agents the
  predicted/actual ratio is 0.93-0.96 per sample.
- **Regression to the mean is mild but real.** The mode is slower (ratio 0.93). Sampling temperature 0.5
  cuts single-sample ADE (0.61 to 0.48) and collisions (0.42% to 0.27% at 0.4 m), but makes agents slower
  (path speed 0.44 to 0.40) and worsens minADE (0.25 to 0.31).
- **Part of the slowness is a measurement artefact.** ATC and STEPS ground truth is jittery: ATC path speed
  is 0.49 m/s but net speed only 0.29. Smooth predictions then look slow by path speed while matching net
  speed (model 0.27-0.32).
- **STEPS is 74% standing agents,** which pulls pooled averages down.
- **The closed-loop slowness** in the ETH/UCY benchmark is compounding over rollouts, not an open-loop bias.

## Fixes tried (validation, 15-epoch screening on all 7 datasets; `val/*.json`, `val/table.txt`)

Columns: ADE values are the macro mean over 6 validation datasets; net speed and close pairs (averaged over
K = 20) are pooled over all agents. Ground truth: net speed 0.441 m/s; close pairs 0.03% at 0.2 m and 0.15%
at 0.4 m.

| # | hypothesis / change | single-sample ADE | minADE | joint ADE | net speed | close pairs < 0.2 m | < 0.4 m | result |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (pooled recipe, all 7 datasets) | 0.545 | 0.232 | 0.437 | 0.415 | 0.19% | 0.55% | reference |
| 1 | refinement round: attend to neighbours' *predicted* futures, re-decode (`--refine 1`) | 0.549 | 0.231 | 0.439 | 0.414 | 0.20% | 0.55% | no change alone (no signal to avoid) |
| 2 | collision hinge on a prior sample, prior trainable (`--collision-weight 10 --no-collision-fixed-prior`) | 1.366 | 0.411 | 0.767 | 0.461 | 0.08% | 0.28% | **fails**: widens the prior (sample spread 2x) |
| 3 | 1 + 2 | 0.947 | 0.435 | 0.631 | 0.308 | 0.02% | 0.10% | ground-truth-level collisions, but slow and inaccurate |
| 4 | hinge, decoder only (gradient-free latent, now the default) | 0.771 | 0.289 | 0.602 | 0.405 | 0.14% | 0.44% | weak, ADE cost |
| 5 | refine + hinge 10, decoder only | 0.683 | 0.270 | 0.533 | 0.401 | 0.03% | 0.14% | ground-truth-level, coordinated (shuffled 0.31%), +25% ADE |
| 6 | 5 with hinge weight 3 | 0.619 | 0.251 | 0.491 | 0.409 | 0.06% | 0.26% | dominated by 9 |
| 7 | 5 with hinge weight 1 | 0.578 | 0.240 | 0.462 | 0.411 | 0.13% | 0.41% | dominated by 9 |
| 8 | 5 with KL 0.03 | 0.900 | 0.418 | 0.698 | 0.326 | 0.03% | 0.12% | worse |
| 9 | 5 with **KL 0.15** | **0.564** | 0.274 | 0.454 | 0.392 | **0.02%** | **0.11%** | best collision-aware setting: ADE +3.4%, minADE +18% |
| 10 | 5 with latent 32 | 0.777 | 0.269 | 0.592 | 0.410 | 0.02% | 0.13% | worse |
| 11 | 5 with LR 6e-4 | 0.793 | 0.282 | 0.600 | 0.408 | 0.02% | 0.13% | worse |
| 12 | baseline + **choose samples at inference**: draw 2x K joint samples, keep the K with fewest non-group close pairs (no future information) | 0.540 | 0.232 | 0.436 | 0.413 | 0.14% | 0.41% | free: -25% collisions |
| 13 | same, 4x K | **0.540** | **0.232** | **0.435** | 0.413 | 0.12% | 0.35% | **chosen**: -37% collisions, accuracy unchanged |
| - | sampling temperature 0.5 (old pooled model) | 0.485 | 0.307 | 0.365 | (path 0.40) | 0.07% | 0.27% | trades minADE and speed for single-sample ADE |
| - | training on all 7 datasets (adds ATC) vs 6 | (test below) | | | | | | ATC ADE 0.84 to 0.67 m, other datasets equal |

Also stopped or not run:
- **Stopped early (low value):** best-of-many prior loss with hinge 10 (about 10 min/epoch); hinge weight
  30; two refinement rounds.
- **Not run:** goal leak removal (`--goal-beyond-horizon` is implemented, untested). Oracle goals are
  privileged inputs that the open-loop baselines don't get.
- **Rotation augmentation is a no-op:** node and edge features are in agents' local SE(2) frames, so the
  model is rotation invariant.
- **Flips were not tried;** they would remove the passing-side convention.

**Tuning budget:** about 13 screening runs of 15 epochs (lr, KL, latent, collision weight, refinement
depth), selected on validation only. The baseline recipe's earlier Optuna tuning (`cvae_hpo53`) was kept.
Total compute was roughly 30-40 GPU-hours, including 4 x 60-epoch final runs.

## Final models

- **`stars_improved`:** baseline pooled recipe on all 7 datasets, plus 4x joint-sample oversampling with
  collision-based selection at inference.
  - Recipe: scratch encoder, CVAE latent 64, decoder 512, KL 0.0799, LR 3e-4 cosine, 60 epochs, kNN 24 / 10 m,
    macro best-of-20 selection.
  - Three seeds: `si_final_base_s{42,1,2}`, best epochs 39, 42 and 42.
  - Benchmark checkpoint: `si_final_base_s2/checkpoints/best_os4.pt`. Seed 2 has the best validation
    selection score (0.2218, against 0.2231 and 0.2240). This copy of `best.pt` sets decoder
    `oversample: 4`, so `predict()` itself selects the samples.
- **`stars_improved_social` (secondary, collision-aware):** row 9's setting (`--refine 1 --collision-weight 10
  --kl-weight 0.15`), 60 epochs, seed 42: `si_final_social_s42/checkpoints/best.pt`.

Checkpoints are in `/scratch/09865/mmunje/stars-forecasting/experiments/stars_ma/runs/si_*` (gitignored).

## Test results (test split, unique windows; `test/*.json`)

ADE/FDE values are macro means over STEPS, PTDPES, MADRAS, Grand Central, UCY, ETH and ATC. Close-pair
rates (%) and net speed are pooled over all agents and averaged over K = 20. Speed W1 is the Wasserstein-1
distance between predicted and true per-step speed distributions, averaged over datasets. "Excl. groups"
leaves out pairs that are a group in the observation.

| model | single-sample ADE / FDE | minADE / minFDE | joint ADE / FDE | close pairs < 0.2 / 0.4 m | excl. groups < 0.2 / 0.4 m | net speed (m/s) | speed W1 |
|---|---|---|---|---|---|---|---|
| `pooled6` (old, `cvae_all_lr3e-4_seed42`) | 0.570 / 1.098 | 0.223 / 0.355 | 0.438 / 0.831 | 0.107 / 0.327 | 0.089 / 0.275 | 0.640 | 0.078 |
| `pooled7` (old, `cvae_all7_lr3e-4_seed42`) | 0.561 / 1.075 | 0.220 / 0.346 | 0.427 / 0.809 | 0.105 / 0.318 | 0.086 / 0.263 | 0.608 | 0.086 |
| `pooled7` + oversample 4 | 0.551 / 1.058 | 0.221 / 0.349 | 0.426 / 0.807 | 0.076 / 0.229 | 0.057 / 0.173 | 0.607 | 0.091 |
| new, 3 seeds | 0.540±0.006 / 1.033±0.011 | 0.219±0.000 / 0.347±0.001 | 0.418±0.004 / 0.793±0.009 | 0.098 / 0.300 | 0.081 / 0.248 | 0.612 | 0.084 |
| **new, 3 seeds + oversample 4 (`stars_improved`)** | **0.536±0.006 / 1.027±0.012** | **0.219±0.000 / 0.348±0.001** | **0.416±0.002 / 0.789±0.004** | **0.071 / 0.214** | **0.053 / 0.162** | 0.611 | 0.087 |
| collision-aware (`stars_improved_social`, seed 42) | 0.587 / 1.118 | 0.257 / 0.423 | 0.441 / 0.833 | 0.018 / 0.084 | 0.014 / 0.065 | 0.587 | 0.110 |
| ground truth | | | | 0.020 / 0.073 | 0.010 / 0.039 | 0.625 | |

Per dataset (seed 2; single-sample ADE in m, then close pairs < 0.2 / < 0.4 m in %):

| dataset | GT close pairs | `pooled6` | new | new + oversample 4 | net speed GT / `pooled6` / new + os4 |
|---|---|---|---|---|---|
| STEPS | 0.078/0.485 | 0.149, 0.084/0.451 | 0.147, 0.079/0.436 | 0.145, 0.050/0.351 | 0.093 / 0.084 / 0.083 |
| PTDPES | 0.018/0.118 | 0.700, 0.055/0.186 | 0.697, 0.052/0.179 | 0.690, 0.032/0.118 | 0.470 / 0.452 / 0.450 |
| MADRAS | 0.000/0.000 | 0.554, 0.012/0.036 | 0.572, 0.015/0.036 | 0.576, 0.006/0.017 | 0.382 / 0.366 / 0.368 |
| Grand Central | 0.065/0.153 | 0.624, 0.134/0.372 | 0.627, 0.130/0.361 | 0.624, 0.110/0.291 | 1.099 / 1.104 / 1.093 |
| UCY | 0.155/1.493 | 0.472, 0.805/3.024 | 0.451, 0.747/2.784 | 0.455, 0.331/1.320 | 0.760 / 0.745 / 0.732 |
| ETH | 0.000/0.089 | 0.648, 0.731/2.340 | 0.596, 0.596/2.011 | 0.596, 0.167/0.630 | 1.330 / 1.332 / 1.321 |
| ATC | 0.003/0.010 | 0.841, 0.119/0.361 | 0.674, 0.105/0.319 | 0.669, 0.072/0.217 | 0.669 / 0.707 / 0.664 |

**Against `pooled6`:**
- **Accuracy:** single-sample ADE -6%, joint ADE -5%, minADE -2%.
- **Collisions:** -35% at 0.4 m and -34% at 0.2 m (-41% excluding groups).
- **Largest collision gains:** ETH (2.34% to 0.63% at 0.4 m) and UCY (3.02% to 1.32%, now below ground
  truth 1.49%).
- **Remaining gap:** close pairs are still 3-4x ground truth overall. Only the collision-aware variant reaches
  ground-truth rates, at an accuracy cost.
- **Speed:** net speed is within 2% of ground truth (0.611 vs 0.625).
- **Collision-aware variant** (`stars_improved_social`): close pairs at ground-truth level (0.018 / 0.084% vs
  0.020 / 0.073%; UCY 0.91% vs 1.49%, ETH 0.50% vs 0.09%). Its samples are coordinated across agents. The cost
  against `stars_improved` is single-sample ADE +10% (0.587, about `pooled6` level), minADE +17% (less sample
  spread), and net speed -4% (0.587 vs 0.625 ground truth). Use it where realistic interaction matters more
  than best-of-20 accuracy, e.g. as a crowd simulator.
- **Caveat:** ETH and UCY training scenes overlap the ETH/UCY benchmark test scenes, the same leakage as
  `pooled6` in the forecast benchmark.

## Code changes (all opt-in; defaults reproduce the old models and old checkpoints load unchanged)
- **`src/baselines/stars_ma_model.py`:**
  - decoder settings `refine` (future-aware refinement rounds) and `oversample` (joint-sample selection
    in `predict()`);
  - `training_forward(prior_samples=, fixed_prior_samples=)`;
  - `observed_groups` and `select_scene_samples`.
- **`src/baselines/train_stars_ma.py`:**
  - `--refine`, `--collision-weight`, `--collision-radius`, `--collision-fixed-prior` (default on),
    `--bom-weight` / `--bom-samples`, `--goal-beyond-horizon`;
  - `latest.pt` is saved every epoch, so 30-60 min jobs can resume.
- **`src/baselines/stars_ma_openloop.py`:** the new open-loop realism report.
- **`configs/stars_ma/all7.yaml`.**
- **Tests:** in `tests/test_stars_ma.py` and `tests/test_stars_ma_batch.py`.

## Commands (from the repository root, `.venv` active, `PYTHONPATH=src`)
```bash
# final baseline seeds (60 epochs; resumable)
bash results/stars_improve_2026-11-15/scripts/train.sh final_base_s2 60 --seed 2
# collision-aware variant
bash results/stars_improve_2026-11-15/scripts/train.sh final_social_s42 60 --seed 42 --refine 1 --collision-weight 10 --kl-weight 0.15
# open-loop test report, with inference-time sample selection
python -m baselines.stars_ma_openloop --split test --datasets steps,ptdpes,madras,grand_central,ucy,eth,atc \
  --oversample 4 --checkpoint new=experiments/stars_ma/runs/si_final_base_s2/checkpoints/best.pt --output test.json
# tables
python results/stars_improve_2026-11-15/scripts/test_table.py test/final_os1.json test/final_os4.json:_os4
```
Slurm: `scripts/node1h.slurm` (submit with `--time=00:30:00`) runs four GPU workers on `runs/stars_improve/q*`
request files and requeues unfinished ones before the time limit.
