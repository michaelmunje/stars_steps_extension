# Closed-loop rollout fine-tuning of STARS (2026-11-18)

**Question.** STARS (multi-agent CVAE forecaster) is trained open-loop but used closed-loop in the ETH/UCY
realism simulator, where it re-forecasts every 0.4 s from its *own* executed positions. There it lost to
constant velocity (CV) at 8/12 s. Does fine-tuning it on its own multi-step rollouts close the gap?

**Answer.** Yes for accuracy. Six short epochs of closed-loop fine-tuning of `stars_improved`
(`si_final_base_s2`) give the following (all 549 windows, one rollout):

- Closed-loop ADE drops from 0.521 / 0.989 / 1.567 m to **0.443 / 0.809 / 1.263 m** at 4.8 / 8 / 12 s.
- That beats CV (0.497 / 0.907 / 1.428) at every horizon. The paired differences vs CV are -0.054, -0.098
  and -0.164 m, and all three 95% block-bootstrap CIs exclude 0.
- It beats ORCA (0.559 / 0.896 / 1.235) at 4.8 s and 8 s. At 12 s it is level with ORCA (+0.028 m, CI -0.044 to 0.102).
- On the held-out pooled-test recordings (seq_eth, crowds_zara02), it is the best method at every horizon:
  0.375 / 0.560 / 0.725 vs CV 0.444 / 0.612 / 0.779 and ORCA 0.431 / 0.606 / 0.737.
- Colliding pairs fall from 2.78% to 1.99%. Speed realism improves: speed W1 goes from 0.052 to 0.040.
- Open-loop single-sample ADE on the test split also improves (macro 0.545 -> 0.531). Min-of-20 ADE barely
  moves (0.219 -> 0.222).

The cost is turning realism. Rollouts get straighter: mean turning rate at 8 s goes from 5.7 to 3.7 deg/s
(ground truth 12.8), turning W1 rises from 7.9 to 10.1, and acceleration W1 rises from 0.131 to 0.144. Collisions
stay well above ORCA's (HACS 0.10 vs 0.002). The collision-aware variant (`stars_improved_social`) gains
less: it is level with CV at 8/12 s and still worse at 4.8 s.

## Method

`train_stars_ma.py --init-checkpoint CKPT --rft-steps K` (all new flags are off by default; old
checkpoints load unchanged and the model code and state dict are untouched):

- **Simulator-faithful closed-loop rollout** (`stars_ma_model.closed_loop_rollout`). This is the same loop as
  `simplenavsim.realism.sim.Stars._rollout`. Every 0.4 s it re-encodes the scene from the agents' simulated
  8-step history, decodes, and executes the first predicted step for every agent. It then shifts the window and
  rebuilds the k-NN graph on the GPU (`GraphBatch.rebuild`).
- **Latents.** CVAE latents are drawn from the *prior* of the current context at each step. They use AR(1)
  noise, eps_t = 0.8 eps_{t-1} + 0.6 n_t, as in the simulator. A `posterior` option (teacher intent) exists,
  but it made closed-loop error worse (train/test mismatch; see "What was tried").
- **Loss.** The K executed positions are compared with ground truth (masked L2, `rft`), with full
  backpropagation through the rollout. Gradients flow through the executed positions into later graphs' node
  and edge features; the edge choice itself is not differentiable. `--rft-bptt N` truncation exists but was
  not needed.
- **Regulariser.** The original open-loop CVAE loss (forecast L2 + KL, and the collision hinge for the
  social checkpoint) is kept with weight 1.
- **Curriculum.** K ramps from 4 to 12 steps (1.6 -> 4.8 s) over 3 epochs. K <= 12 means the existing 8+12
  windows suffice, with no new data extraction.
- **Selection** (`--rft-eval-steps 12 --selection closedloop`). Each epoch reports the same closed loop on
  the pooled validation split (1 prior sample, seeded): per-dataset ADE/FDE, speed and turning. `best.pt` is
  chosen on the mean per-dataset closed-loop ADE. Epoch 0 is the input checkpoint, so fine-tuning can never
  select something worse on that metric.
- **Budget.** 2048 pedestrians per batch, 300 batches per epoch, LR 5e-5 (cosine), AdamW, 6 epochs, no ATC
  (configs/stars_ma/rft_no_atc.yaml). A 12-step rollout step costs about 18x a plain step (~1.3 steps/s,
  114 GB at 2048 nodes; activation checkpointing fits 4096 nodes in 33 GB). One run takes about 45 GPU-minutes.
  All runs, including the ones I dropped, came to roughly 8 GPU-hours.

This is the simplest TrafficSim-style variant: a differentiable closed-loop simulator with an imitation
loss on the rollout. It is not scheduled sampling with detached inputs, DAgger, or CAT-K-style
closest-sample rollouts. Backpropagating through the rollout was cheap enough, so no approximation was needed.

**Reusable step.** `runs/stars_rollout_ft/rft.sh INIT_CHECKPOINT RUN_DIR [flags]` fine-tunes any
multi-agent STARS forecast checkpoint. It reads the graph (knn/radius), KL weight and collision/BoM terms
from the checkpoint. Data defaults to the six non-ATC datasets; `--config`, `--datasets` or `--val-datasets`
override it (e.g. for the LOSO ETH/UCY models). The script is resumable.

## Validation (selection) — closed-loop ETH/UCY sim on the pooled-validation recording (students003), 3 seeds

`runs/stars_rollout_ft/cl_val.py`, 99 windows, simulator identical to the benchmark. Speed / turning are at 8 s; GT speed 0.561 m/s.

| model | ADE 4.8 s | ADE 8 s | ADE 12 s | coll. pairs 8 s | HACS 8 s | speed | speed W1 | accel W1 | turning (deg/s) | turning W1 |
|---|---|---|---|---|---|---|---|---|---|---|
| CV | 0.616 | 1.159 | 1.869 | 0.039 | 0.105 | 0.568 | 0.019 | 0.191 | 0.0 | 12.63 |
| stars_improved (base) | 0.660 | 1.296 | 2.192 | 0.036 | 0.105 | 0.582 | 0.066 | 0.147 | 4.77 | 8.63 |
| **rft_base** (= `stars_rft`) | **0.585** | **1.091** | **1.773** | 0.019 | 0.046 | 0.532 | 0.043 | 0.157 | 3.12 | 9.92 |
| rft_base + progress term | 0.591 | 1.107 | 1.783 | 0.020 | 0.051 | 0.519 | 0.043 | 0.160 | 3.23 | 9.86 |
| base, 4x oversampling | 0.664 | 1.306 | 2.212 | 0.033 | 0.093 | 0.586 | 0.066 | 0.147 | 4.84 | 8.55 |
| rft_base, 4x oversampling | 0.588 | 1.094 | 1.776 | 0.017 | 0.044 | 0.531 | 0.042 | 0.157 | 3.12 | 9.90 |
| stars_improved_social | 0.723 | 1.326 | 2.084 | 0.004 | 0.004 | 0.535 | 0.033 | 0.135 | 6.09 | 7.45 |
| rft_social (collision hinge kept) | 0.687 | 1.284 | 2.138 | 0.012 | 0.018 | 0.601 | 0.052 | 0.152 | 4.52 | 9.10 |
| **rft_social + progress** (= `stars_rft_social`) | 0.652 | 1.196 | 1.889 | 0.011 | 0.016 | 0.494 | 0.068 | 0.154 | 4.69 | 8.77 |

(`val/final.json`, `val/final_social.json`; `val/early.json` holds the earlier ATC-trained runs.)
In-training closed-loop validation (pooled val, all datasets, 4.8 s) for `rft_base_noatc` went from
0.595 m ADE at epoch 0 to 0.487 m at epoch 1 and 0.474 m at epoch 6, the selected epoch (`training/*.metrics.jsonl`).
Both social runs peaked at epoch 1 and then degraded (open-loop min-ADE also rose), so they were stopped
after epoch 3 and their epoch-1 `best.pt` was used.

## ETH/UCY closed-loop benchmark (all 549 windows, 1 rollout; appended to a copy of `results/ethucy_2026-11-16_improved`)

| method | ADE 4.8 s | ADE 8 s | ADE 12 s | colliding pairs % (8 s) | HACS (8 s) |
|---|---|---|---|---|---|
| CV | 0.497 | 0.907 | 1.428 | 3.54 | 0.130 |
| ORCA (calibrated) | 0.559 | 0.896 | **1.235** | 0.13 | 0.002 |
| SFM (calibrated) | 0.574 | 0.968 | 1.354 | 1.13 | 0.014 |
| stars_improved (4x os) | 0.521 | 0.989 | 1.567 | 2.78 | 0.116 |
| stars_improved_social | 0.576 | 1.043 | 1.621 | 0.44 | 0.005 |
| **stars_rft** | **0.443** | **0.809** | 1.263 | 1.99 | 0.101 |
| stars_rft_os4 | 0.445 | 0.812 | 1.273 | 1.73 | 0.090 |
| stars_rft_social | 0.517 | 0.928 | 1.422 | 0.96 | 0.026 |

Paired micro ADE differences, stars_rft minus baseline, with 95% block-bootstrap CI:

| baseline | 4.8 s | 8 s | 12 s |
|---|---|---|---|
| CV | -0.054 [-0.066, -0.041] | -0.098 [-0.123, -0.070] | -0.164 [-0.219, -0.100] |
| ORCA | -0.116 [-0.142, -0.093] | -0.087 [-0.128, -0.044] | +0.028 [-0.044, +0.102] |

By pooled split (STARS was trained on the `train` recordings, selected on `val`; `test` = seq_eth + crowds_zara02 are unseen):

| split | CV 4.8/8/12 | ORCA | stars_improved | stars_rft |
|---|---|---|---|---|
| train | 0.456 / 0.888 / 1.471 | 0.532 / 0.877 / 1.230 | 0.484 / 0.951 / 1.571 | 0.396 / 0.755 / 1.230 |
| val | 0.616 / 1.159 / 1.869 | 0.719 / 1.145 / 1.657 | 0.666 / 1.298 / 2.182 | 0.587 / 1.093 / 1.778 |
| test | 0.444 / 0.612 / 0.779 | 0.431 / 0.606 / 0.737 | 0.430 / 0.661 / 0.813 | **0.375 / 0.560 / 0.725** |

Kinematics at 8 s (all windows; W1 to ground truth):

| method | speed | speed W1 | accel W1 | turning (deg/s) | turning W1 |
|---|---|---|---|---|---|
| ground truth | 0.466 | 0 | 0 | 12.77 | 0 |
| CV | 0.481 | 0.017 | 0.171 | 0.00 | 13.17 |
| ORCA | 0.437 | 0.033 | 0.118 | 3.41 | 9.29 |
| SFM | 0.448 | 0.051 | 0.080 | 11.55 | 2.55 |
| stars_improved | 0.456 | 0.052 | 0.131 | 5.72 | 7.91 |
| stars_improved_social | 0.460 | 0.016 | 0.125 | 6.03 | 7.97 |
| stars_rft | 0.438 | 0.040 | 0.144 | 3.67 | 10.10 |
| stars_rft_social | 0.402 | 0.067 | 0.147 | 4.89 | 9.33 |

Full tables: `ethucy_closed_loop/tables.md`, `aggregate.csv`, `paired.csv`, `kinematics.csv`; per-window rows of
the new methods in `per_rollout_stars_rft.csv.gz`.

## Open-loop accuracy (test split, 7 datasets incl. ATC, K = 20, no oversampling; `stars_ma_openloop`)

Macro mean over datasets; collisions are pooled, in percent of pairs.

| model | ADE (1 sample) | FDE | minADE | minFDE | joint ADE | pair 0.2 % | pair 0.4 % | net speed | speed W1 |
|---|---|---|---|---|---|---|---|---|---|
| stars_improved (si_final_base_s2) | 0.545 | 1.044 | **0.219** | **0.349** | 0.419 | 0.098 | 0.300 | 0.617 | 0.086 |
| stars_rft | **0.531** | **1.016** | 0.222 | 0.351 | **0.415** | 0.097 | 0.298 | 0.613 | 0.085 |
| stars_rft + progress | 0.532 | 1.016 | 0.224 | 0.357 | 0.419 | 0.098 | 0.303 | 0.614 | 0.088 |
| stars_improved_social | 0.591 | 1.129 | 0.258 | 0.424 | 0.440 | 0.018 | 0.083 | 0.587 | 0.110 |
| rft_social | 0.575 | 1.111 | 0.262 | 0.426 | 0.444 | 0.021 | 0.093 | 0.591 | 0.101 |
| stars_rft_social | 0.574 | 1.109 | 0.262 | 0.427 | 0.441 | 0.023 | 0.098 | 0.583 | 0.112 |
| ground truth | | | | | | 0.020 | 0.073 | 0.625 | |

Open-loop accuracy did not degrade: single-sample ADE/FDE improve, and min-of-20 is within 1-2%
(`openloop_test/`).

## What was tried

| run (`experiments/stars_ma/runs/`) | data | variant | outcome |
|---|---|---|---|
| rft_prior_k12 | all 7 (with ATC) | prior latents, K 4->12 | val cl-ADE 0.565 -> 0.462; ETH/UCY val 0.600/1.110/1.767 (`val/early.json`); dropped when ATC was removed |
| rft_post_k12 | all 7 | posterior (teacher-intent) latents | worse than the start (val cl-ADE 0.654 at epoch 1); killed |
| rft_prior_k12_speed | all 7 | + per-step speed-matching term | kept speed, but turning per step rose 5x (zig-zag satisfies a speed match); replaced by the progress term |
| rft_social_prior_k12 | all 7 | from social checkpoint, *without* its collision hinge | collisions rose (pairs 0.004 -> 0.017); led to inheriting the checkpoint's loss terms |
| **rft_base_noatc** | 6, no ATC | prior latents, K 4->12, 6 epochs | **selected (`stars_rft`)** |
| rft_base_progress_noatc | 6 | + progress matching (distance from start, 1.0) | same accuracy, speed not better; not used |
| rft_social_coll_noatc | 6 | social checkpoint, collision hinge 10 kept | best at epoch 1, degrades after |
| **rft_social_coll_progress_noatc** | 6 | + progress matching | **selected (`stars_rft_social`)**, epoch 1 |

Earlier work (decision 47, STEPS only) found the old `--rollout-exec-steps` loss did not help. That loss used
the posterior intent, held z fixed, and selected on open-loop best-of-20. The difference here is training
and selecting on the closed loop the simulator actually runs: prior latents, one sample, and closed-loop ADE.

## Caveats

- One fine-tuning seed per variant and one benchmark rollout (as the stars_improved row). On the validation
  recording, closed-loop ADE is averaged over 3 rollout seeds, and the 1x vs 4x oversampling rows (independent
  draws) agree within 0.01 m, far below the gains.
- The new models are straighter than both the ground truth and the base model, which is the expected effect
  of an L2 rollout loss under uncertainty. Turning W1 is worse than stars_improved, though better than CV.
  Speed falls slightly below the ground truth (0.438 vs 0.466 m/s).
- Collisions improve relative to the base model, but only the collision-aware variant approaches ORCA/SFM.
  That variant pays in accuracy, and its fine-tuning was unstable beyond one epoch.
- Validation for selection used students003 (pooled val). Test recordings were only evaluated after the
  checkpoints were fixed.

## Commands

```bash
# fine-tune (any checkpoint; resumable; GPU)
bash runs/stars_rollout_ft/rft.sh experiments/stars_ma/runs/si_final_base_s2/checkpoints/best.pt \
     experiments/stars_ma/runs/rft_base_noatc --rft-speed-weight 0 --epochs 6
bash runs/stars_rollout_ft/rft.sh experiments/stars_ma/runs/si_final_social_s42/checkpoints/best.pt \
     experiments/stars_ma/runs/rft_social_coll_progress_noatc --epochs 6        # epoch-1 best.pt used
# closed-loop ETH/UCY validation (students003)
python runs/stars_rollout_ft/cl_val.py --out results/rft_val/final.json --seeds 3 cv rft_base=experiments/stars_ma/runs/rft_base_noatc/checkpoints/best.pt
# benchmark (stars_rft / stars_rft_os4 / stars_rft_social are registered in sim.py + ethucy_benchmark.py)
python simplenavsim/scripts/ethucy_benchmark.py run --run results/ethucy_rft_eval --append --methods stars_rft stars_rft_os4 stars_rft_social --seeds 1
python simplenavsim/scripts/ethucy_benchmark.py aggregate --run results/ethucy_rft_eval
python runs/stars_rollout_ft/bench_table.py results/ethucy_rft_eval
# open-loop test
bash runs/stars_rollout_ft/final_eval.sh openloop
```

Slurm: `runs/stars_rollout_ft/{node.slurm,worker.sh,keeper.sh}`. The queue of `*.req` command files per
GPU is copied from `runs/stars_improve`; idle workers exit. Checkpoints:
`experiments/stars_ma/runs/rft_base_noatc/checkpoints/best.pt` (sha256 9f6cbfde15b9bbbc…),
`rft_social_coll_progress_noatc/checkpoints/best.pt` (9d00693a8f7e4f00…).
