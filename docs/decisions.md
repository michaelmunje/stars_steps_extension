# Decisions log

Choices made while working autonomously, for review. Newest last. Each entry: the decision,
why, and how to undo it.

## 2026-10-21

1. **Slurm jobs are allowed again.** The earlier "no new jobs" instruction was lifted by the
   request to "iterate on training it and optimizing the training for the available GPUs".
   Jobs use the `debug` partition and allocation `CDA26011`; benchmark/profiling jobs are kept
   to <= 2 h, training jobs to <= 12 h.

2. **Training data path rewritten for scale** (`src/baselines/stars_ma_batch.py`). Per-window
   Python graph building (2-11 ms/window) was replaced by cached `WindowStore`s on the GPU and
   batched on-device graph construction; it is bit-identical to `build_graph` (tests). The old
   `StarsMultiAgentDataset` path is kept for reference and the original tests.

3. **Batches are sized in pedestrians (`--max-nodes`), not windows**, because datasets differ
   by ~50x in people per window (ETH ~8, MADRAS up to 544). Default 16,384 per GPU: the largest
   power of two that fits every mode except closed-loop training (see benchmark below).

4. **Learning rate scales with the global batch** (`--lr-scaling sqrt`, relative to the STARS
   recipe's ~600 pedestrians per step, capped at 3e-3), with 2 warm-up epochs and cosine decay.
   This is a heuristic, not tuned; the Optuna search (next phase) should replace it. Override
   with `--learning-rate`.

5. **bf16 autocast is the default**; the decoder's GRU recurrence and all graph construction
   stay in fp32 so positions do not accumulate bf16 rounding (resolution would be ~4 cm at 10 m).
   bf16 is 1.5x faster and uses 40% less memory than fp32 (benchmark).

6. **Forecast checkpoints are selected on the macro mean of per-dataset validation agent ADE**
   (`--selection macro`), so a large dataset (PTDPES, STEPS) does not drown the small ones in
   multi-dataset runs. Identical to pooled ADE for one dataset.

7. **Dataset `weight` multiplies a dataset's per-window sampling rate** (1 = natural mixture).

8. **ATC is not in `all.yaml`**: `datasets/atc` is empty and `atc_subset` has no scene index.
   Building it needs `python -m preprocess datasets/atc_subset ...` (CPU, untested size).

9. **ETH has no validation windows** under the grouped split (two recordings), so in
   multi-dataset runs it contributes training data only.

10. **Fused attention batch limit**: GPU attention kernels refuse > 65,535 sequences per call;
    temporal transformers now run in chunks of 32,768 sequences (identical results).

11. **`torch.compile` needs `CC=gcc`**; both Slurm launchers now export `CC=gcc CXX=g++`
    (the system `nvc` breaks Triton's helper build).

12. **Stale git index locks** were left in the worktree twice by killed git processes; they
    were removed after checking no git process was running.

13. **Intermittent test failure**: one full-suite run failed a single test once with `-x`; three
    reruns passed. Not yet identified; watch for it.

14. **Pushing**: the scaling work goes to `main` of `stars_steps_extension` once optimisations
    are in; later work (Optuna search, SEAN 3.0 simulator) continues on a new branch.

15. **SEAN 3.0 simulator lives in this repository** (a `sean3/` package next to `src/`), because
    it will use the trajectory forecaster for pedestrians and share the dataset loaders.
    Splitting it into its own repository later is straightforward.

## Benchmark (job 65466, one GB200 node, STEPS unless noted)

| Mode | Pedestrians/GPU-batch | Pedestrians/s | Peak memory |
| --- | --- | --- | --- |
| Phase 1 (SSL) | 4,096 | 43k | 15 GB |
| Phase 1 | 16,384 | 50k | 57 GB |
| Phase 1, fp32 | 16,384 | 33k | 92 GB |
| Phase 1, fully connected graphs | 16,384 | 33k | 140 GB |
| Phase 1, 4 GPUs DDP | 16,384 x 4 | 174k (87% scaling) | 57 GB |
| Deterministic forecaster | 16,384 | 47k | 59 GB |
| CVAE forecaster | 16,384 | 48k | 59 GB |
| CVAE, 4 GPUs DDP | 16,384 x 4 | 168k | 59 GB |
| CVAE + 4-step rollout loss | 4,096 | 10.6k | 59 GB |
| any | 65,536 | out of memory | |
| Phase 1, all six datasets | 16,384 | 56k | 55 GB |

## Decisions on this branch

The full log (entries 1-101, with every experiment and job) is on branch `scale-and-sim-archive`.
Results: [`results/README.md`](results/README.md).

**Forecaster**
- **CVAE decoder, KL weight 0.1.** At 1.0 the latent collapses (samples about 9 mm apart); at
  0.1 samples are diverse and agent ADE on STEPS falls from 0.110 to 0.080. The KL weight explains
  most of the variance in the hyperparameter search; everything else is within seed noise.
- **Fine-tune the phase-1 encoder.** Fine-tuned beats from scratch, which beats frozen, by small
  margins. Closed-loop (multi-step) training did not help and cost 3-15x compute.
- **Pooled training needs LR 3e-4** (`configs/stars_ma/all.yaml`). At the STEPS-tuned LR it was
  seed-sensitive; at 3e-4 the six-dataset model matches or beats a per-dataset model everywhere.
- **References:** `cvae_kl01_finetune` (STEPS) drives the simulated crowd and the robot's
  predictor, because its rollouts keep the real walking speed; `cvae_all_lr3e-4_seed42` is the
  pooled model. Both are in release `weights-v1`.
- `train_stars_ma.py` skips steps with non-finite gradients and exits multi-GPU runs without
  interpreter teardown (a bus error there marked finished jobs as failed).

**Simulator (SimpleNavSim)**
- New batched PyTorch code; SEAN 2.0 was only a reference for behaviours and metrics.
- **Scenarios:** nine seeded kinds (eight scripted, one recorded STEPS window); train, validation
  and test seeds are disjoint, recorded scenes split by recording.
- **Pedestrian models:** replay, social forces, reactive replay, and the STARS crowd: after a
  3.2 s warm-up on the script, the forecaster moves everyone (re-forecast every 0.4 s, seeded
  AR(1) intents so the robot-absent run uses the same draws). With the robot in its graph,
  pedestrians react to it as to another person; they slide along walls and never overlap the robot.
  ORCA and a STARS + social-force hybrid were tried and removed.
- **Disruption metric `robot_influence`:** how much the robot changes the STARS forecasts. Trajectory
  counterfactuals in a learned crowd diverge chaotically, so they do not measure the robot's effect.
- **Success** within 1.0 m of the goal; at 0.3 m the robot orbited the goal in dense flow.
- **Controller:** constant `(v, w)` arcs (6 x 11 plus reverse) scored every 0.1 s over about 2 s;
  arcs within 0.05 m of a pedestrian in the next 1 s, or touching a wall, are removed (hard
  safety filter); 11 cost weights are tuned.

**Tuning**
- **v4** (`simplenavsim/configs/cmaes_crowd_v4.yaml`): CMA-ES, population 32, 100 generations,
  in the STARS crowd, objective J = success - 2 collision - robot_influence - 0.3 time
  - intimate-zone rate - (1 - path efficiency) - 0.1 jerkiness - 0.2 loops. Kept by validation.
- **CMA-ES vs PPO:** PPO over the same 11 weights, same objective and scenes, ties with CMA-ES;
  re-tunes from v4 keep v4. A PPO network choosing subgoals for v4 is slightly worse; PPO choosing
  arcs directly collides 4-5%. v4 is a local optimum of this controller and objective. Random
  weights from the same search range: 61% success, 11.5% collisions in the training crowd.
- **Noise:** scores in the STARS crowd vary by about 0.01-0.02 between identical runs (bf16), so
  smaller differences are not measurable.
- Tried and dropped: extra cost terms and memory against dense-flow loops, wait-then-go plans,
  tuning in a pooled-data crowd, behaviour-cloned and per-step-adjusting networks.

**Evaluation across pedestrian models** (`simplenavsim/scripts/evaluate.py`)
- Every tuned policy reaches 95-100% success with at most 1.9% collisions under all five models.
- The objective is comparable only within one model: `robot_influence` exists only in the STARS
  crowd that sees the robot.
- v4's tuning is specific to the STARS crowd: with social-force pedestrians the untuned controller
  scores higher.
- Weak spots: head-on in the STARS crowd (6.7% collisions); narrow passages with pedestrians that
  ignore the robot (27% timeouts).
- **Pedestrians following the robot is learned, not a bug.** Forecasts are rotation-, translation-
  and order-equivariant to 1e-5 m. Pedestrians follow the robot about twice as often when STARS
  sees it (6.6% vs 2.5% of near-robot time): in STEPS a close person walking the same way is
  usually a group member. Not fixed; options are keeping the robot out of the graph with a
  social-force repulsion, or retraining with the robot as its own node type.
