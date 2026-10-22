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
    are in; later work (Optuna search, SEAN-JAX simulator) continues on a new branch.

15. **SEAN-JAX simulator lives in this repository** (a `seanjax/` package next to `src/`), because
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

## 2026-10-21 (continued): simulator and profiling

16. **The simulator package is `seanjax`** (renamed from `sean3` as requested). Despite the name
    it is implemented in PyTorch, which the specification allows ("PyTorch ... with JAX
    optional"); a JAX port is not planned for now. Package: `src/seanjax/`; configs and scripts:
    `seanjax/configs/`, `seanjax/scripts/`.

17. **Isaac Sim / Isaac Lab (Phase 3) is high risk on this cluster.** Findings (survey of the
    official docs and package indexes):
    - Isaac Sim 6.2.0 pip wheels do exist for aarch64 / Python 3.12.
    - It officially supports only Ubuntu 22.04/24.04, and aarch64 only on DGX Spark.
    - GB200 is a data-centre Blackwell part, likely without RT cores, so RTX rendering and
      camera sensors are probably unavailable.
    - Isaac Lab 3.0.0 needs a urllib3 override to resolve.
    - Apptainer is available on compute nodes only (`module load tacc-apptainer`).

    Decision: build Phases 1-2 on the lightweight batched backend (which runs on the GPUs).
    Isaac is attempted later only as a headless physics smoke test inside an Apptainer
    container; it is not on the critical path.

18. **ROS 2 (Phase 4)** is obtainable without sudo via RoboStack (`robostack-jazzy`, linux-aarch64,
    py3.12: ros-base, rclpy, nav2). `people_msgs` is missing there. Not installed yet.

19. **ORCA (Phase 2)**: there is no pip binding. `pyrvo` 0.4.3 builds from sdist with `CC=gcc CXX=g++`
    and can serve as the reference oracle for a batched torch ORCA port.

20. **Robot after the episode ends**:
    - A robot that reached its goal leaves the scene.
    - A robot that collided or timed out stays as a stopped obstacle.

    Otherwise an early crash would look less disruptive than a careful pass.

21. **Reactive pedestrians**:
    - Social-force repulsion from pedestrians and the robot has a sideways component
      (`sidestep: 0.5`, as in SEAN 2.0's agents). Without it, a pedestrian stalls ~1.5 s face
      to face with a robot.
    - Pedestrians cannot interpenetrate the robot: any overlap is resolved after each step.
      So a robot-pedestrian collision comes from the robot's motion, matching SEAN's
      robot-at-fault counting.
    - Robot repulsion was raised to strength 5, range 0.5 m.

22. **Recorded STEPS scenarios**:
    - Each scenario is the busiest of 8 random 30 s windows.
    - The robot crosses the occupied area through where the crowd is at mid-window.
    - Recordings are split by STEPS's grouped manifest.

    Random windows were too sparse: the robot never came within 3.9 m of anyone.

23. **CMA-ES objective weights** (success 1, collision 2, disruption 1, time 0.3, comfort 0.5;
    disruption = mean counterfactual displacement, comfort = personal-zone fraction) are a first
    guess. Change them in `seanjax/configs/cmaes*.yaml`.

24. **Profiling (job 65475)**: in closed-loop training, efficient-attention backward with dropout
    takes 28% of GPU time, bf16<->fp32 casts 22%, LayerNorm backward 12%, matmuls 12%.
    - `--attention math` is +10% faster for phase 1 (54k vs 49k pedestrians/s, 64 vs 57 GB).
      Its closed-loop number is pending.
    - `torch.compile` is slower (46k) and compiles for 66 s.
    - Activation checkpointing costs 27% but fits 64k pedestrians per batch.

    Defaults are unchanged until the closed-loop comparison is in.

25. **Never modify `.venv` while jobs are starting**: a job importing during `uv add` saw
    inconsistent package metadata and failed (`A distribution name is required`).

26. **Dense-flow scenario bug fixed**: pedestrians used to spawn at random corridor positions at
    random times, sometimes inside the robot. That caused unavoidable collisions, which dominated
    the first CMA-ES run (job 65489, discarded). Now half of them start at t = 0 away from the
    robot and the rest enter from the corridor ends. Defaults then succeed in 48/48 validation
    scenarios. The CMA-ES runs were enlarged (population 32, 150 generations, 6 training and 12
    validation scenarios per kind) since a generation costs ~3 s on a GB200.

27. **CMA-ES v1 results (job 65497)**, on held-out test scenarios:
    - Scripted only (80 scenarios): the tuned policy reaches objective 0.612, versus 0.439 for
      the default social controller and -0.572 for goal-seeking. It has 0% collisions (vs 2.5%),
      3x less disruption (0.065 m vs 0.204 m) and 2x the minimum clearance. The cost is 1.4 s
      slower trips and 5% timeouts.
    - With recorded STEPS traffic (90 scenarios): 0.521 vs 0.437, with 100% success, 0
      collisions and faster trips, but no disruption gain.

    The search did not converge: the step size kept growing and the mean's validation score fell
    after generation 90. The causes are objective noise (fresh scenarios each generation) and
    weight parameters sitting at their bounds (pedestrian_weight 17.4 of 20). v2 widens the
    upper bounds of the five weights (goal 50, collision/pedestrian/disruption/obstacle 200) and
    doubles the training scenarios per generation (12 per kind). Default parameters are
    unchanged.

28. **Batch size matters more than expected (to confirm).** The CVAE trained with the closed-loop
    loss runs at 4,096 pedestrians per batch (358 steps/epoch). It reached 0.080 m validation
    agent ADE by epoch 3. The plain fine-tuned CVAE runs at 16,384 (~90 steps/epoch) and was at
    0.095 m after 49 epochs. Either the closed-loop loss helps, or large batches with sqrt LR
    scaling under-train on STEPS's ~1.5M training pedestrians. The Optuna searches include
    max_nodes 4,096/8,192/16,384 and should separate the two. If smaller batches win, the
    default per-GPU batch drops and scale comes from more GPUs/seeds rather than larger
    batches.

29. **The tuned navigation policy transfers across pedestrian models.** The v1 policy was tuned
    against reactive replay; on test scenarios with other pedestrian models (objective; default
    social -> tuned):
    - ORCA: 0.636 -> 0.726.
    - Social force: 0.248 -> 0.442.
    - Replay: 0.427 -> 0.700. Here goal-seeking collides in 65% of scenarios and the default
      social controller in 8.7%.

    The tuned policy has 0 collisions under all three, so tuning against one pedestrian model
    did not overfit to it.

30. **Early Optuna trends** (30-epoch trials; validation best-of-6 agent ADE ~0.081-0.086 m vs
    ~0.095 m best-of-20 for the hand-set recipe):
    - The best trials use KL weight 0.09-0.18 (recipe: 1.0) and dropout 0.02-0.12 (recipe: 0.2).
    - They use 4-8k pedestrians per batch, decoder width 512 and LR 4e-4-1e-3.

    The recipe defaults stay until the searches finish; the winning settings then get full-length
    runs with several seeds.

31. **First forecaster test result (deterministic decoder, frozen phase-1 encoder, STEPS test,
    12,500 windows).**

    | Metric | STARS | Constant velocity |
    | --- | --- | --- |
    | All-agent ADE / FDE | 0.123 / 0.219 m | 0.276 / 0.518 m |
    | Moving agents ADE / FDE | 0.604 / 1.192 m | 0.894 / 1.771 m |
    | Stationary agents ADE | 0.046 m | 0.178 m |

    - Closed-loop forecasting is stable: ADE 0.123-0.127 m at 1, 4 and 12 executed steps.
    - 20 s simulations are too passive: mean speed 0.057 m/s (real 0.141), mean acceleration
      0.03 m/s^2 (real 0.31). The deterministic decoder regresses toward standing still.

    So the deterministic forecaster is not used as a pedestrian *simulator*. The CVAE (sampled
    intents) is the candidate for that and is evaluated next. For the navigation policy's
    *predictions*, the CVAE prior mean is used (`policy_stars.yaml`).

32. **The CVAE's latent collapses at KL weight 1.0.** On the test split:

    | CVAE | Agent ADE (best of 20) | Moving ADE | Final-position spread |
    | --- | --- | --- | --- |
    | Frozen encoder | 0.113 m | 0.545 m | 0.021 m |
    | From scratch | 0.126 m | 0.625 m | 0.014 m |

    The samples are nearly identical, so best-of-K is barely better than one sample, and the
    20 s simulations are as passive as the deterministic ones. This matches the Optuna searches'
    preference for KL weight 0.09-0.18. Next: retrain the CVAE with the searched low KL weight
    and check sample diversity and simulation speed.

33. **CMA-ES v2 (wider bounds, 2x scenarios) is not better than v1 on scripted scenarios.** Test
    objective: scripted 0.547 vs v1 0.612; recorded 0.538 vs v1 0.521. v2 found a faster, more
    direct trade-off: time to goal 11.1 vs 14.5 s, disruption 0.185 vs 0.065 m. With the current
    objective weights the two are close. The best scripted policy stays
    `checkpoints/seanjax/cmaes/best_policy.yaml` (v1); the best recorded-traffic policy is
    `checkpoints/seanjax/cmaes_recorded_v2/best_policy.yaml`.
    - `evaluate.py` labelled policy files by file name only, so same-named files overwrote each
      other in comparisons. Now labelled `<folder>/<name>`.

34. **It is the KL weight, not the batch size (revises 28).** Every small-batch CVAE run peaks in
    the first epochs and then stalls: 1k pedestrians/batch 0.085 m at epoch 2, 4k 0.092 at
    epoch 3, 4k + closed-loop loss 0.080 at epoch 3. Those epochs are inside the KL warm-up
    (weight 0.1 -> 1.0 over 10 epochs). The runs are best while the weight is ~0.2-0.3 and get
    worse as it reaches 1.0 and the latent collapses. The deterministic 4k-batch run, which has
    no KL term, improves steadily (0.101 at epoch 12, close to the 16k run's 0.099). Fixed
    low-KL runs (0.1, job 65571) test this directly. If confirmed, the forecast recipe's
    kl_weight default becomes ~0.1.
    `docs/results/stars_ma_steps_test.md` (written by `baselines.report_stars_ma`) has the test
    table so far.

35. **Low KL confirmed; the default CVAE KL weight is now 0.1.** Validation agent ADE, best of
    20, early epochs:

    | CVAE, encoder | KL 1.0 | KL 0.1 |
    | --- | --- | --- |
    | Fine-tuned | 0.095 | 0.072 |
    | Frozen | 0.098 | 0.074 |
    | Search config, from scratch | — | 0.075 |

    That is ~24% lower error. `FORECAST_RECIPE["kl_weight"]` changed 1.0 -> 0.1. Earlier runs are
    labelled by their run directories and keep their recorded configs. The KL-0.1 runs still
    peak early (epoch 5-7 so far), so an even lower weight or a longer warm-up is the next
    thing to try if they stall.

36. **STARS forecasts in the navigation policy (first run, KL-1.0 fine-tuned CVAE, prior mean).**
    Scripted test scenarios:
    - Default parameters: objective 0.460 with STARS vs 0.439 with constant velocity;
      collisions 1.3% vs 2.5%.
    - After CMA-ES: 0.543 with STARS. Constant velocity: v2 0.547, v1 0.612.

    The tuned runs are not paired (different search runs), and the forecaster's latent had
    collapsed. Next: rerun with the best low-KL forecaster, and compare predictors by evaluating
    the *same* parameter sets with each predictor (paired).

37. **Paired predictor comparison: STARS forecasts help the navigation policy.** Same parameters,
    80 scripted test scenarios, constant velocity -> best low-KL STARS CVAE
    (`cvae_hpo0_scratch_full`, prior mean):

    | Policy | Constant velocity | STARS |
    | --- | --- | --- |
    | Social, default | 0.439 (2.5% collisions) | 0.536 (0 collisions, 100% success) |
    | Tuned v1 (cautious) | 0.612 | 0.647 (disruption 0.065 -> 0.059 m), best so far |
    | Tuned v2 (fast) | 0.547 | 0.525 |
    | Tuned with the old STARS model | 0.560 | 0.542 |

    The two fast policies lose about 0.02. Their weights were tuned around constant velocity's
    errors. CMA-ES with the best forecaster in the loop (job 65618) tests whether re-tuning
    recovers that. Tables: `runs/seanjax/paired/{constant_velocity,stars_best}/`.

38. **Best forecaster so far: low-KL CVAE from scratch with the search settings**
    (`cvae_hpo0_scratch_full`). STEPS test, best of 20, versus constant velocity:

    | Metric | Low-KL CVAE | Constant velocity |
    | --- | --- | --- |
    | All-agent ADE / FDE | 0.081 / 0.129 m | 0.276 / 0.518 m |
    | Moving ADE / FDE | 0.342 / 0.600 m | 0.894 / 1.771 m |
    | Joint ADE | 0.097 m | 0.279 m |

    - Against the previous best (KL-1.0 fine-tuned CVAE: moving ADE 0.532 m) that is -36%.
    - Samples are now diverse: 0.16 m final spread, against 0.009 m before.
    - 20 s simulation mean speed is 0.102 m/s (real 0.141; KL-1.0 models 0.06-0.08).
    - Simulated acceleration remains far below the data (0.03 vs 0.31 m/s^2). Part of the data
      value is likely tracking jitter at 2.5 Hz; to be checked against smoothed tracks before
      treating it as a model deficiency.

    Seeds 1-2 of this configuration (job for `cvae_hpo0_scratch_seed{1,2}`) measure the spread.

39. **Most of the real data's acceleration is tracking jitter.** STEPS test futures:

    | | Mean speed | Mean acceleration |
    | --- | --- | --- |
    | Raw 2.5 Hz positions | 0.141 m/s | 0.31 m/s^2 |
    | After one 1-2-1 smoothing pass | 0.115 m/s | 0.12 m/s^2 |
    | Raw, standing pedestrians only | — | 0.21 m/s^2 (pure jitter) |

    Against smoothed data the best forecaster's simulated speed (0.102 m/s) is close to real.
    Its acceleration (0.03 m/s^2) is still ~3x too low, so simulations are smoother and slightly
    more passive than reality. The evaluator now also reports `real_smoothed_*` statistics and
    Wasserstein distances against them.

40. **CMA-ES from scratch does not find the best policy region; warm-start instead.**
    - Tuning with the best STARS forecaster in the loop reached 0.544 on the scripted test,
      against 0.536 for the default with the same predictor. The CV-tuned v1 policy reaches 0.647
      with that predictor.
    - goal_weight again ended at its bound (49.6/50).
    - The recorded-traffic run with the older forecaster ended at 0.520.

    Changes:
    - `train_refresh_every` holds the training scenarios fixed for N generations (common random
      numbers; default 1 keeps the old behaviour).
    - `cmaes_stars_best_warm.yaml` warm-starts from the v1 policy with sigma 0.1 and refresh 10,
      fine-tuning it with STARS forecasts in the loop.

41. **Optuna (after 72 + 49 trials) confirms the KL weight is what matters.**

    | Study | Most important parameters (fANOVA) | Batch size |
    | --- | --- | --- |
    | From scratch | KL weight 58%, dropout 19% | 4% |
    | From phase-1 encoder | KL weight 51%, warm-up 13%, latent size 10% | 2% |

    Frozen vs fine-tuned encoder accounts for 1%. The top trials share:
    - KL 0.07-0.09 and decoder width 512;
    - an encoder of width 128 (smaller than the 256 recipe);
    - 4,096 pedestrians/batch, LR 6e-4-1e-3, warm-up 2-4 epochs.

    The best trial (53, val 0.0798 best-of-6) is now trained at full length with seeds 42 and 1
    (`cvae_hpo53_scratch_seed*`). Reports: `docs/results/hpo_steps_cvae_{scratch,ssl}.md` (from
    `baselines.hpo_report`).

42. **Bus error at exit in multi-GPU all-dataset runs.** `ssl_all` (job 65478) finished normally:
    early stop at epoch 294, best 5.361 at epoch 274, checkpoint saved. Rank 0 then died with a
    bus error during interpreter teardown, so Slurm reported FAILED. The 4-GPU all-dataset
    benchmark run showed the same (exit 135). Cause not diagnosed; /dev/shm or CUDA/NCCL teardown
    are the usual suspects. Workaround: after a successful multi-GPU run every rank synchronises,
    shuts down distributed communication, flushes and `os._exit(0)`s. Single-process runs are
    unchanged.

43. **Controller fix: the tuned policy drove in loops instead of waiting.** The side-by-side
    figure (`seanjax/scripts/compare_policies.py`, `docs/results/navigation_compare_stars.png`)
    showed the v1-tuned policy orbiting in `dense_flow` and `narrow_passage`. Three causes:
    - The speed-tracking cost penalised slowing down.
    - The tuned weights (pedestrian 17.4, goal 0.76) made circling cheaper than waiting.
    - The robot could not reverse (minimum speed 0).

    Changes, all fixed controller/robot settings (not tuned):
    - `retreat_cost: 2.0` penalises negative progress toward the goal.
    - The speed cost fades to zero when a pedestrian is within 2x personal space.
    - The robot may reverse at up to 0.3 m/s (`min_speed: -0.3`, a reverse candidate in the
      controller), as Jackal-class bases can.

    The narrow-passage loop became a short back-off and dense-flow orbits mostly disappeared.
    Navigation numbers in entries 27-40 used the old controller. The warm-start CMA-ES run (job
    65638) and all later evaluations use the new one, and re-evaluate the reference policies.

44. **Seed variance and the current best forecaster.**
    - Search configuration 0 (low-KL CVAE from scratch), seeds 42/1/2: validation agent ADE
      0.0698 / 0.0693 / 0.0707 m (mean 0.0699, sd ~0.0007). Differences between configurations
      below ~0.002 m are within noise.
    - Best on the STEPS test split: `cvae_kl01_finetune` (low-KL CVAE, fine-tuned STEPS phase-1
      encoder). Agent ADE/FDE 0.080/0.126 m, moving 0.336/0.588 m, joint ADE 0.097 m.
    - Its 20 s simulations move at 0.115 m/s mean speed, equal to the smoothed real tracks.
    - Pretraining helps slightly once the KL weight is right: fine-tuned 0.336 < scratch 0.342
      < frozen 0.362 moving ADE.
    The navigation predictor (`policy_stars_best.yaml`) now uses `cvae_kl01_finetune`; the warm-start CMA-ES run (job 65638, not yet started) picks it up.
