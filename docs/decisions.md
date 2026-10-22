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

45. **The warm-start policy scored best by reward hacking; objective v2.** Warm-start CMA-ES (from
    v1, fixed controller, best STARS predictor) reached test objective 0.707 vs 0.576 for v1:
    - disruption 0.032 m, personal-space violations 0.075, min clearance 1.24 m;
    - but command smoothness 2.85 (4x the others), and five parameters at their bounds
      (goal_weight at the minimum 0.1).

    The rollout figure (`docs/results/navigation_compare_warm.png`) shows it circling a crossing
    pedestrian, driving around the outside of the narrow-passage walls, and looping in dense flow.
    The objective charged only 0.3 x time for such detours and nothing for jerky commands.

    Changes:
    - New metric `path_efficiency` (straight-line distance / path driven; 0 on failure).
    - Objective v2 adds `detour: 1.0` x (1 - path_efficiency) and `smoothness: 0.1` x
      command_smoothness. The defaults in code stay 0, so old configs reproduce old numbers.
    - goal_weight's lower bound is raised 0.1 -> 0.5.

    The warm start is rerun under v2 (`cmaes_stars_best_warm_v2`). The 0.707 policy is kept for
    reference but is not the recommended policy.

46. **The forecaster drives the crowd: `stars_crowd` pedestrian model.**
    - Each pedestrian follows its reference for its first 3.2 s (to build history). After that
      it moves by joint STARS forecasts every 0.4 s.
    - CVAE intents are seeded per pedestrian, so the robot-absent counterfactual uses the same
      draws.
    - The robot is added to the graph as one more person (STEPS has no robot). Pedestrians cannot
      interpenetrate it.

    Tested (`tests/test_seanjax.py`). Limitations:
    - Pedestrians ignore walls: the forecaster has no map, and recorded scenes have only an
      artificial arena.
    - Long rollouts make smooth sweeping arcs that drift from the recorded paths (mean deviation
      0.67 m over 30 s).
    - The 3.4 m/s maximum speeds seen in these runs come from the recorded tracks during the
      warm-up (they appear in plain replay too), not from the model.

    `cmaes_crowd.yaml` tunes the navigation policy in this STARS-driven crowd (objective v2, warm
    start from v1). The same forecaster also supplies the policy's predictions.
    The git index lock seen again at 02:08 came from search workers started before the lock-free
    fix (entry after 39); removed as stale (>60 s old, no local git process).

47. **Ablation: closed-loop (multi-step) training does not help on STEPS.** Low-KL fine-tuned CVAE
    with vs without `--rollout-exec-steps 4`, test split:

    | | Teacher-forced only | Plus closed-loop loss |
    | --- | --- | --- |
    | Agent ADE | 0.080 m | 0.080 m |
    | Moving ADE | 0.336 m | 0.345 m |
    | Joint ADE | 0.097 m | 0.103 m |
    | Closed-loop ADE at 1 / 4 / 12 executed steps | 0.084 / 0.081 / 0.080 | 0.085 / 0.081 / 0.080 |
    | 20 s simulated speed (smoothed real 0.115 m/s) | 0.115 m/s | 0.098 m/s |

    The teacher-forced model is already stable in closed loop; the multi-step loss costs ~3x
    compute and slightly hurts. It stays available as a flag but is not used by default.

48. **Which forecaster to use.** The best Optuna trial (53) trained at full length:
    - Test, seeds 42/1: agent ADE 0.079/0.079, moving ADE 0.332/0.334, joint 0.097/0.098.
    - Low-KL fine-tuned CVAE: 0.080 / 0.336 / 0.097.

    Trial 53 is marginally more accurate (within seed noise), but its 20 s simulations are ~25%
    too slow (0.084-0.087 m/s vs 0.115 smoothed real). The fine-tuned CVAE matches real speed.
    Decision: `cvae_kl01_finetune` stays the forecaster for the crowd model and the navigation
    predictor. Trial 53 is reported as the most accurate pure forecaster
    (`docs/results/forecasts_cvae_hpo53.png`). Accuracy-only hyperparameter search can trade
    away simulation realism; a realism term in the search objective is a follow-up.

49. **Disruption in a learned crowd must be measured locally.** CMA-ES in the STARS-driven crowd
    (first attempt, job 65680, cancelled) scored around -0.7 to -0.9:
    - `disruption_displacement` was ~1.13 m; reactive replay gives ~0.2 m.
    - The cause is chaos, not the robot: the robot node slightly changes every nearby
      pedestrian's forecast, and the closed-loop learned dynamics amplify such differences over
      30 s. Pedestrians that never approached the robot end up elsewhere than in the
      counterfactual.

    New metric `disruption_local`: matched displacement only while a pedestrian is within 3 m of
    the robot, averaged over pedestrians that came that close. `cmaes_crowd.yaml` now uses it as
    its disruption term and the run restarted. The global metric is still reported. For the
    social-force-type models (damped dynamics) the global metric remains the default.

50. **Objective v2 warm start: natural behaviour, a real trade-off.** Scripted test, best STARS
    predictor, fixed controller, objective v2:

    | Policy | Objective v2 | Success | Path efficiency | Disruption | Min clearance | Time | Smoothness |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Default | 0.425 | 100% | 0.98 | 0.206 m | — | 13.6 s | 0.62 |
    | v1 | 0.354 | 91% | 0.85 | 0.067 m | 0.74 m | 13.5 s | — |
    | Reward-hacking warm start | 0.173 | — | — | — | — | — | — |
    | Warm start v2 | 0.449 | 96% | 0.91 | 0.115 m | 0.46 m | 12.4 s | 0.37 |

    The rollout figure (`docs/results/navigation_compare_warm_v2.png`) shows clean narrow-passage
    and head-on behaviour and no circling around crossing pedestrians. A small loop near a
    crowded goal in dense flow remains.

    v2 is more efficient and smoother than v1 but less cautious; which is preferable depends on
    the objective weights, which remain a research choice. Recommended:
    `checkpoints/seanjax/cmaes_stars_best_warm_v2/best_policy.yaml` for objective v2, v1 for
    minimum disruption.

    Side fix: the social-force sidestep now applies only to walking pedestrians. Standing-group
    members orbited in place before; now they settle within ~5 s (0 m/s after 15 s). Results
    above used the old pedestrian model; the difference is confined to standing pedestrians.

51. **Measuring disruption in the STARS crowd, and a limitation it revealed.**
    - The local metric (49) still read ~0.85 m: pedestrians that reach the robot have already
      diverged chaotically.
    - New `robot_influence`: at each re-forecast the crowd model also forecasts without the robot
      in the graph, from the same histories. The metric is the mean distance between the two 4.8 s
      forecasts. It is attributable to the robot and immune to divergence. Also
      `robot_influence_max` and `peds_influenced` (> 5 cm).

    Finding: the STEPS-trained forecaster barely reacts to the robot node. Mean influence is 7 mm
    within 2 m of the robot and 1-2 mm beyond. STEPS contains no robot, and most of its
    interactions are weak (90% of pedestrians stand). In this crowd, avoidance is therefore almost
    entirely the robot's responsibility, plus the no-interpenetration rule. To make learned crowds
    reactive, the forecaster would need interaction-rich data or robot-in-the-loop data; this is
    a limitation to report, not a bug.

    For tuning: `cmaes_crowd.yaml` uses `robot_influence` with weight 10 (it is ~100x smaller than
    trajectory displacements) and now warm-starts from the objective-v2 policy. Restarted
    (previous run cancelled at generation 54).

52. **Correction to 51: the influence weight.** I set `disruption: 10` from three recorded scenes,
    where influence was ~0.005 m. Over the full training mix (scripted encounters included)
    `robot_influence` averages 0.24 m at the start, the same magnitude as trajectory disruption
    with the other crowd models. Weight 10 made the influence term dominate the objective
    (~-1.8), so the run was cancelled at generation 42 and restarted with weight 1.0.

    The "barely reacts" finding in 51 holds for recorded STEPS scenes. In scripted head-on and
    crossing encounters the forecaster does change its forecasts by tens of centimetres when the
    robot is present.

53. **Pretraining on all six datasets does not transfer better to STEPS.** A low-KL CVAE
    fine-tuned on STEPS, STEPS test:
    - from the all-datasets phase-1 encoder: agent ADE 0.081, moving 0.339, joint 0.096;
    - from the STEPS-only encoder: 0.080 / 0.336 / 0.097.

    Within seed noise. On STEPS, extra pedestrian datasets in self-supervised pretraining add
    nothing measurable. A single forecaster trained on all six datasets matches the STEPS-only
    models on STEPS validation (0.068-0.070) while also covering MADRAS, Grand Central, UCY and
    PTDPES. Per-dataset test results are pending (job 65718).

54. **Navigation in the STARS-driven crowd (test, scripted + recorded scenes; objective v2 with
    robot_influence).**

    | Policy | Objective | Success | Collisions | Path efficiency |
    | --- | --- | --- | --- | --- |
    | Goal-seeking | -2.326 | 34% | 66% | — |
    | Default social | -0.239 | 82% | 7.8% | — |
    | v1 | -0.150 | 83% | 3.3% | — |
    | Objective-v2 policy | 0.044 | 91% | 3.3% | 0.86 |
    | Crowd-tuned | 0.043 | 91% | 3.3% | 0.85 |

    - Tuning inside the learned crowd gives nothing over the v2 policy, which transfers as is.
    - On recorded STEPS scenes only, success is 30-50% for every social policy, against 100% with
      reactive-replay pedestrians. Collisions are ~0, so the failures are timeouts. The learned
      crowd barely yields to the robot (51), so in dense recorded crowds the robot rarely finds
      a gap within 30 s: a freezing-robot outcome.

    This is the main limitation of using the STEPS forecaster as a reactive crowd simulator.
    Options, not yet done:
    - longer time limits for recorded scenes;
    - an interaction-aware forecaster trained on data with robots;
    - a hybrid crowd (reactive replay near the robot, STARS elsewhere).

    The global `disruption_displacement` column (1.1-1.3 m) is meaningless in this crowd (49).

55. **Two fixes from the STARS-crowd figure (`docs/results/navigation_compare_crowd.png`).**
    - **Spiralling pedestrians.** A fixed intent draw is read in each pedestrian's re-centred
      local frame at every 0.4 s re-forecast, so a slight turn repeats into a circle. Intents now
      follow an AR(1) process (`intent_correlation: 0.8` per re-forecast, ~2 s correlation),
      still seeded and on a fixed schedule so the counterfactual matches. In `group_toward` with
      the robot present, mean net turning per pedestrian fell from 1.41 to 0.67 rad (max 2.7 ->
      1.1 rad).
    - **Unreachable recorded goals.** Several recorded-scene goals sat next to standing
      pedestrians, who do not yield in the learned crowd. Recorded scenarios now reject goals
      within 1 m of any pedestrian position in the window. This changes which recorded scenes are
      drawn, so recorded-scene numbers before this entry are not directly comparable.

56. **After the fixes in 55, tuning inside the STARS crowd pays off.** Re-evaluated on 90 test
    scenarios (80 scripted + 10 recorded STEPS scenes) with `stars_crowd` pedestrians
    (`runs/seanjax/crowd_v2`), the crowd-tuned policy re-tuned on the corrected crowd is now best:

    | Policy | Objective | Success | Collision | Path eff. | Recorded success / objective |
    | --- | --- | --- | --- | --- | --- |
    | Goal-seeking | -2.260 | 37% | 63% | 0.37 | 10% / -3.23 |
    | Default social | -0.187 | 84% | 7.8% | 0.83 | 70% / -0.38 |
    | v1 | 0.028 | 90% | 1.1% | 0.84 | 70% / -0.26 |
    | Objective-v2 policy | 0.144 | 96% | 3.3% | 0.91 | 90% / 0.20 |
    | **Crowd-tuned** (`cmaes_crowd`) | **0.233** | **99%** | **1.1%** | 0.91 | **100% / 0.37** |

    - The freezing-robot failure on recorded scenes from 54 is gone: it came largely from the
      spiralling pedestrians and goals beside standing people, not from the policy.
    - The gain over the v2 policy is mostly fewer collisions (3.3% -> 1.1%) at the cost of 0.5 s
      longer trips. `robot_influence` is about the same for all social policies (0.28-0.30 m).
    - Figure: `docs/results/navigation_compare_crowd_v2.png`. In `dense_flow` all policies still
      make a small loop near the goal while waiting for the cross flow; left as is, since it is
      collision-free and the objective already charges for the detour.

57. **Splitting the all-dataset test evaluation (job 65718 at risk of its 4 h limit).** The STEPS
    test evaluation took 12 min. The full six-dataset test set is about 13x larger (Grand Central
    alone is 441 MB of windows), and closed-loop evaluation at exec 1/4/12 with K = 20 runs about
    320 forward passes per batch. The evaluator writes nothing until the end, so a timeout would
    lose everything. Instead of cancelling 65718, job 65741 runs a lighter version next to it:
    teacher-forced metrics plus closed-loop at exec 4 only, with Grand Central on its own GPU.
    Outputs: `eval_test_gc.json` and `eval_test_rest.json` per all-dataset model. Whichever
    finishes first is reported.

58. **Crowd tuning v2: intimate-zone comfort term, larger validation set.** Two problems showed up
    in the first crowd-tuning run (`checkpoints/seanjax/cmaes_crowd/convergence.png`):
    - **The comfort term does not separate policies.** In the test set, the fraction of time a
      pedestrian is within 1.2 m is 0.35-0.40 for every policy, goal-seeking included, because
      dense scenes make that unavoidable. CMA-ES therefore drove `personal_space_radius` to its
      lower bound (0.3 m). The 0.45 m intimate-zone rate does separate them: 13% for the
      crowd-tuned policy against 0.8% for v1. `cmaes_crowd_v2.yaml` uses
      `intimate_violation_rate` with weight 1.0.
    - **Selection noise.** Validation J moved by about ±0.05 between generations with 12
      scenarios per kind, which is as large as the differences between policies. Validation now
      uses 24 per kind.

    Job `cma_crowd2` runs three searches:
    - warm start from `cmaes_crowd`, seed 42;
    - the same with seed 7, to measure how much the result depends on the search seed;
    - warm start from the cautious v1 policy.

    All policies will then be re-evaluated on the test set under the v2 objective. The v1 tables
    are left as they are.

59. **All-dataset models on the test split** (`docs/results/stars_ma_all_test.md`).
    - The from-scratch model with trial-53 settings beats the SSL-pretrained, fine-tuned one on
      every dataset, by 0.002-0.010 m of ADE. Pretraining pays off on STEPS alone (46) but not with
      these settings on the full mixture, so `cvae_hpo53_scratch_all` is the recommended
      all-dataset model.
    - It is 2-3.5x better than constant velocity everywhere. On STEPS it scores 0.078, level
      with the STEPS-only models (0.079-0.080) within about twice the seed noise.
    - **Compounding error.** Re-forecasting every model step (exec 1) from the model's own
      predictions costs little on STEPS (0.078 -> 0.081) but 14-20% on PTDPES, Madras, UCY and
      ETH (e.g. ETH 0.234 -> 0.280). The STARS crowd runs at exec 1, so it should use the
      STEPS model for STEPS scenes. A crowd for other datasets would need closed-loop
      fine-tuning, which did not help on STEPS (47) but may on these datasets.
    - Job 65718 finished just inside its limit, so the backup from 57 was not needed. Its
      exec-4 numbers agree with 65718's to within 0.003 m.

60. **Annealing and closed-loop fine-tuning for the all-dataset model.** `cvae_hpo53_scratch_all`
    early-stopped at epoch 53 (best 33). Its cosine schedule was set for 200 epochs, so the LR
    was still 0.83e-3 of a 0.96e-3 peak when it stopped: the model was never annealed.
    Separately, closed-loop error at exec 1 is 14-20% above teacher-forced outside STEPS (59).
    - **New option.** `--init-checkpoint` starts a forecast run from a forecast checkpoint (all
      weights and recipes; KL warmup off, since the latent is already trained).
    - **Job 65758** runs four 20-epoch cosine fine-tunes on all six datasets:
      - LR 3e-4, teacher-forced only (does annealing alone help?);
      - LR 3e-4 plus the closed-loop loss at exec 1 (the simulator's setting);
      - LR 3e-4 plus the closed-loop loss at exec 4;
      - LR 1e-4 plus the closed-loop loss at exec 1.

      Run directories are `experiments/stars_ma/runs/cvae_hpo53_all_anneal*`.
    - **How it will be judged:** per-dataset test ADE, teacher-forced and at exec 1, against the
      table in `stars_ma_all_test.md`. Closed-loop training did not help on STEPS (47), where
      compounding error was already small, so it gets a fair test only on these datasets.

61. **The scratch Optuna search has converged** (`docs/results/hpo_steps_cvae_scratch.md`). After 475
    trials (23 complete, 452 pruned by the median pruner), trial 53 is still the best (0.0798 m
    validation ADE). The top eight trials are within 0.0005 m of each other, which is below seed
    noise (about 0.0007 m, decision 44). KL weight (52%) and encoder dropout (22%) explain most
    of the variance. No further search on STEPS with this search space: remaining gains have to
    come from data, schedule (60) or architecture, not from these hyperparameters.

62. **The SSL fine-tuning Optuna search has ended** (`docs/results/hpo_steps_cvae_ssl.md`). 213 trials
    ran (82 complete, 131 pruned). The best is trial 165 at 0.0792 m validation ADE: fine-tuned
    encoder, KL 0.061, LR 2.4e-3, encoder LR ratio 0.19. The top eight trials are all fine-tuned
    rather than frozen. Against the best scratch trial (0.0798, decision 61) the difference is
    0.0006 m, about one seed standard deviation, so pretraining on STEPS remains a small effect.
    The importances repeat the scratch study: KL weight 56%, warmup 18%. Trial 165 is not
    retrained at full length: the expected gain is within seed noise, and the GPUs are better
    spent on the annealing and closed-loop runs (60).

63. **Exec-1 closed-loop fine-tunes overrun their time limit.** Training with the exec-1
    closed-loop loss takes about 26 min per epoch, against 1.7 min teacher-forced and 6.8 min at
    exec 4. The 20 epochs (about 9 h) will not fit in job 65758's 6 h limit. The runs are not
    restarted with fewer epochs, because that would change the cosine schedule mid-experiment.
    Instead they are resumed with `--resume` after the timeout; `latest.pt` is saved every epoch.
    The crowd v2 test evaluation and figures are queued behind the tuning job (dependency on
    65742) and written to `runs/seanjax/crowd_v3/` and `docs/results/navigation_compare_crowd_v3.png`.

64. **Crowd tuning v2 results, and a wall bug in the STARS crowd.**
    Test results under the v2 objective (intimate-zone comfort), from `runs/seanjax/crowd_v3`:

    | Policy | Objective | Success | Collision | Intimate-zone rate | Min gap (m) | Recorded success |
    | --- | --- | --- | --- | --- | --- | --- |
    | Goal-seeking | -2.263 | 36% | 64% | 13.3% | 0.16 | 10% |
    | Default social | -0.208 | 83% | 8.9% | 17.5% | 0.24 | 70% |
    | v1 (cautious) | 0.198 | 90% | 1.1% | 0.8% | 0.64 | 70% |
    | Objective-v2 policy | 0.238 | 97% | 2.2% | 14.6% | 0.32 | 90% |
    | Crowd-tuned, first run (`cmaes_crowd`) | 0.258 | 98% | 2.2% | 12.8% | 0.36 | 100% |
    | **Crowd v2, seed 42** (`cmaes_crowd_v2`) | **0.281** | 96% | 2.2% | 5.3% | 0.47 | 80% |
    | Crowd v2, seed 7 | 0.250 | 94% | 2.2% | 3.6% | 0.51 | 70% |
    | Crowd v2 from v1 | 0.237 | 92% | 3.3% | 2.2% | 0.52 | 60% |

    - The new comfort term works: intimate-zone time fell from 13% to 2-5% and the minimum gap
      grew. Only seed 42 beats the first crowd-tuned policy overall. The spread between seeds and
      starts (0.237-0.281) is as large as the gains, so tuning results are reliable to about
      ±0.02.
    - Recorded-scene success (10 scenes) varies from 60% to 100% between runs. That is too few
      scenes to rank policies on.
    - **Wall bug.** `navigation_compare_crowd_v3.png` shows forecaster-driven pedestrians walking
      through walls in `narrow_passage` and out of the `dense_flow` corridor: the forecaster has
      no notion of obstacles. Fix: a model-driven move that would cross a wall now slides along
      it, or stops at a corner (`geometry.block_wall_crossings`). Recorded scenes' arena box is
      not a physical wall (recorded people cross it), so it is exempt
      (`Scenario.meta["pedestrian_walls"] = False`, `ScenarioBatch.ped_walls`).
    - All STARS-crowd numbers above were simulated with the bug, in the scenes that have walls.
      They are re-evaluated in the fixed crowd as `runs/seanjax/crowd_v4`, and the v2 tuning is
      rerun there, warm-started from `cmaes_crowd_v2`.

65. **Wall-respecting crowd: re-evaluation, and unstable exec-1 closed-loop training.**
    - **Navigation** (`runs/seanjax/crowd_v4`, 270 test scenarios, 30 per kind, objective of 58):

      | Policy | Objective | Success | Collision | Intimate-zone rate | Min gap (m) | Recorded success (30) |
      | --- | --- | --- | --- | --- | --- | --- |
      | Goal-seeking | -2.149 | 38% | 62% | 11.4% | 0.21 | 27% |
      | Default social | -0.167 | 84% | 7.0% | 16.6% | 0.30 | 77% |
      | Objective-v2 policy | 0.128 | 93% | 4.4% | 13.6% | 0.37 | 87% |
      | v1 | 0.209 | 90% | 0.7% | 1.3% | 0.68 | 67% |
      | Crowd-tuned, first run | 0.251 | 96% | 0.7% | 13.3% | 0.42 | 83% |
      | Crowd v2, seed 7 | 0.291 | 95% | 1.1% | 3.8% | 0.56 | 73% |
      | Crowd v2, seed 42 | 0.303 | 96% | 1.5% | 5.4% | 0.51 | 90% |
      | Crowd v2 from v1 | 0.311 | 95% | 0.7% | 2.3% | 0.57 | 73% |

      With 3x the test scenes, all three v2-tuned policies beat every earlier one by 0.04-0.06.
      Seed 42 is kept as the reference policy: it is within 0.01 of the best and has the best
      recorded-scene success. In `navigation_compare_crowd_v4.png` pedestrians now stay in the
      corridor and out of the walls.
    - **Known issue.** In `dense_flow` and `narrow_passage` every tuned policy waits by circling
      instead of stopping. The detour term charges for it, but not enough to remove it. Not yet
      addressed: a stop-and-wait candidate or a heading-reversal penalty would be the next
      thing to try.
    - **Exec-1 closed-loop fine-tuning diverges** (60). At LR 3e-4 the KL term exploded to about
      1e6 within one epoch (validation ADE 1.27). At 1e-4, validation rollout ADE was 0.68 after
      one epoch, against about 0.27 before fine-tuning. Backpropagating through 12 re-encoding
      rounds is unstable here. Both runs were stopped. Exec 4 trains normally (validation 0.2176
      at epoch 3) and continues.
    - **Operational mistake.** I stopped the two exec-1 runs with `pkill -f <run-dir>`. The
      pattern also matched the job's batch script, which carries every command, so it ended the
      whole job, including the healthy exec-4 run. That run was resumed from `latest.pt` as job
      65775. In future, kill by PID after checking with `pgrep -a`.

66. **No-loop controller option.** Tuned policies circle while waiting (65). The retreat cost only
    compares the distance to the goal at the end of the plan (about 2.2 s for tuned policies)
    with the current distance. A loop that comes back by then is free, and CMA-ES drove
    `turn_penalty` to 0. New controller option `retreat_along_path` (default off, so earlier
    results are unchanged) charges retreat at the worst point along the plan. Job `cma_noloop`:
    - evaluates `cmaes_crowd_v2` with the option on and no re-tuning
      (`runs/seanjax/crowd_v4_noloop`), to compare against its 0.303 in 65;
    - re-tunes from it with the option on (`cmaes_crowd_v3_noloop.yaml`).

    The option becomes the default if it removes the loops without lowering the objective.

67. **No-loop option, first result.** `cmaes_crowd_v2` with `retreat_along_path` on and no
    re-tuning: objective 0.305 against 0.303, and dense-flow path efficiency unchanged at 0.89.
    The end-of-horizon retreat check is therefore not what produces the loops; they come from
    re-planning (each 0.1 s plan is acceptable, but successive plans curve the same way). The
    re-tune with the option on (`cmaes_crowd_v3_noloop`) is still running. If it does not help
    either, the next candidate is a hysteresis or commitment term on the turn direction.
    **Side note.** Crossing scenes score negative for every policy (v2: -0.30) only because
    `robot_influence` is about 0.9 m there. A single pedestrian crossing the robot's path must
    change course whatever the robot does. It is the same for all policies, so it does not affect
    the ranking.

68. **Why tuned policies circle: they choose to, standing still is allowed.**
    `seanjax/scripts/diagnose_waiting.py` (`runs/seanjax/diagnose_waiting/`, 20 test scenes per
    kind) logs whether the stop command `(v=0, w=0)` passed the hard safety check at each step.
    - Stopping is allowed 91-99% of the time for every policy, so the circling is not forced
      evasion. (A first run reported 100% "ruled out": the turn-rate grid's middle value is
      -6e-8, not 0, so the stop candidate was not found. Fixed.)
    - In `dense_flow`, the tuned policies turn on 22-24% of the steps where stopping is allowed,
      against 6% for the default controller. Net heading change per episode is 2.6-2.7 rad,
      against 0.9.
    - **Likely cause: the fixed speed cost.** `speed_cost` (0.5, not tuned) charges
      `((v - preferred_speed) / v_max)^2`, and it fades only when a pedestrian is within twice the
      personal radius. With preferred speed about 1.0 m/s, stopping costs about 0.5 whenever the
      blocking pedestrian is more than about 1 m away, while an arc with no net progress costs
      little.
    - Job 65801 tests the same policy, without re-tuning, at `speed_cost` 0.2 and 0.0, with the
      waiting diagnostic and a figure (`runs/seanjax/crowd_v4_speed*`). The goal term already
      rewards progress, so a lower speed cost should not slow free-space driving much.

69. **The speed cost is not the cause of circling** (job 65801, `runs/seanjax/crowd_v4_speed*`).
    `cmaes_crowd_v2` without re-tuning:

    | `speed_cost` | Objective | Success | Dense-flow path eff. | Recorded success | Turning while stop allowed (dense flow) |
    | --- | --- | --- | --- | --- | --- |
    | 0.5 (tuned with) | 0.303 | 96% | 0.89 | 90% | 24% |
    | 0.2 | 0.250 | 94% | 0.85 | 70% | — |
    | 0.0 | 0.187 | 90% | 0.81 | 63% | 20% |

    A lower speed cost makes every metric worse and barely changes the turning. Decision 68's
    hypothesis is rejected; the other policy parameters were tuned with 0.5, so this is not a
    fair test of a lower value, only evidence that it is not the main driver.

    **What changes next.** I am stopping hand-diagnosis of the controller. It has three
    hand-set terms that shape waiting behaviour: `speed_cost`, `retreat_cost` and the goal-
    alignment weight. The principled fix is to let CMA-ES tune them along with the rest, and to
    add a circling penalty to the objective so that the evaluation charges for loops directly
    (net heading change per metre travelled). Both are queued behind the current no-loop
    re-tune result.
