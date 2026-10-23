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

70. **Crowd v4: tune the waiting costs, charge for loops.** Following 69:
    - **New policy parameters,** all defaulting to the old fixed behaviour, so existing policies
      are unchanged:
      - `speed_scale` multiplies `speed_cost` (0.1-5, log scale);
      - `retreat_scale` multiplies `retreat_cost` (0.1-10, log scale);
      - `alignment_weight` replaces the fixed 0.1 on goal alignment (0-1).
    - **New metrics:** `heading_change`, the total absolute heading change in rad, and `loops`,
      the full turns beyond half a turn per episode.
    - **New objective term:** `circling`, weight 0.2 per loop in `cmaes_crowd_v4.yaml`.
    - **Job `cma_crowd4`:** three seeds (42, 7, 3), warm-started from `cmaes_crowd_v2`. The test
      evaluation and figures under the v4 objective are queued behind it
      (`runs/seanjax/crowd_v6`, `docs/results/navigation_compare_crowd_v6.png`). The v4
      objective differs from v3's, so all policies are re-scored under it there.

71. **Annealing alone barely helps** (`cvae_hpo53_all_anneal`, test split, against
    `cvae_hpo53_scratch_all`):

    | Dataset | Agent ADE | Moving ADE | Closed-loop exec 1 ADE |
    | --- | --- | --- | --- |
    | STEPS | 0.078 -> 0.078 | 0.332 -> 0.327 | 0.081 -> 0.083 |
    | PTDPES | 0.284 -> 0.282 | 0.405 -> 0.400 | 0.325 -> 0.317 |
    | Madras | 0.209 -> 0.207 | 0.231 -> 0.229 | 0.245 -> 0.235 |
    | Grand Central | 0.244 -> 0.241 | — | 0.261 -> 0.259 |
    | UCY | 0.193 -> 0.194 | 0.304 -> 0.308 | 0.239 -> 0.250 |
    | ETH | 0.234 -> 0.229 | 0.239 -> 0.234 | 0.280 -> 0.294 |

    Teacher-forced error drops by at most 2%, and closed-loop error moves in both directions.
    The early-stopped model was nearly converged. `cvae_hpo53_scratch_all` stays the
    all-dataset reference unless the exec-4 fine-tune (65775) does better in closed loop.

72. **Re-tuning in the wall-respecting crowd** (`runs/seanjax/crowd_v5`, 270 test scenes, objective
    of 58):

    | Policy | Objective | Success | Collision | Intimate-zone rate | Path eff. | Loops / episode | Recorded success |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Default social | -0.178 | 83% | 7.4% | 16.4% | 0.82 | 0.02 | 77% |
    | Crowd v2 (tuned before the wall fix) | 0.297 | 96% | 1.5% | 5.4% | 0.90 | 0.11 | 87% |
    | v3 from crowd v1 | 0.314 | 96% | 1.1% | 2.6% | 0.89 | 0.12 | 80% |
    | v3 seed 7 | 0.321 | 96% | 1.1% | 2.6% | 0.88 | 0.14 | 77% |
    | v3 with no-loop controller (`crowd_v5_noloop`) | 0.322 | 95% | 0.7% | 2.9% | 0.88 | 0.13 | 73% |
    | **v3 seed 42** (`cmaes_crowd_v3`) | **0.347** | 96% | **0.7%** | **1.3%** | 0.88 | 0.18 | 77% |

    - Tuning in the corrected crowd gives +0.02 to +0.05. The gain comes from comfort: intimate-
      zone time halves again, to 1.3%.
    - Every tuned policy loops more than the default controller (0.11-0.18 extra full turns per
      episode, against 0.02), and v3 seed 42 loops most. The no-loop controller option does not
      change that even after re-tuning, so it stays off by default.
    - **Evaluation noise.** The same policy on the same scenes scores 0.297-0.304 across
      evaluation runs (the default controller: -0.167 to -0.201). The forecaster crowd runs in
      bf16 on the GPU and is not bit-for-bit reproducible, and small differences grow in closed
      loop. Differences under about 0.02 are not meaningful.
    - `cmaes_crowd_v3` becomes the reference policy unless the loop-penalised v4 tuning (70)
      matches it with fewer loops.

73. **Closed-loop fine-tuning closed out; a non-finite-loss guard.**
    - **The exec-4 fine-tune went NaN at epoch 5**, the first epoch after the resume in 65, and
      ran 15 more epochs on NaN weights. Its best checkpoint (epoch 3, validation 0.2176) is
      level with annealing alone (0.2181, decision 71). Combined with 47 and 65 (exec 1
      diverges), closed-loop training gives no gain for this model on any dataset mix tried. It
      stays available as a flag but is dropped from the plan. Compounding error outside STEPS
      (7-24% at exec 1) is accepted. The simulator uses the STEPS model on STEPS scenes, where
      it is 4%.
    - **Guard.** `train_stars_ma` now skips any update whose (post-all-reduce) gradient norm is
      non-finite, counts skipped steps per epoch (`train_skipped_steps`), and stops after
      `MAX_NONFINITE_STREAK` (100) in a row. The norm is identical on every DDP rank, so all
      ranks skip together.
    - `cvae_hpo53_scratch_all` stays the all-dataset reference model.

74. **Single-dataset models with the trial-53 recipe** (job 65888). Every per-dataset number so
    far comes from the all-dataset model, so nothing yet says whether mixing datasets helps or
    hurts each one. One model each for PTDPES, Madras, Grand Central, UCY and ETH
    (`experiments/stars_ma/runs/cvae_hpo53_scratch_<dataset>`) is trained with exactly the
    all-dataset model's settings, then evaluated on its test split (teacher-forced and exec 1).
    STEPS already has its own (`cvae_hpo53_scratch_seed42`).
    - **ETH has no validation split** (it is also missing from the all-dataset model's
      validation metrics), so the ETH model selects its checkpoint on UCY validation, the
      closest related data.
    - **Housekeeping.** The two exec-1 run directories from 60 still say `"status": "running"`;
      they were stopped by hand (65) and are not resumable results.

75. **Crowd v4 (tuned waiting costs, loop penalty) is the new reference policy**
    (`checkpoints/seanjax/cmaes_crowd_v4/best_policy.yaml`).
    270 test scenes under the v4 objective, which charges 0.2 per loop (`runs/seanjax/crowd_v6`
    and `crowd_v6_vs_v3`):

    | Policy | Objective | Success | Collision | Intimate-zone rate | Path eff. | Loops / episode | Recorded success (30) |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Default social | -0.196 | 83% | 7.8% | 16.4% | 0.82 | 0.01 | 77% |
    | v1 | 0.196 | 91% | 0.7% | 1.3% | 0.85 | 0.09 | 67% |
    | Crowd v2 | 0.276 | 96% | 1.5% | 5.4% | 0.90 | 0.10 | 87% |
    | Crowd v3 (previous reference) | 0.321 | 97% | 0.4% | 1.4% | 0.89 | 0.19 | 77% |
    | v4 seed 3 | 0.340 | 98% | 1.1% | 3.0% | 0.89 | 0.12 | 90% |
    | v4 seed 7 | 0.367 | 99% | 0.4% | 5.2% | 0.91 | 0.11 | 97% |
    | **v4 seed 42** | **0.375-0.388** | **98-99%** | 0.7-1.1% | 2.6% | 0.90 | **0.08** | **100%** |

    (Two evaluation runs of v4 seed 42; the spread is evaluation noise, 72.)
    - **The gain over v3** (+0.07) is mostly recorded-scene success (77% to 100%) and fewer
      loops (0.19 to 0.08). v3 is slightly more comfortable (1.4% against 2.6% intimate-zone
      time); with the loop penalty taken out, v4 still leads by about 0.03.
    - **What the tuner chose.** All three seeds pushed `speed_scale` to 2.7-5 (the upper bound
      for seed 42) and `alignment_weight` to 0-0.07. A strong pull toward cruising speed and no
      reward for facing the goal mean fewer slow pivots. This is the opposite direction from
      the hand experiment in 69. The bound of 5 is hit, so a wider range is a candidate for the
      next run.
    - **Loops.** `navigation_compare_crowd_v6.png`: the narrow-passage loop is gone (11.3 s
      against 16.6 s), but dense flow still loops (0.16 per episode). The circling diagnostic
      still shows turning on 23% of the steps where stopping was allowed. Accepted for now.

76. **Crowd v5: wider `speed_scale` range.** v4 seed 42 ended at the old upper bound (5), so the
    bound is now 20 and two seeds re-tune from v4 (`cmaes_crowd_v5.yaml`, job 65938). Existing
    policies are unaffected; their values lie inside the new range. The same job renders v4
    rollout GIFs (four recorded scenes plus dense flow, narrow passage and group) to
    `runs/seanjax/crowd_v6/visualize/`.
    The single-dataset job (74) was lost to a node failure (`NODE_FAIL`, job 65888). UCY, ETH and
    Madras had finished. PTDPES resumes from `latest.pt` (epoch 14) and the Grand Central test
    evaluation is rerun (job 65932).
    Update: job 65938 was also lost to `NODE_FAIL` at generation 88-89. Both v5 searches were
    resumed from `checkpoint.pkl` (job 65986). The test evaluation and comparison figure
    (`runs/seanjax/crowd_v7`, `docs/results/navigation_compare_crowd_v7.png`) are queued with
    `afterok` (job 65987). Node failures are recurring, so every long job here keeps
    resumable checkpoints.

77. **Crowd v5 does not beat v4; v4 stays the reference.** Test results (`runs/seanjax/crowd_v7`,
    v4 objective):

    | Policy | Objective | Success | Collision | Intimate-zone rate | Loops / episode (dense flow) | Recorded success |
    | --- | --- | --- | --- | --- | --- | --- |
    | v4 | 0.388 | 99% | 0.7% | 2.6% | 0.08 (0.17) | 100% |
    | v5 seed 42 (= v4: the search never improved on its start) | 0.375 | 98% | 1.1% | 2.5% | 0.08 (0.18) | 100% |
    | v5 seed 7 (`speed_scale` 10.7, `retreat_scale` 6.3) | 0.315 | 96% | 2.6% | 4.7% | 0.04 (0.11) | 87% |

    - The wider range lets seed 7 halve the loops, but collisions rise from 0.7% to 2.6%. Its
      validation lead (0.383 against 0.374) did not hold on test, which is selection on a noisy
      validation score.
    - v4 and its unchanged copy differ by 0.013, inside evaluation noise.
    - **Conclusion.** The controller's tunable range is now explored. Removing the remaining
      dense-flow loops costs safety with this controller. A structural change, such as a
      wait-in-place mode chosen when progress is blocked, is the next step if loops matter
      more than collisions. Stopping navigation tuning here.

78. **Structural fix for loops: penalise turning without progress.** Decision 77 found that the
    existing cost terms cannot remove dense-flow loops without giving up safety. A loop is an arc
    that turns while making no net progress. The new policy parameter `idle_turn_weight`
    (default 0, so existing policies are unchanged; range 0-5) adds
    `idle_turn_weight * (1 - clamp(progress, 0, 1)) * |w| / w_max` to each candidate's cost.
    Turning stays free when it buys progress, and waiting (v = 0, w ~ 0) becomes the cheaper
    option when the way is blocked. Job 66008:
    - v4 with `idle_turn_weight = 1` and no re-tuning, on the test set
      (`runs/seanjax/crowd_v8_idle1_untuned`), to see the term's direct effect;
    - three CMA-ES runs with the term free (`cmaes_crowd_v6.yaml`): from v4 with seeds 42 and 7,
      and from v4 with the term at 1.
    **Update: first form rejected.** The first version charged
    `idle * |w|`, any turning on arcs without progress. On v4 with weight 1 and no re-tuning it
    made loops worse: 0.51 per episode against 0.08, objective -0.145 against 0.380
    (`runs/seanjax/crowd_v8_idle_turn_all_untuned`). It also charges the turn back toward the
    goal, which a robot facing away needs, so it blocked recovery and produced wider loops. The
    tuning job was cancelled. The term now charges only the loss of goal alignment on arcs without
    progress: `idle * relu(alignment_now - alignment_end) / 2`. Turning back toward the goal is
    free. The untuned effect at weights 1 and 3 is checked first (job 66013,
    `runs/seanjax/crowd_v8_idle{1,3}`), before any re-tuning.

79. **The idle-turn term does not help; navigation controller work stops here.** Revised term
    (charging only turning away from the goal without progress) on v4 without re-tuning
    (`runs/seanjax/crowd_v8_idle{1,3}`):

    | `idle_turn_weight` | Objective | Success | Loops (dense flow) | Recorded success |
    | --- | --- | --- | --- | --- |
    | 0 (v4) | 0.388 | 99% | 0.08 (0.17) | 100% |
    | 1 | 0.256 | 94% | 0.08 (0.30) | 83% |
    | 3 | -1.560 | 9% | 0.07 (0.38) | 10% |

    - At weight 3 the robot mostly freezes. In dense flow every move turns it away from the goal
      for a moment, so waiting wins until the episode times out.
    - At weight 1, dense-flow loops increase. The re-planning at each step does not commit to a
      side, so penalising part of a loop only stretches it.
    - Re-tuning would most likely drive the weight back to 0, so it is not run.
    - The parameter stays in the code at default 0 (no effect) and is documented as not helpful.
    - **Conclusion across 68-79.** With a one-step trajectory-scoring controller, the remaining
      dense-flow loops (0.17 per episode) are the price of v4's 99% success and 0.7% collisions.
      Removing them needs a controller with memory, for example committing to a passing side for
      a few seconds. That is a new controller, not a cost term, and it is left as a recorded
      next step rather than started.
    - **Reference policy:** `checkpoints/seanjax/cmaes_crowd_v4/best_policy.yaml`.

80. **One model per dataset vs one model for all** (`docs/results/stars_ma_single_vs_all.png`).
    Same recipe (trial 53) for every model; test split, best-of-20:

    | Dataset | Training windows | Own model ADE / FDE | All-dataset model ADE / FDE | Exec-1 ADE, own / all | ADE change |
    | --- | --- | --- | --- | --- | --- |
    | STEPS | 76,675 | 0.079 / 0.125 | 0.078 / 0.123 | 0.083 / 0.081 | -1% |
    | PTDPES | 90,972 | 0.290 / 0.421 | 0.284 / 0.413 | 0.335 / 0.325 | -2% |
    | Grand Central | 4,781 | 0.250 / 0.412 | 0.244 / 0.403 | 0.297 / 0.261 | -3% |
    | Madras | 3,110 | 0.211 / 0.318 | 0.209 / 0.315 | 0.245 / 0.245 | -1% |
    | UCY | 1,097 | 0.262 / 0.521 | 0.193 / 0.383 | 0.310 / 0.239 | -26% |
    | ETH | 272 | 0.393 / 0.706 | 0.234 / 0.373 | 0.468 / 0.280 | -40% |

    - **Pooling never hurts.** The all-dataset model is better than or level with each dataset's
      own model on every dataset. The gain grows as the dataset shrinks: 1-3% for the large ones,
      26% for UCY and 40% for ETH (272 windows, validated on UCY).
    - **Closed loop gains more than teacher forcing on Grand Central:** exec-1 ADE 0.261 against
      0.297 (-12%). Training on other datasets makes rollouts more stable there.
    - **Recommendation.** Use `cvae_hpo53_scratch_all` for every dataset except STEPS, where the
      STEPS fine-tuned `cvae_kl01_finetune` stays the simulator model (its 20 s rollouts are
      validated for realism, 48). Per-dataset models are not worth keeping. This closes the
      "train on individual datasets and on all at once" task.

81. **A controller with memory: side commitment.** Following 79, the controller now keeps a little
    state across control steps. When the chosen command turns faster than `commit_min_turn`
    (0.3 rad/s), that turn direction is committed for `commit_steps` (20 steps, 2 s). Candidates
    that turn the other way within that window cost `side_commit_weight` more (new parameter,
    default 0, so existing policies are unchanged). The memory resets at the start of every
    rollout. The aim is to stop successive plans from alternating sides, which is how a dense-flow
    loop forms. Job 66087:
    - v4 with the weight at 0.5 and at 2, no re-tuning (`runs/seanjax/crowd_v9_commit*`), plus
      the waiting diagnostic at 2;
    - two CMA-ES runs with the weight free (`cmaes_crowd_v7.yaml`, seeds 42 and 7, starting from
      v4 with the weight at 0.5).

    Tuning runs alongside the untuned check this time because both fit on one node. If the
    untuned check is clearly harmful, the tuning result is not used.
    **Result: no gain, and the design was wrong.** On v4 without re-tuning
    (`runs/seanjax/crowd_v9_commit*`):

    | `side_commit_weight` | Objective | Collision | Loops (dense flow / narrow) |
    | --- | --- | --- | --- |
    | 0 (v4) | 0.382 | 0.7% | 0.08 (0.17 / 0.14) |
    | 0.5 | 0.384 | 0.4% | 0.10 (0.18 / 0.21) |
    | 2 | 0.375 | 0.4% | 0.11 (0.27 / 0.23) |

    The objective is unchanged and loops rise slightly with the weight. In hindsight this
    follows from the design: a loop *is* a sustained turn in one direction, so committing to the
    turn direction encourages it. The memory should hold the side on which the robot passes the
    blocking pedestrian (left or right of them, in the goal frame), not the sign of `w`. The
    v7 tuning runs already in flight are allowed to finish but are not expected to help.
    A pass-side memory is the corrected design. It needs the controller to identify the blocking
    pedestrian, which is a larger change, so it is recorded here and not started without review.
    **Tuning closes it out** (`runs/seanjax/crowd_v9`, 270 test scenes). CMA-ES drove
    `side_commit_weight` to 0 (seed 42) and 0.02 (seed 7). The tuned policies score 0.320 and
    0.327 against v4's 0.375 in the same run. They loop a little less (0.05-0.07) but collide
    2-2.5x more (2.2-2.6% against 1.1%), the same trade-off as v5 seed 7 in 77. v4 stays the
    reference. This ends the cost-term and turn-memory line of work (68-81); only the pass-side
    memory design (above) remains, pending review.

82. **Pass-side memory (the corrected design from 81).** New parameter `pass_side_weight`
    (default 0, so existing policies are unchanged). Each step:
    - The *blocker* is the nearest visible pedestrian ahead of the robot (toward the goal) within
      `pass_range` (2 m).
    - Each candidate plan ends either left or right of the blocker's predicted position, in the
      goal frame.
    - When a new blocker appears, the side of the chosen plan is committed for `commit_steps`
      (2 s). Plans that end on the other side of the same blocker cost `pass_side_weight` more.

    Unlike 81, continuing to turn is free, and only switching sides around the same person is
    charged. Job `eval_pass` runs v4 with the weight at 1 and at 3, no re-tuning
    (`runs/seanjax/crowd_v10_pass*`), plus the waiting diagnostic and a comparison figure at 3.

    **Seed replicates** (job 66145). The all-dataset model's lead over single-dataset models is
    1-3% on STEPS, PTDPES, Madras and Grand Central (80), and seed noise has only been measured on
    STEPS. Two more seeds of `cvae_hpo53_scratch_all`, plus one more each of the PTDPES and Grand
    Central single-dataset models, put error bars on those gaps.
    **Result: no effect.** v4 with `pass_side_weight` 1 and 3, no re-tuning
    (`runs/seanjax/crowd_v10_pass*`): objective 0.352 and 0.349 against 0.367 in the same run;
    loops 0.08 for all three; collisions unchanged. The comparison figure shows paths identical
    to v4. Not tuned. The parameter stays at its default 0.

83. **Per-episode outcomes depend on batch composition; aggregates are stable.** v4 collided on
    the seed-0 `dense_flow` and `narrow_passage` scenes in `runs/seanjax/crowd_v10_pass3/compare.png`.
    In `navigation_compare_crowd_v6.png` it reached the goal on the same scenes. Checks
    (`runs/seanjax/repro/`):
    - Two runs of the same three-scene figure on the current code agree (dense-flow collision at
      14.8 s against 14.6 s; narrow-passage collision at 6.9 s in both).
    - The full 270-scene test evaluation on the current code gives 0.387 objective, 98.5%
      success and 0.7% collisions: the same as v4's earlier results (0.375-0.388). The controller
      changes since 78 do not alter v4 when their weights are 0.
    - **Cause.** `compare_policies.py` simulates its scenes as one batch, and the earlier figure
      had five scenes against three now. Batch shape changes bf16 GPU reductions in the STARS
      crowd, and in close encounters that can flip an episode's outcome.
    - **Consequence.** Single-episode figures illustrate behaviour, not rates. Collision and
      success rates come only from the 270-scene evaluation. A deterministic crowd (fp32,
      `torch.use_deterministic_algorithms`) would make figures reproducible at some speed cost;
      not done.

84. **The all-dataset recipe is seed-sensitive; a 60-epoch schedule is tested.** Seed replicates
    of `cvae_hpo53_scratch_all` (job 66145) reach best validation ADE 0.247 (seed 1, epoch 12)
    and 0.251 (seed 2, epoch 8), against 0.220 for seed 42 (epoch 33). After that, seeds 1 and 2
    get worse while the LR is still near its peak (seed 1 rises to 0.290, and its KL grows). The
    cosine schedule is set for 200 epochs, so the LR has barely decayed when early stopping
    triggers (as in 60). Seed 42 may be the lucky case, so the 1-3% leads of the all-dataset
    model on the large datasets (80) may not be robust.
    - For comparison, seed replicates of the single-dataset models are tight: Grand Central
      0.2501 / 0.2519, PTDPES 0.2899 / 0.2915.
    - Job `fc_all_e60` trains the same recipe with `--epochs 60`, so the LR anneals within the
      run, for seeds 42, 1 and 2, each evaluated on test.
    - Decision 80 will be restated with seed spreads once the test results of both jobs are in.
      If the 60-epoch recipe is stable, it replaces the 200-epoch one for all-dataset training.
    **Seed replicates on test** (original 200-epoch recipe; agent ADE, seeds 42 / 1 / 2, against
    the single-dataset model):

    | Dataset | All-dataset seeds 42 / 1 / 2 | Mean | Single-dataset (2 seeds where run) |
    | --- | --- | --- | --- |
    | STEPS | 0.078 / 0.087 / 0.085 | 0.083 | 0.079 |
    | PTDPES | 0.284 / 0.313 / 0.320 | 0.306 | 0.290 / 0.292 |
    | Madras | 0.209 / 0.219 / 0.215 | 0.215 | 0.211 |
    | Grand Central | 0.244 / 0.271 / 0.274 | 0.263 | 0.250 / 0.252 |
    | UCY | 0.193 / 0.227 / 0.223 | 0.214 | 0.262 |
    | ETH | 0.234 / 0.271 / 0.263 | 0.256 | 0.393 |

    **Correction to 80.** Seed 42 was the best of three by a wide margin. Averaged over seeds, the
    original all-dataset recipe is *worse* than per-dataset models on the large datasets: STEPS
    +5%, PTDPES +6%, Grand Central +5%, Madras +2%. It is clearly better only on the two small
    ones (UCY -18%, ETH -35%). "Pooling never hurts" holds only for the lucky seed. The
    recommendation in 80 is suspended until the 60-epoch recipe results are in:
    - if it is stable and matches seed 42, pooling stands;
    - otherwise the recommendation becomes per-dataset models for large datasets and pooled
      models for small ones.

85. **Why the all-dataset model is seed-sensitive, and pool-then-fine-tune.**
    Per-dataset validation curves (`metrics.jsonl` of the all-dataset runs):
    - In every run, the lucky seed 42 included, the training loss flattens by about epoch 5
      (0.29-0.31) and barely moves after that, even while the LR decays in the 60-epoch runs.
    - Validation ADE per dataset then fluctuates around its level, by ±0.02 on PTDPES and Madras.
      Where a run ends up is largely decided early, and seed 42 settled lower.

    So the mixture is not overfitting. The optimiser stalls on the heterogeneous batches, and
    annealing does not rescue it. The 60-epoch runs are at 0.232-0.249 validation (best so far),
    against 0.220 for the original seed 42 and 0.247-0.251 for its two replicates.
    **Next.** Pool, then specialise. Job `fc_pool_ft` fine-tunes all-dataset models on PTDPES and
    Grand Central: 20 epochs, LR 3e-4 cosine, `--init-checkpoint`. It starts from both the lucky
    seed 42 and the ordinary seed 1, so the result is not tied to the lucky checkpoint. The
    question is whether pool-then-fine-tune beats per-dataset training (PTDPES 0.290 / 0.292,
    Grand Central 0.250 / 0.252) for both starting points.
    **Result (validation): pool-then-fine-tune inherits the starting point's luck.**

    | Model | PTDPES val ADE | Grand Central val ADE |
    | --- | --- | --- |
    | Single-dataset (2 seeds) | 0.321 / 0.323 | 0.264 / 0.265 |
    | Fine-tuned from pooled seed 42 | **0.313** (test 0.282) | **0.255** (test 0.240) |
    | Fine-tuned from pooled seed 1 | 0.340 | 0.273 |

    Fine-tuning from the lucky pooled checkpoint beats per-dataset training by 3-4% on test.
    From an ordinary one it is worse than training from scratch; 20 epochs at 3e-4 do not recover
    it. The problem is the pooled training itself. The flat training loss from epoch 5 suggests
    the LR, 0.96e-3, which was tuned by Optuna on STEPS alone, is too high for mixed six-dataset
    batches. Job `fc_all_lr` sweeps the pooled LR at 3e-4 and 1.5e-4, seeds 42 and 1, on the
    60-epoch schedule (`experiments/stars_ma/runs/cvae_all_lr*`).

    **Interim recommendation:**
    - UCY and ETH: the pooled model (better on every seed).
    - STEPS, PTDPES, Madras and Grand Central: per-dataset models.
    - The STEPS simulator model is unchanged (`cvae_kl01_finetune`).
    **Nuance on the stall.** `train_loss` includes the KL term, whose weight ramps up over the first
    10 epochs. That makes the total look flat, and in the LR sweep even rising (0.30 to 0.31-0.33,
    epochs 7-11), while the reconstruction still improves. Looking at `train_forecast` alone, the
    stall is real but milder: in the original seed-42 run it falls 0.479 -> 0.220 by epoch 5, then
    only to 0.207 by epoch 49. **Test results:**
    - Pool-then-fine-tune from the ordinary seed 1 is worse than per-dataset training: PTDPES
      0.303 against 0.290 / 0.292, Grand Central 0.259 against 0.250 / 0.252.
    - The 60-epoch seed-42 model lands at the typical level, not the lucky one: STEPS 0.087,
      PTDPES 0.313, Grand Central 0.260, UCY 0.215, ETH 0.256. The original seed 42 was a one-off,
      not something about the seed number.

86. **A lower pooled learning rate fixes the seed sensitivity; pooling beats per-dataset models.**
    Pooled six-dataset training at peak LR 3e-4 and 1.5e-4 (60-epoch cosine, two seeds each, job
    66434). All four runs reach validation 0.219-0.220, against 0.220-0.251 at 9.6e-4. Test
    agent ADE:

    | Dataset | LR 3e-4, seeds 42 / 1 | LR 1.5e-4, seeds 42 / 1 | Single-dataset | Pooled vs single |
    | --- | --- | --- | --- | --- |
    | STEPS | 0.0787 / 0.0783 | 0.0802 / 0.0791 | 0.0791 | -1 to +1% |
    | PTDPES | 0.282 / 0.282 | 0.283 / 0.284 | 0.290 | -2 to -3% |
    | Madras | 0.210 / 0.210 | 0.209 / 0.209 | 0.211 | -1% |
    | Grand Central | 0.243 / 0.243 | 0.242 / 0.244 | 0.250 | -2 to -3% |
    | UCY | 0.203 / 0.199 | 0.203 / 0.201 | 0.262 | -22 to -24% |
    | ETH | 0.235 / 0.235 | 0.229 / 0.238 | 0.393 | -40 to -42% |

    - **Cause.** The Optuna LR (9.6e-4) was tuned on STEPS alone and is too high for mixed
      batches. Seed 42 of the original recipe had been a lucky run at that LR.
    - **Conclusion, replacing the suspended recommendations of 80, 84 and 85.** With LR 3e-4, one
      pooled model is better than or level with a per-dataset model on every dataset, robustly
      across seeds. Per-dataset models are not needed.
    - **New all-dataset reference model:** `experiments/stars_ma/runs/cvae_all_lr3e-4_seed42`,
      chosen by validation (0.2193), not test. `configs/stars_ma/all.yaml` now sets
      `learning_rate: 3.0e-4`.
    - **Figures.** `docs/results/stars_ma_single_vs_all.png` shows the four runs as mean and
      range; `docs/results/stars_ma_all_test.png` shows the new reference model.
    - **STEPS simulator model unchanged** (`cvae_kl01_finetune`, 0.080 ADE, with 20 s rollouts
      validated for realism). The pooled model is level with it on STEPS (0.079), but its
      simulation realism has not been checked, so it does not replace it in SEAN-JAX.

87. **The pooled forecaster is as realistic in simulation; testing it as the SEAN-JAX crowd.**
    20 s STEPS test rollouts of `cvae_all_lr3e-4_seed42` against `cvae_kl01_finetune`:
    - mean speed 0.116 against 0.115 m/s (smoothed real 0.115);
    - speed Wasserstein to smoothed real 0.060 against 0.059;
    - acceleration Wasserstein 0.080 against 0.089;
    - collision rate 0.15% against 0.17%.

    So the pooled model passes the realism check of 48.

    Job `eval_pooled` evaluates v4 (and crowd v2) on the 270 test scenes in three setups:
    - the pooled model as both crowd and predictor (`cmaes_crowd_v4_pooled.yaml`);
    - the pooled model as the crowd, with the STEPS model as the predictor
      (`cmaes_crowd_v4_pooledcrowd.yaml`), to see whether v4 transfers to a crowd it was not tuned in;
    - the current STEPS setup, as a same-run reference.

    If v4 scores the same, the pooled model becomes the simulator default, so one model serves
    forecasting on every dataset and the crowd.
    **Result: the pooled crowd reacts about 3x less to the robot; the STEPS crowd stays the default.**
    v4 on 270 test scenes (`runs/seanjax/crowd_{steps_ref,pooled,pooledcrowd}`):

    | v4 | STEPS crowd + predictor | Pooled crowd + predictor | Pooled crowd, STEPS predictor |
    | --- | --- | --- | --- |
    | Success | 99% | 97% | 97% |
    | Collision | 0.7% | 0.4% | 0.4% |
    | Intimate-zone time | 2.6% | 2.2% | 1.9% |
    | Recorded success (30) | 100% | 93% | 90% |
    | `robot_influence` | 0.28 m | 0.11 m | — |
    | Objective | 0.387 | 0.554 | 0.559 |

    - Every policy scores higher in the pooled crowd (the default controller goes from -0.20 to
      +0.22). That is because `robot_influence`, the disruption term, falls to about a third: the
      pooled model's pedestrians change course much less for the robot. Objectives are therefore not
      comparable across crowds; behaviour metrics are.
    - v4 transfers to the pooled crowd without re-tuning (97% success, 0.4% collisions). That is
      useful robustness evidence.
    - **Decision.** The STEPS crowd stays the SEAN-JAX default. v4 was tuned in it, its realism was
      validated (48), and it is the more responsive, harder crowd. Neither model has seen a robot (the
      data has none), so which reaction strength is right is not something these data can settle.
      The pooled configs stay as a robustness check: `cmaes_crowd_v4_pooled.yaml` and
      `cmaes_crowd_v4_pooledcrowd.yaml`.

88. **Tuning for robustness across crowd models.** Decision 87 left open how strongly real people
    react to the robot: the STEPS crowd reacts about 3x more than the pooled one, and no data here
    can settle it. Rather than pick one, the reference policy should do well in both.
    - Job `cma_pooled` tunes in the pooled crowd (`cmaes_crowd_v4_pooled.yaml`), warm-started from
      v4, seeds 42 and 7.
    - Job `eval_cross` then evaluates both new policies and v4 in both crowds on the 270 test
      scenes (`runs/seanjax/cross_{steps,pooled}`).
    - **Selection rule:** the reference is the policy with the best *worse* result of the two
      crowds. The comparison uses behaviour metrics (success, collision, intimate-zone time,
      recorded success, loops) and each crowd's own objective, never objectives across crowds.
      v4 stays the reference unless a new policy is at least as good in the STEPS crowd and better
      in the pooled one.
    **Result: v4 stays the reference; tuning in the less reactive crowd makes policies aggressive.**
    (`runs/seanjax/cross_{steps,pooled}`, 270 test scenes each.)

    | Policy | STEPS crowd: success / collision / < 0.45 m / recorded / objective | Pooled crowd: success / collision / < 0.45 m / recorded / objective |
    | --- | --- | --- |
    | **v4** (tuned in STEPS crowd) | 98.5% / 0.7% / 2.6% / 100% / 0.388 | 97.4% / 0.4% / 2.1% / 93% / 0.558 |
    | pooled-tuned, seed 42 | 94.1% / 4.4% / 5.6% / 87% / 0.209 | 97.8% / 0.4% / 4.1% / 87% / 0.608 |
    | pooled-tuned, seed 7 | 94.8% / 3.7% / 6.6% / 80% / 0.246 | 97.0% / 1.9% / 5.7% / 87% / 0.551 |

    - Policies tuned in the pooled crowd, where pedestrians react about 3x less to the robot,
      learn to cut close. Intimate-zone time doubles, and in the more reactive STEPS crowd their
      collisions rise to 3.7-4.4%, five to six times v4's.
    - v4 is the best worst case. It is first in the STEPS crowd and, in the pooled crowd, level on
      collisions with half the intimate-zone time. The pooled-tuned seed 42 has the higher
      pooled-crowd objective only by trading comfort for fewer loops.
    - **Practical rule:** tune in the more responsive crowd. A crowd that barely yields teaches the
      policy that people will move, and that does not transfer.

89. **Score progress along the whole plan (another try at the dense-flow loops).** Correction
    first: in a progress note I proposed pass-side memory as "next". That was already built and
    found to have no effect (82); I should have checked this log first.
    - **New observation from the legible GIFs** (`docs/latest_policy/`): the dense-flow loops
      happen *near the goal*, which sits inside the cross flow. The controller's progress term
      compares only the goal distance at the plan's end. Near the goal, a plan that curls round
      and ends about as close scores the same as one that drives straight in, so the loop costs
      nothing.
    - **New option `progress_along_path`** (default off, so existing policies are unchanged):
      progress comes from the mean goal distance over the plan, normalised by half the distance a
      straight plan at full speed would cover. That rewards arriving sooner.
    - **Job 66878:**
      - v4 with the option on and no re-tuning, plus the waiting diagnostic and a dense-flow GIF
        (`runs/seanjax/meanprog_untuned`);
      - a re-tune from v4 with it on (`checkpoints/seanjax/cmaes_crowd_meanprog`).
    - Unlike 78-82, this changes how the goal term is computed rather than adding a penalty; the
      untuned check comes first.
    **Result: not adopted; the controller-tweak line of work ends here.** 270 test scenes, same
    objective:

    | Policy | Objective | Success | Collision | < 0.45 m | Loops (dense flow) | Recorded | Time |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | v4 | 0.387 | 98.5% | 0.7% | 2.6% | 0.08 (0.17) | 100% | 12.4 s |
    | v4 + option, untuned | 0.344 | 97.8% | 0.4% | 2.6% | 0.13 (0.35) | 90% | 12.5 s |
    | Re-tuned with option (`cmaes_crowd_meanprog`) | 0.357 | 97.8% | 1.9% | 4.8% | 0.07 (0.26) | 100% | 11.6 s |

    - Untuned, the option lowers *net* turning (2.08 to 1.15 rad in the waiting diagnostic) but
      raises total heading change: left and right turns cancel in the net figure, so it trades
      circles for zig-zags. The diagnostic's `net_turn_rad` is a poor proxy for loops; use the
      `loops` metric.
    - Re-tuned, it is 0.8 s faster but cuts closer: collisions nearly triple and intimate-zone time
      almost doubles. Dense-flow loops are worse (0.26 against 0.17).
    - **Across 68-89,** cost terms and memory (speed, retreat, idle turning, turn commitment,
      pass-side memory, progress along the plan) all fail to remove the dense-flow loops without
      giving up safety. They are a property of this one-step, re-planned trajectory-scoring
      controller in a crowd that does not yield. v4 stays the reference. A real fix needs a
      different planner, for example a short-horizon search over sequences of manoeuvres or a
      learned policy. That is recorded as future work, not started.

90. **Two-stage candidate plans: let "wait, then go" be scored.** Decisions 68-89 changed cost
    terms. The likely root cause is the plan structure. Every candidate is one constant `(v, w)`
    held for the whole plan, about 2 s, so "wait for the gap, then go" cannot be represented. A
    waiting plan ends no closer to the goal and scores badly, while an arc that curves round the
    flow scores better, and repeated every 0.1 s that becomes a loop.
    - **New option `two_stage`** (default off). Each `(v, w)` also appears as a two-stage plan:
      hold it for `stage1_time` (1 s), then steer for the goal, with turn rate set by the heading
      error at gain 2.5/s and speed slowed while turning. The safety and comfort costs apply to
      the whole plan. Only the first command is executed, as before.
    - Candidates double (132 instead of 66), so compute roughly doubles. That is the reason for
      this design over a full grid of command pairs (about 4,000 candidates).
    - **Job `twostage`:**
      - v4 untuned with the option (`runs/seanjax/twostage_untuned`, with the waiting
        diagnostic and a dense-flow GIF);
      - two re-tunes, seeds 42 and 7 (`checkpoints/seanjax/cmaes_crowd_twostage*`).

      The time limit is 4 h, after 89's 1 h limit cut a job short.
    - **Judged on the `loops` metric,** not net turning (89), alongside success, collisions,
      intimate-zone time and recorded success, all against v4.
    **First form failed badly; revised.** Giving *every* command a goal-steering second stage
    meant any first move, spinning in place included, was "rescued" by stage 2. All staged plans
    then scored nearly alike, and the robot executed whatever first command won by a hair. v4
    untuned: 42% success and 2.20 loops per episode, against 98.5% and 0.08
    (`runs/seanjax/twostage_untuned_v1`). The re-tunes were cancelled. Revised: only *waiting*
    gets a second stage, one wait-then-go candidate per hold time in `stage1_times` (0.5, 1.0,
    1.5 s), so there are 69 candidates instead of 66. The untuned check comes first this time.
    **Revised form, untuned: no effect.** v4 with wait-then-go candidates scores the same as v4:
    0.381 against 0.387, success 98.1%, collisions 0.7%, loops 0.08 (0.17 in dense flow). The three
    new candidates are essentially never chosen: with v4's weights, an arc round the flow still
    scores better than waiting and then going. A re-tune (two seeds, job `twostage_tune`) can
    rebalance that; its test evaluation is queued behind it (`runs/seanjax/twostage_tuned`).
    **Re-tuned: no gain; closed.** 270 test scenes (`runs/seanjax/twostage_tuned`):
    - v4: 0.367 objective, 97.8% success, 1.1% collisions, 2.6% intimate, 0.08 loops (0.16 dense flow).
    - Wait-then-go re-tune, seed 7: the same metrics, 0.12 dense-flow loops. Its search essentially
      kept v4's weights, and the difference is within noise.
    - Seed 42: 0.303, collisions 2.2%, intimate-zone time 4.6%.

    Even with the weights free, CMA-ES does not make use of waiting to cut loops. v4 stays the
    reference. With 90, the planner-structure route on the current controller (wait-then-go,
    one-stage grid) is also exhausted. The dense-flow loops (about 0.16 per episode, collision-free)
    are accepted as a known limitation, documented on the latest-policy page.

91. **A learned navigation policy, trained in the STARS crowd.** After 68-90, the remaining
    dense-flow loops look like a limit of re-planned one-step scoring. A learned policy can
    represent context-dependent behaviour, such as holding still while a cross flow passes, which
    a per-step cost rule cannot.
    - **Policy** (`seanjax/policies/learned.py`): an MLP with inputs 64 -> 64 -> 64 -> 2 (8.4k
      weights), outputs squashed to the robot's limits. Inputs, in the robot frame:
      - the goal;
      - the robot's own motion;
      - the 6 nearest visible pedestrians, with relative position and velocity and the STARS
        forecast 1 s and 2 s ahead;
      - the nearest obstacle.
    - **Safety shield**, mirroring v4's hard constraint: stop if the command, held for 1 s, would
      come within 0.05 m of a predicted pedestrian or a wall. The policy runs batched with one
      weight vector per environment. It loads in `build()` from
      `{policy: learned, weights: <.npy>}`, so `evaluate.py`, `compare_policies.py` and
      `visualize_rollout.py` work unchanged.
    - **Training** (`seanjax/scripts/train_learned.py`):
      1. behaviour cloning from v4 on 40 training scenes per kind (disjoint seeds), by MSE;
      2. OpenAI-ES: 48 antithetic perturbations, sigma 0.02, centred ranks, Adam lr 0.01,
         6 fresh training scenes per kind per generation, 300 generations. The objective is v4's
         (`cmaes_crowd_v4.yaml`), the validation set is fixed, and the best checkpoint is kept.
    - **Job `learned`:** seeds 0 and 1 (`checkpoints/seanjax/learned_v1*`). Afterwards: the
      270-scene test against v4, cross-crowd robustness (88), and loops in dense flow. It is
      adopted only if it beats v4 on the decision-88 worst-case rule.
    - Tested: a CPU smoke run of the whole pipeline; evaluate and visualise with a learned policy;
      a unit test of shapes, weights and the shield.
    **First 30 ES generations; shield v2.** Behaviour cloning alone gives 50% success and 20%
    collisions on validation (objective -1.30 / -1.35). ES lifts that to about -0.6 within 30
    generations: success about 70%, loops 0.05-0.08 (v4: 0.08). Collisions stay at 9-14%, against
    v4's 1%. The shield only *stops*, and in this crowd people walk into a stopped robot. v4
    instead picks another safe command. New `shield_fallback`: an unsafe command is replaced by
    the nearest safe command on a 7 x 11 `(v, w)` grid, checked against the pedestrian forecast
    and walls over 1 s, and the robot stops only if none is safe (tested). Job `learned2` trains
    with it from the same BC weights (`checkpoints/seanjax/learned_v2*`). Job `learned` keeps
    running as the stop-only comparison.
    **Progress at about generation 40-80.**
    - Stop-only shield (`learned_v1`): it plateaus at validation -0.5 with 75% success and 10-15%
      collisions by generation 80. Cancelled (job 67097).
    - Fallback shield (`learned_v2`): -0.16 / -0.17 at generation 40, with 88-89% success, 7-8%
      collisions and 0.19-0.23 loops. Far better, but still well short of v4 (about 0.39, 99%
      success, 1% collisions).
    - **Next:** ES with 48 perturbations is noisy for 8.4k weights. Job `learned3` uses 96 (same BC
      start, fallback shield, seeds 0 and 1, `checkpoints/seanjax/learned_v3*`), alongside v2.
    **The steering network plateaus.** `learned_v2` holds at -0.08 to -0.22 validation from
    generation 40 to 90 (88-91% success, 6-10% collisions). `learned_v3` (population 96) reaches
    the same level faster but no higher. Both are far below v4 (about 0.39). v2 is cancelled; v3
    runs to completion for the record.

92. **Context-dependent controller settings instead of a steering network.** A network that
    steers has to relearn everything v4 already does well, above all keeping clear of people. New
    `seanjax/policies/contextual.py`: the social controller still plans, with its hard safety
    constraint, and a small MLP adjusts 10 of v4's tuned settings at every step from the
    observation. Inputs are 64 -> 32 -> 10, about 2.4k weights, and each setting moves at most
    ±0.5 of its CMA-ES range in unit scale. The tuned settings are:
    `goal_weight`, `pedestrian_weight`, `disruption_weight`, `preferred_speed`,
    `personal_space_radius`, `prediction_horizon`, `smoothness_weight`, `speed_scale`,
    `retreat_scale` and `alignment_weight`.
    - The output layer starts at zero, so **training starts exactly at v4** (unit-tested); no
      behaviour cloning is needed. For example, the network can raise the retreat cost or lower
      the preferred speed only in dense cross flow.
    - Context features use constant-velocity pedestrian extrapolation, so v4's STARS predictor
      and its forecast schedule are unchanged.
    - `train_learned.py --kind contextual` writes
      `{policy: contextual, base: <v4 yaml>, weights: <.npy>}`, which `build()` loads.
    - **Job `contextual`:** ES with population 48 for 200 generations, sigma 0.02 and 0.05, seeds
      0 and 1 (`checkpoints/seanjax/contextual_v1*`). Adopted only if it beats v4 on the
      worst-case rule of 88.
    **First run: ES drifts away from v4 instead of improving.** Within 30 generations validation
    falls from 0.36-0.38 to 0.30-0.33, and collisions rise from 0 to 0.5-1.4%. The training-set
    mean is flat (0.28-0.34), and the best member of a generation beats the mean by about 0.125.
    v4 sits at an optimum of this very objective, so most perturbations of its settings hurt, and
    Adam at lr 0.01 moves every weight by about lr per step whatever the gradient's
    signal-to-noise. With noise-dominated ES gradients that is a random walk away from v4.
    Restarted (job `contextual2`, `checkpoints/seanjax/contextual_v2*`):
    - lr 0.002 (sigma 0.02) and lr 0.001 (sigma 0.05);
    - population 96 and 8 scenes per kind per generation, for less noisy gradients;
    - 150 generations.

    The first runs are kept as `contextual_v1*_lr01`. `learned_v3` (steering network,
    population 96) was cancelled at generation 48: validation -0.13 to -0.20, 86-89% success,
    8-9% collisions, a plateau like v2's.
    **Restart: still drifts; the learned-policy work concludes (91-92).** Even at lr 0.001-0.002
    with population 96 and 8 scenes per kind, all four runs decline. Validation falls from
    0.36-0.38 to 0.33-0.37 by generation 10, the training-set mean from 0.33-0.35 to 0.29-0.31,
    and collisions rise from 0 to 0.5-0.9%. The ES gradient signal is weaker than its noise. From
    an optimum of the very objective being optimised (v4 is CMA-ES-tuned on it), every step size
    random-walks away. Cancelled (job 67219).

    **Conclusions:**
    - A network that steers (91) plateaus at about -0.15 validation, with 87-90% success and
      7-10% collisions, even with a fallback shield and population 96. It is far below v4.
    - Context-dependent settings around v4 (92) start at v4 but ES cannot find adjustments that
      beat it at this budget.
    - v4 (`checkpoints/seanjax/cmaes_crowd_v4/best_policy.yaml`) remains the reference. On the
      evidence of 68-92 it is close to the best this controller and objective allow in the STARS
      crowd. Its remaining weakness, about 0.16 collision-free loops per dense-flow episode, is
      documented.
    - Further gains would need either much more compute (gradient-based RL, which needs a
      differentiable or model-based surrogate, or ES with thousands of members) or a different
      objective or crowd (for example robot-interaction data, 87). That is the user's call.

93. **Adding ATC to pooled forecaster training.** The pooled configs left ATC out "until its scene
    index is built". `datasets/atc` is empty, but `datasets/atc_subset` holds the converted
    two-day subset: 2013-05-22 and 2013-09-29, about 77k tracks. It has a defined split:
    `held_out_recording`, with 29 September as the entire test split and 22 May divided 90/10
    into train and val along its timeline. Only the forecast scene index was missing.
    - The registry entry `atc` now points at the subset with that split strategy.
      `configs/stars_ma/all7.yaml` is the six-dataset mix plus ATC, at the robust pooled LR of
      3e-4.
    - **Job `fc_atc`**, in order:
      1. `preprocess.py` builds the ATC scene index;
      2. zero-shot test of the reference pooled model on ATC (no ATC in training);
      3. a seven-dataset pooled model, `cvae_all7_lr3e-4_seed42`;
      4. its per-dataset test evaluation.
    - **Question:** does adding ATC (a large, dense, different environment) keep the
      six-dataset results, and how much does ATC gain over zero-shot?
    **Preprocessing stall.** The first pass ran at about 765 windows/s while the earlier
    categorisation cache covered the data (22 May, and 29 September up to about 3.8M of 5.18M
    windows), then fell to about 7 windows/s at the first uncached windows. The node has 144 cores
    but a load average of about 2.4: the 16 workers mostly wait. The pipeline keeps 4 tasks per
    worker in flight and resolves the oldest first, so a few very slow windows (the densest ATC
    crowds) stall everything. This is likely why the ATC index was never built: the earlier
    cache stops at the same point. Job 67265 was cancelled. `atc_index` reruns only the
    preprocessing with 128 workers (512 tasks in flight). The zero-shot evaluation and the
    seven-dataset training follow as separate jobs. **Fallback** if it still stalls: index only
    2013-05-22, split along its own timeline, and record that ATC's test day is then a different
    part of the same day.
    **Index built; ATC is capped.** The ATC scene index has 8.08M windows: train 2.67M, val 0.24M,
    test 5.18M. Each person present at each step starts a window, so they overlap heavily. At
    that size ATC would make every pooled epoch about 15x longer and swamp the other six datasets:
    dataset weights only change sampling, not epoch length. New per-dataset option `max_windows`
    (a number, or a per-split mapping) draws a seeded subset of index rows *before* window
    extraction; the cache key includes the row count. `all7.yaml` caps ATC at 80k train, 10k val
    and 50k test windows, the scale of STEPS and PTDPES. Smoke-tested on CPU.
    **Jobs:**
    - `atc_zeroshot`: the six-dataset reference on all seven test sets
      (`eval_test_all7_zeroshot.json`);
    - `fc_all7`: seven-dataset models, seeds 42 and 1, each followed by a seven-dataset test.
