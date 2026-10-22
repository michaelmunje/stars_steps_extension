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
