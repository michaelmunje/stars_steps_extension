# Evaluation audit: ETH/UCY closed-loop simulation benchmark

Audit of what the repository provided before building the ETH/UCY realism benchmark
(`docs/ethucy_benchmark.md`), on branch `ethucy-benchmark` (from `scale-and-sim`, 2026-11-05).

## Data

| Scene | Recording(s) | File | Notes |
| --- | --- | --- | --- |
| eth | `seq_eth` | `datasets/eth/converted/seq_eth.traj.parquet` | |
| hotel | `seq_hotel` | `datasets/eth/converted/seq_hotel.traj.parquet` | |
| univ | `students001`, `students003` | `datasets/ucy/converted/*.traj.parquet` | dense, about 40 pedestrians per window |
| zara1 | `crowds_zara01` | same | |
| zara2 | `crowds_zara02` | same | |
| calibration only | `crowds_zara03`, `arxiepiskopi1`, `uni_examples` | same | UCY recordings outside the five scenes |

- **Format.** Converted tracks have columns `track_id, step, x, y, heading`, in world metres.
  - The converters (`src/datasets_preprocessing/{eth,ucy}`) declare `ANNOTATION_HZ = 2.5`, so one step is 0.4 s.
  - `tests/test_ethucy_benchmark.py::test_units_and_rate` checks that the median walking speed lies between 0.5 and 2 m/s.
- **Gaps.** No track in the five scenes has an internal gap; tracks only start and end.
- **What breaks the windows is density, not gaps.** In univ almost every window contains someone whose track ends within the horizon. So the cohort cannot be required to be fully scored (that rule gave 0 univ windows), and the benchmark scores a subset instead (see the protocol).

## Checkpoints

| Row | Run | Exposure to ETH/UCY |
| --- | --- | --- |
| `stars_steps` | `experiments/stars_ma/runs/cvae_kl01_finetune/checkpoints/best.pt` | none (trained on STEPS only): zero-shot |
| `stars_pooled` | `experiments/stars_ma/runs/cvae_all_lr3e-4_seed42/checkpoints/best.pt` | trained on `steps, ptdpes, madras, grand_central, ucy, eth` |

The pooled checkpoint's ETH/UCY split covers whole recordings (`datasets/{eth,ucy}/split_manifests/grouped-seed42.csv`):

| Split | Recordings | Exposure |
| --- | --- | --- |
| train | `seq_hotel`, `crowds_zara01`, `students001` | trained on |
| val | `students003` | checkpoint selection |
| test | `seq_eth`, `crowds_zara02` | held out |

The three calibration recordings are in neither split, so neither checkpoint has seen them.

**`stars_pooled` is in-domain and not zero-shot.** Every table labels its exposure per dataset, and the benchmark adds a `heldout_eth_zara2` aggregate.

Both checkpoints share a window of 8 observed and 12 predicted steps at dt = 0.4 s. That matches the benchmark.

## Capabilities that existed

- **STARS multi-agent forecaster.**
  - `load_forecaster`, `graph_batch_from_padded`, `encode`, `decoder.prior_params`, `decode` and `to_scene`.
  - A closed-loop crowd recipe: `simplenavsim/pedestrians/stars_crowd.py` re-forecasts every 0.4 s, executes the first step, and uses AR(1)-correlated CVAE intents with ρ = 0.8.
  - The benchmark reimplements this loop without a robot, walls or a reference warm-up; `realism/sim.py::Stars` does this.
- **Social force.**
  - `simplenavsim/pedestrians/social_force.py` provides `social_forces` and `integrate`.
  - The benchmark reuses these as-is, with an absent robot and no obstacles.
- **ORCA.**
  - It had been removed from `scale-and-sim` in a simplification.
  - `src/simplenavsim/pedestrians/orca.py` and `tests/test_simplenavsim_orca.py` are restored from `release/scale-and-sim-archive`.
  - One change was needed: the robot velocity is now built from the current `RobotState` (`v`, `pose`).
  - The archive's ORCA tests pass.
- **Rendering.** `simplenavsim/visualization` targets robot scenes, so a small matplotlib renderer is used instead (`realism/render.py`).
- **ffmpeg.** No system ffmpeg is available. The `imageio_ffmpeg` binary under `/scratch/09865/mmunje/.cache/ffmpeg_pkgs` is used through `PYTHONPATH`; `.venv` is not modified.

## Gaps and decisions

1. **No goal conditioning in STARS.**
   - The checkpoints are not goal-conditioned.
   - The `*_goal` rows use an *inference-time goal-steering adapter*: each executed step is blended toward the oracle goal at the forecast's own step length, `step' = (1 − λ)·step + λ·|step|·unit(goal − pos)`.
   - λ is calibrated on the calibration recordings.
   - This is an adapter, not a property of the model. Retraining with goals is deferred.
2. **No ETH/UCY closed-loop harness.**
   - It is new: `src/simplenavsim/realism/` plus `simplenavsim/scripts/ethucy_benchmark.py`.
   - The protocol is in `docs/ethucy_benchmark.md`.
3. **Baseline calibration.**
   - SFM, ORCA and λ are grid-searched only on the three calibration recordings, never on the benchmark scenes. The objective is ADE at 8 s.
   - The uncalibrated settings are reported as separate rows:
     - `sfm_default`: the repository's Helbing-style defaults;
     - `orca_default`: CrowdNav pedestrian ORCA settings, i.e. 10 m neighbour distance, 10 neighbours, 5 s horizon and 0.3 m radius.
4. **Collision severity** follows Mahzoon et al. (IJCAI 2025, hACS and sACS). Counted pairs and pair-time rates are reported alongside.
5. **Missing dependencies.** `tabulate` is not installed, so the report writes its own Markdown tables.
6. **Compute.** The login node has one core. Full runs go through `scripts/horizon/gpu_jobs.slurm`, with STARS on the GPU and the classical baselines on the CPU.
