# ETH/UCY closed-loop crowd-simulation benchmark

How realistic is STARS when it drives a whole real crowd in closed loop? Each pedestrian starts from its real observed history, and the result is compared with what the recording shows next. The same comparison runs for constant velocity, straight-to-goal, social force and ORCA.

- Code: `src/simplenavsim/realism/` and `simplenavsim/scripts/ethucy_benchmark.py`.
- Tests: `tests/test_ethucy_benchmark.py`.
- What existed and what was added: [`evaluation_audit.md`](evaluation_audit.md).

```bash
# Everything (calibrate, run, aggregate, plot, render, report) into results/ethucy_<date>_<commit>
sbatch --time=04:00:00 scripts/horizon/gpu_jobs.slurm \
  "PYTHONPATH=\$PYTHONPATH:/scratch/09865/mmunje/.cache/ffmpeg_pkgs python simplenavsim/scripts/ethucy_benchmark.py all --device cuda"
# Stages separately, or after changing only the statistics or the rendering
python simplenavsim/scripts/ethucy_benchmark.py aggregate --run results/<run_id>   # then plot / render / report
# Tiny end-to-end check on CPU (zara1, 6 windows per recording, 2 seeds, 1 scenario), into a temp dir
python simplenavsim/scripts/ethucy_benchmark.py smoke
python simplenavsim/scripts/ethucy_benchmark.py --help
```

## Protocol

### Scenes and rate

- The five scenes are eth, hotel, univ (students001 + students003), zara1 and zara2.
- Positions are in world metres at 0.4 s, from 8 observed steps (3.2 s).
- Horizons are **2, 4, 4.8, 8 and 12 s** (5, 10, 12, 20 and 30 steps). Every horizon is cut from the same 12 s rollout.

### Windows and fixed cohort

- A window starts every 10 steps.
- The **cohort** is every track present at the last observed step t0. They are simulated jointly. Missing history steps are masked for STARS; the classical models use the current position.
- Pedestrians who enter later are *not* simulated (a fixed cohort), and they are counted as late arrivals.

### Scored agents

An agent is scored at horizon H only if all three of these hold:

- it has a full 8-step history;
- it has a continuous recorded track through t0 + H;
- its track continues past t0 + H, so its goal lies beyond every scored position.

The other cohort members are *context agents*: simulated, but never scored or drawn. Per-horizon counts of windows, scored agents and every exclusion reason are in `metrics/counts.csv` and `report.md`.

### Goals and leakage

- **Oracle goal:** the last recorded position of the track. Tracks that end on the recording's last step are flagged (`goal_at_boundary`).
- **No leakage:** simulators receive only `Window.inputs()`, i.e. the history, its mask and the goals. A test checks that changing the ground-truth future leaves every rollout unchanged.

### Closed loop

- **STARS** re-forecasts every 0.4 s from its own simulated history and executes the first step. CVAE intents follow `z = mu + sigma·eps`, with eps an AR(1) process (ρ = 0.8), seeded.
- **SFM and ORCA** integrate at 0.1 s with a desired velocity toward the goal at the agent's mean observed speed, and stop within 0.3 m of the goal.
- **CV** extrapolates the last observed velocity.
- **goal_straight** walks straight to the goal at the mean observed speed.

### Methods

| Row | What |
| --- | --- |
| `ground_truth` | recorded futures (reference for collisions and kinematics) |
| `stars_steps` | STEPS-only checkpoint, **zero-shot** on ETH/UCY |
| `stars_pooled` | six-dataset checkpoint. **In-domain**: trained on hotel, zara1 and students001, validated on students003, held out only on eth and zara2 |
| `stars_*_goal` | the same plus the inference-time goal-steering adapter (λ calibrated); the checkpoints themselves are not goal-conditioned |
| `cv`, `goal_straight` | simple baselines |
| `sfm`, `orca` | calibrated on `crowds_zara03`, `arxiepiskopi1` and `uni_examples` (never on the benchmark scenes), by grid search on ADE at 8 s |
| `sfm_default`, `orca_default` | repository Helbing-style defaults; CrowdNav ORCA settings |

STARS rows use 5 seeds, and the report gives the **expected** value over seeds, never best-of-K.

### Metrics (scored agents)

- ADE and FDE (Euclidean, m).
- SDE, the mean squared error in m². It is reported separately and never mixed into ADE.
- nADE, over non-linear ground-truth futures.
- Collisions between scored agents, with a disc radius of 0.2 m so contact means d ≤ 0.4 m:
  - unique colliding pairs and colliding pair-steps;
  - agents involved in a collision;
  - hACS and sACS (Mahzoon et al., IJCAI 2025).
- Minimum separation.
- Minimum distance to the oracle goal, the share coming within 0.5 m of it, and the minimum distance to the ground-truth endpoint.
- Speed, acceleration and turning-rate distributions, including ground truth, with the Wasserstein-1 distance to ground truth.
- Runtime.

### Aggregation and statistics

- **Averages.**
  - *Micro*: all scored agents pooled.
  - *Macro*: the mean of the five dataset averages.
  - *Per dataset*.
  - `heldout_eth_zara2`.
- **Confidence intervals.**
  - 95% blocked bootstrap with 2,000 resamples.
  - Each block is 50 steps (20 s) of one recording, resampled within its dataset. This accounts for overlapping windows sharing agents.
- **Paired comparisons** (`metrics/paired.csv`) use the same resamples for both methods.

### Media

- **Scenarios.** 10 are drawn deterministically: 2 per dataset at 8 s, with a seeded draw among windows that have at least 4 scored agents and at least 60% of the cohort scored.
- **For each scenario:**
  - anonymous single clips of all 9 study models (`preference_export/clips`; 960×720, 12 fps, H.264 yuv420p MP4 plus GIF);
  - a labelled 3×3 comparison grid (`media/comparison_grids`);
  - ground truth vs both STARS models (`media/{mp4,gifs}`).
- **Case studies** are kept apart, in `media/case_studies`: the windows with the largest `stars_steps` ADE at 8 s.
- **Rendering** is identical for every model: same view, discs, 2 s trail, and linear interpolation for display only.

### Preference export

- `preference_export/manifest.json` follows `/scratch/09865/mmunje/preference_manifest_schema_v1.md` (schema 1.0).
- It is checked by `validate_manifest` and a test.
