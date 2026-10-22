# Data pipeline, analysis and paper artifacts

Commands run from the repository root with `.venv` active (or prefixed with `uv run`).

## Task representations

Canonical Parquet stores the original converted metric-world coordinates directly at
2.5 Hz. Smoothing, filtering, target-centric normalization, and window materialization
are derived views and never overwrite canonical `x, y`. The loader promotes one complete
pedestrian to primary agent zero and returns positions plus validity masks; it does not
silently bake analysis smoothing into forecasting inputs.

Two intentionally separate sample representations are supported:

- OpenTraj analysis trajlets: 12 returned samples with starts seven samples apart. A
  thirteenth source sample is required only for the strict
  `||x[s+12] - x[s]|| > 1 m` static-retention test.
- Forecasting scenes: 8 observed + 12 future samples, with dense one-timestep training
  stride. OpenTraj's static filter is not applied automatically.

## Loader

After running `uv sync` (or activating `.venv`), use the loader with:

```python
from config import Config, DatasetSpec
from data import make_loaders

train, val, test = make_loaders(
    train_datasets=[DatasetSpec("atc", "datasets/atc")],
    eval_dataset=DatasetSpec("eth", "datasets/eth"),
    cfg=Config(batch_size=64),
)

for batch in train:
    xy = batch["xy"]          # [batch, 20, max_agents, 2]
    valid = batch["valid"]    # [batch, 20, max_agents]
    tags = batch["scene_tag"] # primary trajectory type, 1..4
```

Scene indices, categorization, acceptance sampling, and sizing are explicit
preprocessing operations:

```bash
python -m preprocess datasets/eth --data-seed 42
```

STEPS (the indoor atrium's cleaned fixed-lag tracker exports) is converted and indexed with:

```bash
python -m datasets_preprocessing.steps.convert datasets/steps
python -m preprocess datasets/steps --name steps --data-seed 42
```

The STEPS adapter stores confirmed observed posterior states at 2.5 Hz, excludes
the incomplete final fixed-lag second, and saves prediction/camera/finalization audit
fields under `datasets/steps/quality/`. See
[`src/datasets_preprocessing/steps/README.md`](../src/datasets_preprocessing/steps/README.md).

Splits are grouped by the indivisible `scene_sequence` stored in Parquet metadata.
Dense overlapping windows may exist inside one partition, but one source recording or
sequence can never appear in multiple partitions. Preprocessing writes persistent JSON,
CSV, and Markdown manifests under `split_manifests/` and asserts disjoint source
identities. Generic LOSO is a separately named policy grouped by physical `scene_id`:

```bash
python -m preprocess datasets/ucy --name ucy --split-strategy leave_one_scene_out --held-out-scene zara1
```

LOSO is rejected when fewer than two physical scenes exist. The established ETH/UCY
five-location protocol remains separate under `configs/protocols/eth_ucy_five_fold.json`.

For the native `target_complete` membership policy, each sample is one `(window, primary pedestrian)` pair. Agent zero is the primary
and has all 20 timesteps. Neighbors may contain NaNs and are sorted by distance
to the primary at the eighth observation. The `steps_default` target-centric SE(2) transform uses
observations only: it places the primary's last observed position at the origin and
chooses heading from observed velocity or observed displacement, with deterministic
fallbacks. Predictions are inverse-transformed before metrics and plotting.

## OpenTraj-compatible analysis

Build the non-causal analysis view, exact OpenTraj trajlet index, indicators, and
upstream-style PDF figures without changing canonical data:

```bash
python -m preprocess datasets/eth --name eth --data-seed 42
python -m analysis_pipeline datasets/eth --data-seed 42
python -m opentraj_report datasets/eth --plot-dir paper/figures/opentraj/eth
```

The order above lets analysis indexing attach the independently computed TrajNet++
category to matching `(scene_sequence, agent, start_step)` analysis trajlets. Every
trajlet records its returned end and separate
static-test endpoint, timestamps, displacement, OpenTraj-static flag, retention flag,
and recording/sequence identity. Pedestrian identities are always namespaced as
`(scene_sequence, recording, track_id)`; a numeric tracker ID is never assumed to be
globally unique.

The analysis view uses OpenTraj's full-trajectory Kalman/RTS smoothing and is explicitly
non-causal. Forecast inputs default to `raw`. Leakage-safe alternatives are
`causal_kalman` and `observation_only_smooth`; the non-causal compatibility mode is
explicitly named `opentraj_offline_smooth_compat`. Supported training adapters accept,
for example:

```bash
./scripts/pretrain_baseline.sh --model moflow --dataset ptdpes --seed 0 --forecast-input-preprocessing causal_kalman
```

Trajectron++ uses its faithful raw canonical environment adapter by default. Every run
config and checkpoint embeds the fully resolved recipe, preprocessing hash, upstream
repository/commit, canonical release, and leakage-relevant fields.

### Publication Tables 1 and 2

The publication tables have a dedicated source of truth and do not use legacy
`stats.py`. Membership and static metadata are frozen in
`configs/protocols/paper_datasets.json`; the resolved sequence list and full-precision primitive
counts are written to `paper/stats/paper_dataset_metrics.json`.

```bash
python -m scripts.prepare_publication_analysis --workers 8
python -m paper_dataset_metrics
```

The first command builds explicit publication OpenTraj artifacts independently of
forecasting folds/splits and writes OpenTraj-compatible PDFs under
`paper/figures/opentraj/`. The second writes only `paper/table1.tex` and
`paper/table2.tex`. Normal mode fails if any of the eight required rows is unavailable;
`--allow-partial` is only for development. Table values use the complete retained
trajlet population. Plotting downsampling in `opentraj_report.py` is never part of table
aggregation.

Publication provenance is chained explicitly: each trajlet index records the exact
canonical file-set revision and analysis implementation hash, each indicator artifact
records the SHA-256 of its source index and indicator implementation, and the final JSON
records those artifact hashes. Stale or mixed-version artifacts are rejected with a
regeneration instruction.

The publication CV position signal is explicit in `configs/task.json` under
`publication.cv_preprocessing`. It is currently `opentraj_kalman_smooth`, matching the
offline-smoothed representation used by the OpenTraj Figure 1 trajlets. This applies the
Schöller last-observed-displacement rule to smoothed positions and evaluates against the
smoothed future. Reproduce the unsmoothed result without changing configuration with:

```bash
python -m paper_dataset_metrics --cv-preprocessing raw_world_coordinates
```

Both modes use the same complete, unsampled 8+12 window population. Named JSON and
LaTeX copies are retained for each mode so they cannot be confused. The standard model
benchmark remains raw Schöller CV; offline smoothing is publication analysis only.

The three representations are intentionally distinct:

- OpenTraj analysis trajlet: 12 returned smoothed samples, a 13th endpoint for the
  strict static test, descriptive metrics only.
- Forecasting window: configurable model population, normally 8 observed + 12 future.
- Publication CV window: an independently enumerated complete, consecutive 8+12
  single-track window with stride one and no model-index filters; its raw or
  OpenTraj-smoothed position signal is recorded explicitly.

### Per-fold ETH/UCY statistics and OpenTraj Table 1 reproduction

Two further paper-artifact generators are Python module entry points (not shell launchers, so
they are not in `scripts/README.md`) that are run by hand rather than chained into the
regeneration scripts above:

```bash
python -m paper.generate_eth_ucy_folds --data-seed 42
```

For each of the five standard ETH/UCY folds (`eth`, `hotel`, `univ`, `zara1`, `zara2`), writes
`paper/stats/eth_ucy_folds/<fold>.json` (constant-velocity ADE/FDE/collision metrics and the
OpenTraj-compatible indicator report) and the corresponding figures under
`paper/figures/eth_ucy_folds/`, then aggregates everything into
`paper/constant_velocity_eth_ucy_folds.tex`, `paper/table_eth_ucy_fold_stats.tex`,
`paper/table_eth_ucy_fold_indicators.tex`, and `paper/eth_ucy_fold_metrics.json`. `--data-seed`
defaults to `42` and must match the forecast-window index (`scenes-seed<N>.parquet`) already built
for `eth` and `ucy`; it also reads each dataset's `opentraj-analysis-trajlets.parquet`. Re-run
after regenerating the ETH or UCY canonical dataset, its forecast-window index, or its OpenTraj
analysis trajlet index.

```bash
python -m scripts.audit_opentraj_table1 eth_univ
python -m scripts.audit_opentraj_table1 eth_hotel
python -m scripts.audit_opentraj_table1 ucy_zara
python -m scripts.audit_opentraj_table1 ucy_students
```

Reproduces OpenTraj's published Table 1 (pedestrian counts, recording duration, trajectory
duration, trajlet counts, non-static percentage) against the pinned OpenTraj source values baked
into the script, one dataset group per invocation (the positional argument is one of `eth_univ`,
`eth_hotel`, `ucy_zara`, `ucy_students`). Each run writes
`paper/stats/opentraj_table1_reproduction/<dataset>.json` and rebuilds
`paper/opentraj_table1_reproduction.md` from every per-dataset JSON found so far, so all four
groups must be run (in any order) to regenerate the complete log. `ucy_zara` and `ucy_students`
read the archived seven-sequence UCY conversion under
`datasets/_backups/ucy_seven_sequence_v3_20260907/` for the sequences (`crowds_zara03`,
`uni_examples`) that the standard five-fold forecasting corpus excludes. The script independently
recomputes the expected `opentraj-analysis-trajlets.parquet` implementation hash and canonical
source-set revision and raises `ValueError` if the on-disk index is stale, so re-run it after
regenerating the relevant canonical dataset or its OpenTraj analysis trajlet index (i.e.
`analysis_pipeline.py`, `analysis_trajlets.py`, or `preprocessing_views.py`).

### Forecasting recipes and ablations

Inspect the literature-compatible defaults without loading data:

```bash
python -m forecasting_config --list-forecast-presets
python -m forecasting_config --forecast-preset trajectron_pp --show-resolved-forecast-config
python -m forecasting_config --forecast-config my_ablation.yaml --coordinate-frame world --show-resolved-forecast-config
```

The canonical preset names are `social_gan`, `scholler`, `trajectron_pp`,
`social_vae`, and `steps_default`; common literature spellings are accepted as aliases.
Training/evaluation CLIs accept `--forecast-preset` plus optional overrides including
`--coordinate-frame`, `--feature-representation`, `--derivative-mode`, signal modes,
standardization, window sizes/stride,
membership/completeness policy, and all rotation/flip/scale parameters.
Unspecified CLI arguments remain `None` and therefore never overwrite a preset or
config-file value. Precedence is library defaults, named preset, config-file overrides,
then explicit CLI overrides.

Example exact recipe and a deliberately modified ablation:

```bash
python -m baselines.train_trajectron --stage pretrain --run-dir experiments/tpp --forecast-preset trajectron_pp
python -m baselines.train_social_vae --stage pretrain --run-dir experiments/sgan_ablation --forecast-preset social_gan --forecast-input-preprocessing causal_kalman
```

Modified recipes are recorded with `forecast_preset_modified=true` and
`forecast_literature_exact=false`; they are never labelled exact literature
preprocessing.

The Trajectron++ adapter validates this contract and fails loudly if an override cannot
be represented faithfully by upstream Trajectron++ rather than silently ignoring it.

Numerical parity is pinned to OpenTraj commit
`97ec1d5e1b579f26febeaa9083705952deac338a`:

```bash
python -m scripts.validate_opentraj_parity /path/to/OpenTraj \
  datasets/eth/analysis/opentraj_kalman_smooth/*.analysis.parquet
```

The recorded ETH/UCY comparison is in `references/opentraj_parity_eth_ucy.json`.
OpenTraj supplies characterization utilities and a constant-velocity baseline, not the
learned training harness needed here. Learned forecasting therefore consumes the same
canonical Parquet through the separate leakage-safe forecasting branch below.

Generate the consolidated OpenTraj-compatible recording duration, summed trajectory
duration, raw-trajectory count, exact 4.8 s trajlet count, and average adaptive local
density for every available dataset with:

```bash
python -m opentraj_dataset_stats
```

Machine-readable output is written to `paper/stats/opentraj_dataset_summary.{json,csv}` and
paper-ready output to `paper/table_opentraj_dataset_summary.tex`. The local-density
column is the arithmetic mean of OpenTraj's per-non-static-trajlet maximum density; it
uses OpenTraj's adaptive nearest-neighbour Gaussian estimator, not a fixed-radius count.
For a complete already-converted dataset pass, run:

```bash
./scripts/regenerate_dataset_phase1.sh eth 42
```

### Regenerating a canonical dataset from raw input

The dataset wrapper runs conversion, dense forecast indexing, OpenTraj analysis,
indicators, and statistics in sequence:

```bash
./scripts/regenerate_canonical_dataset.sh eth 42
./scripts/regenerate_canonical_dataset.sh ucy 42
./scripts/regenerate_canonical_dataset.sh atc 42
./scripts/regenerate_canonical_dataset.sh steps 42
./scripts/regenerate_canonical_dataset.sh madras 42
./scripts/regenerate_canonical_dataset.sh grand_central 42
./scripts/regenerate_canonical_dataset.sh ptdpes 42
```

Set `SKIP_ANALYSIS=1` to stop after forecast preprocessing and statistics, or
`SKIP_FORECAST=1` to resume analysis/statistics from an existing forecast index.
`PREPROCESS_WORKERS=8` controls categorization and recording-level OpenTraj smoothing;
`ATC_WORKERS=8` controls ATC's streaming conversion pool. PTDPES and MADRAS commands fail before doing work when their local raw
directories are absent.
Publication tables take each dataset's rows from its analysis records
([`configs/protocols/paper_datasets.json`](../configs/protocols/paper_datasets.json)).

## Visualization

Generate GIFs and final PNGs from loader-backed samples:

```bash
python -m visualize datasets/atc \
    --split val \
    --count 10 \
    --seed 42 \
    --output-dir paper/figures/trajectory_debug/atc-val
```

Use `--selection crowded` to prioritize high-pedestrian-count windows while
keeping selected windows temporally separated, rather than sampling uniformly.

The visualizer uses `SceneDataset` directly and reads all timing from `configs/task.json`.
Solid lines show loader trajectories and the dashed line is the primary CV baseline.

For tracking diagnostics, bypass forecasting-scene filtering and render only named
track IDs around an exact recording-relative time. The requested time is placed at
the final observation timestep of the 8+12 window:

```bash
python -m visualize datasets/steps \
    --recording atrium_20260818_094501 \
    --track-ids 7 43 \
    --time-seconds 199.2 \
    --output-dir paper/figures/trajectory_debug
```

## Zero-training pipeline check

```bash
python -m paper.generate_tables --model constant_velocity --all-datasets
```

This evaluates every preprocessed dataset, appends frozen-schema rows to
`experiments/results/runs.parquet`, and invokes the dedicated publication generator for
`table1.tex` and `table2.tex`.
It fails if a converted dataset lacks its Phase-1 scene index; use
`--allow-partial` only for an explicitly partial smoke test.

Dataset statistics and coverage can also be inspected independently:

```bash
python -m stats datasets/ucy --name ucy
python -m data_quality datasets/eth datasets/ucy datasets/atc
python -m coverage datasets/eth datasets/ucy --output paper/figures/coverage/coverage.pdf
```

All three commands read the canonical storage frequency from [`src/canonical.py`](../src/canonical.py).

The main density column uses one denominator for every dataset: the median
per-recording effective walked area on a 0.25 m grid, defined by the smallest
cells containing 75% of position observations. Full documented or supplied
coverage areas are reported separately in the density appendix. The separate
ATC calibration artifact's default comparison uses the same 0.25 m / 75%-mass
parameters as that cross-dataset density column, against the documented 900 m2
monitored area, and retains the full swept comparison (including 99%-mass) so
alternative parameterizations stay visible; it is not substituted into the
cross-dataset effective-density column.
