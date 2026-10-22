# Pedestrian trajectory data

**STEPS** — *Scene Trajectories of Everyday Pedestrians* — is an automated pipeline that turns raw
pedestrian recordings into a common trajectory format, together with the forecasting baselines,
evaluation, and dataset analysis built on its output. The premise is that what limits forecasting
and social-navigation research is not architecture but data spanning time, place, and density, and
that the way to obtain it is a pipeline cheap enough to run repeatedly rather than an annotation
campaign. STEPS is the first instance: one fixed university campus scene, an indoor atrium,
recorded across months at varying densities. The pipeline is the durable
contribution; the dataset is what it produces, and it should keep producing. See
[`research-plan.md`](research-plan.md) for the research goals, the fixed data contract, and the
open questions.

Common conversion and storage for STEPS, ATC, ETH, UCY, MADRAS,
PTDPES, and Grand Central.

Clone the repository with its pinned upstream baseline implementations:

```bash
git clone --recurse-submodules https://github.com/michaelmunje/stars_steps_extension.git
cd stars_steps_extension
```

For an existing clone made without submodules, run
`git submodule update --init --recursive` once.

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then create the
locked project environment (including the default development dependencies) with:

```bash
uv sync
```

`uv` uses the Python version in [`.python-version`](.python-version), creates `.venv/`,
and installs the project editably. [`uv.lock`](uv.lock) pins the complete dependency
graph; use `uv sync --locked` when the lockfile must not change. The editable install
exposes the modules under `src/` while keeping datasets and generated artifacts at the
repository root.

Run one-off commands inside the environment with `uv run`, for example:

```bash
uv run python -m pytest
uv run python -m preprocess datasets/eth --data-seed 42
```

Alternatively, run `source .venv/bin/activate` once per shell and use the existing
`python -m ...` examples below unchanged. Shell launchers under `scripts/` also set the
source path themselves, so background jobs do not depend on the invoking shell.

`requirements.txt` is retained as a pip-compatible list of the direct runtime
dependencies. Keep it in sync with `[project.dependencies]` when changing dependencies;
`pyproject.toml` and `uv.lock` are authoritative for the `uv` workflow.

Large raw and generated data belongs under the gitignored `datasets/` directory.
Converters, lightweight source metadata, split templates, and dataset-specific audit
utilities are committed under `src/datasets_preprocessing/`.

On TACC Horizon (aarch64 GB200 nodes, Slurm, no internet on compute nodes), follow
[`docs/horizon.md`](docs/horizon.md) for setup, dataset transfer and job submission. The
multi-agent STARS forecaster and its training phases are described in
[`docs/stars_multiagent.md`](docs/stars_multiagent.md).

## Task

Given each person's past BEV positions, predict their future BEV positions while using the other visible people as context.

- Input: `x, y`
- Output: future `x, y`
- Units: metres
- Canonical frame: dataset-specific metric world coordinates; model coordinate
  transforms are derived at sample-loading time and never written back to canonical data
- Task window: 8 history steps and 12 future steps at 2.5 Hz

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

## Layout

```text
.
├── README.md
├── .python-version      # Python version selected by uv
├── pyproject.toml        # editable-install and src-layout metadata
├── uv.lock               # reproducible Python dependency graph
├── requirements.txt      # pip-compatible direct runtime dependencies
├── configs/
│   ├── task.json         # analysis and forecasting task definition
│   ├── transfer.yaml     # transfer-learning defaults
│   └── protocols/        # fixed paper and benchmark membership policies
├── src/
│   ├── data.py          # splits and PyTorch loader
│   ├── baselines/       # model adapters, training, and evaluation
│   ├── paper/           # publication table generators
│   ├── scripts/         # Python command utilities
│   └── datasets_preprocessing/
│       ├── README.md    # adapter-folder contract
│       ├── atc/
│       ├── eth/
│       └── ...
├── scripts/             # shell launchers
├── references/          # unmodified upstream submodules plus their pinned-commit records
├── datasets/            # local raw/converted/indexed data (gitignored)
├── experiments/         # checkpoints, run outputs, manifests, and aggregate results
├── logs/                 # preprocessing and detached-job logs
├── paper/
│   ├── figures/          # generated visualizations
│   └── stats/            # machine-readable publication statistics
└── tests/
```

Folder READMEs cover local conventions in more depth: [`configs/README.md`](configs/README.md),
[`experiments/README.md`](experiments/README.md), [`scripts/README.md`](scripts/README.md),
[`src/baselines/README.md`](src/baselines/README.md),
and [`src/datasets_preprocessing/README.md`](src/datasets_preprocessing/README.md) with one README per
dataset adapter.

Files under `configs/protocols/` define fixed scientific policies, such as which
sequences form an ETH/UCY fold or a publication-table row. Experiment manifests are
different: they record the exact tracks or samples selected by a seeded run so that the
same subset can be reproduced. Those live under `experiments/manifests/`; consolidated
metrics live under `experiments/results/`, while individual run directories live directly
under `experiments/`.

Downloads are placed manually in each `raw/` directory. Each converted file represents one recording, and all conversion code stays with its dataset.

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
[`src/datasets_preprocessing/steps/README.md`](src/datasets_preprocessing/steps/README.md).

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
([`configs/protocols/paper_datasets.json`](configs/protocols/paper_datasets.json)).

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

All three commands read the canonical storage frequency from [`src/canonical.py`](src/canonical.py).

The main density column uses one denominator for every dataset: the median
per-recording effective walked area on a 0.25 m grid, defined by the smallest
cells containing 75% of position observations. Full documented or supplied
coverage areas are reported separately in the density appendix. The separate
ATC calibration artifact's default comparison uses the same 0.25 m / 75%-mass
parameters as that cross-dataset density column, against the documented 900 m2
monitored area, and retains the full swept comparison (including 99%-mass) so
alternative parameterizations stay visible; it is not substituted into the
cross-dataset effective-density column.

## Forecasting baselines

The integrated baselines are Trajectron++ (K=20), SocialVAE (K=20), MoFlow (K=20), and
AgentFormer VAE+DLow (K=20). CUDA is the default and CPU is
used only when explicitly requested.

SocialVAE and MoFlow use explicit, stage-aware transfer recipes. Pretraining and
fine-tuning never share optimizer or scheduler state, all parameters are fine-tuned,
and scratch/pretrained target runs use identical fractions and target optimization.
The completed September 9 SocialVAE/MoFlow PTDPES runs predate this protocol and are
legacy diagnostics; rerun them before using transfer results in the paper.

### Copy-paste pretraining commands

Run these from the repository root. Change only the final seed number.
ATC currently has converted trajectories but still needs its seed-42 scene index;
run `python -m preprocess datasets/atc --data-seed 42` once before its commands.

Trajectron++:

```bash
./scripts/pretrain_baseline.sh --model trajectron --dataset atc --seed 0 --batch-size 512
./scripts/pretrain_baseline.sh --model trajectron --dataset ptdpes --seed 0 --batch-size 512
./scripts/pretrain_baseline.sh --model trajectron --dataset steps --seed 0 --batch-size 512
```

SocialVAE:

```bash
./scripts/pretrain_transfer_model.sh --model socialvae --sources atc --seed 0
./scripts/pretrain_transfer_model.sh --model socialvae --sources ptdpes --seed 0
./scripts/pretrain_transfer_model.sh --model socialvae --sources steps --seed 0
```

MoFlow:

```bash
./scripts/pretrain_transfer_model.sh --model moflow --sources atc --seed 0
./scripts/pretrain_transfer_model.sh --model moflow --sources ptdpes --seed 0
./scripts/pretrain_transfer_model.sh --model moflow --sources steps --seed 0
```

AgentFormer:

```bash
./scripts/pretrain_baseline.sh --model agentformer --dataset atc --seed 0 --batch-size 1
./scripts/pretrain_baseline.sh --model agentformer --dataset ptdpes --seed 0 --batch-size 1
./scripts/pretrain_baseline.sh --model agentformer --dataset steps --seed 0 --batch-size 1
```

Each command pretrains on the selected dataset using CUDA, saves all logs/manifests/curves and
parameters under `experiments/pretrained/`, and registers a reusable ID such as
`trajectron_ptdpes_0` or `moflow_transfer_v1_atc_0`. AgentFormer automatically runs and registers both its VAE and
DLow stages. Optional overrides include `--epochs`, `--device`, `--data-seed`,
`--batch-size`, `--num-workers`, `--iterations`, `--patience`, `--min-delta`,
`--output`, and `--python`. SocialVAE pretraining defaults to 600 epochs × 100
batches, Adam at `5e-4`, and no early stopping. MoFlow pretraining defaults to
150 epochs, AdamW at `1e-4`, batch 48, 5% warmup plus cosine decay, and no
early stopping.

`--iterations` provides a fixed optimizer-step budget independent of dataset size. Increase
the batch size until GPU memory is well utilized; 128 is a practical starting point, while
256 is worth trying if it fits. Keep the chosen batch size and iteration budget identical
across comparable experiment arms. The validation cap applies only during checkpoint
selection; held-out evaluation still uses the complete test split. AgentFormer retains
the upstream one-scene batching.

Measured on the default 11 GB RTX 2080 Ti, the largest useful batches were:

| Model | Recommended batch | Observed result |
|---|---:|---|
| Trajectron++ | 512 | 1024 fit but provided essentially no throughput improvement over 512 |
| SocialVAE | 128 | Official batch size; increase only after checking GPU memory and validation parity |
| MoFlow | 48 for the literature protocol | Larger batches are explicitly labeled throughput ablations |
| AgentFormer VAE/DLow | 1 scene | Required by the upstream scene-oriented data/model contract |

Trajectron++ caches fully prepared interaction graphs as bounded temporal shards
under each dataset's `.cache/trajectron/` directory. The first run can spend substantial
time building them; later model seeds and reruns load them directly. Writes are atomic,
locked, resumable, and use streaming Zstandard compression. Training loads one shard at
a time and deterministically caps its examples to the accepted unified-index count. Each
shard contains at most two overlapping 100-timestep scenes, so a dense or long recording
cannot create an unbounded in-memory environment. This also prevents native expansion to
tens of millions of examples.

Build caches for every currently available dataset before training with:

```bash
python -m scripts.cache_trajectron_datasets --datasets all --data-seed 42
```

Use `--datasets PTDPES,atc` for selected datasets. Add `steps` after its converted files and
scene index exist. Completed temporal shards are skipped automatically.

For the current detached PTDPES transfer-pretraining runs, one launcher starts
MoFlow and SocialVAE for seeds 0, 1, and 2. It pins the two models for a seed to
the same physical GPU and persists one terminal log per model/seed:

```bash
./scripts/pretrain_ptdpes_transfer_tmux.sh ptdpes-transfer-v2
./scripts/add_trajectron_ptdpes_tmux.sh ptdpes-transfer-v2
tmux attach-session -t ptdpes-transfer-v2
tail -f logs/experiments/moflow-transfer_v2-ptdpes-seed0.log
tail -f logs/experiments/socialvae-transfer_v2-ptdpes-seed0.log
```

Attach to a still-running session with `tmux attach -t SESSION_NAME`.

MoFlow and SocialVAE checkpoints contain model, optimizer, scheduler,
early-stopping, RNG, epoch, within-epoch batch, and DataLoader generator state.
Restart an interrupted run with the identical command plus `--resume` (or its
alias `--resuming`); without a path it uses `RUN_DIR/checkpoints/latest.pt`:

```bash
./scripts/pretrain_transfer_model.sh --model moflow --sources ptdpes --seed 0 --run-version transfer_v2 --resume
./scripts/run_transfer_experiment.sh --model moflow --stage finetune --checkpoint moflow_transfer_v2_ptdpes_0 --target eth_ucy --fold eth --seed 0 --resume
```

Resume validates the original model, stage, seeds, data sources, preprocessing,
and optimization recipe. It refuses a changed contract rather than silently
continuing a different experiment. Transfer initialization remains separate:
fine-tuning loads pretrained model weights but starts a fresh optimizer and
scheduler, while `--resume` restores an interrupted stage's optimizer and
scheduler.

The Trajectron++ launcher adds seeds 0--2 to the same tmux session at batch 128.
Its worker reuses the bounded PTDPES shard cache, automatically resumes an
unfinished `transfer_v2` checkpoint, and skips a completed run. A `gpu-monitor`
window records one-minute VRAM/utilization samples in
`logs/experiments/ptdpes-transfer-v2-gpu.log`.

Pretraining and downstream training are disjoint. Completed pretrained models
are registered under stable IDs:

```text
trajectron_ptdpes_0
socialvae_transfer_v1_ptdpes_0
moflow_transfer_v1_ptdpes_0
agentformer_vae_ptdpes_0
agentformer_dlow_ptdpes_0
```

The same pretrained ID can be supplied to any number of downstream runs:

```bash
python -m baselines.train_trajectron \
  --stage finetune --fold eth --checkpoint trajectron_ptdpes_0 \
  --data-seed 42 --device cuda \
  --run-dir experiments/downstream/trajectron_ptdpes_0/eth

python -m baselines.train_trajectron \
  --stage finetune --fold zara1 --checkpoint trajectron_ptdpes_0 \
  --data-seed 42 --device cuda \
  --run-dir experiments/downstream/trajectron_ptdpes_0/zara1
```

SocialVAE/MoFlow ETH/UCY target training reserves one complete non-test scene
sequence for validation by default. `best.pt` is selected by target validation
ADE/minADE with early stopping; the held-out test fold is never used for
checkpoint selection.

Set `TRAJECTORY_MODEL_REGISTRY` when using a non-default registry directory.
Inspect or resolve registered models with:

```bash
python -m baselines.registry list
python -m baselines.registry resolve trajectron_ptdpes_0
```

Checkpoint arguments accept a registered ID, a checkpoint file, or a run
directory containing `checkpoints/best.pt`.

Every run contains:

- `config.json` and atomic `status.json`
- append-only `training.log` and `metrics.jsonl`
- `datasets/train.parquet`, `val.parquet`, or `test.parquet`; streamed
  Trajectron++ pretraining additionally records the complete
  `*_trajectron_shards.parquet` population and realized per-shard budgets
- copied `configs/task.json`, the five-fold policy, and hashes of source scene indices
- `checkpoints/latest.pt` and `checkpoints/best.pt`
- `curves/training_curves.png` and `training_curves.pdf`
- `evaluation.json` for evaluation runs
- `visuals/scene_XX.png` and `.pdf`, `contact_sheet.png`, and per-scene metrics

Dataset preparation stages are printed immediately, and each training epoch has a live
batch progress bar with iteration and loss. The same lifecycle events and periodic metrics
are persisted in `training.log`, `status.json`, and `metrics.jsonl`, so progress remains
inspectable when a job is detached or its terminal output is unavailable.

Evaluation renders 10 deterministic held-out scenes by default. Set
`--visual-scenes N` on an evaluator; values are capped at
30. Each plot shows observed and context trajectories, future ground truth,
constant velocity, and the model forecast. The legend reports ADE/FDE for both
CV and the displayed model trajectory. For K=20 models, the displayed trajectory
is the sample with the best ADE, and its matching FDE is reported.

### Individual commands

For SocialVAE or MoFlow, one command trains one target condition and immediately
evaluates its selected checkpoint. The same command with `--stage scratch`
omits `--checkpoint` and uses the identical target fraction, split, maximum
epochs, checkpoint metric, and early-stopping rule:

```bash
./scripts/run_transfer_experiment.sh --model moflow --stage finetune --checkpoint moflow_transfer_v1_ptdpes_1 --target eth_ucy --fold eth --fraction 0.25 --seed 1
./scripts/run_transfer_experiment.sh --model moflow --stage scratch --target eth_ucy --fold eth --fraction 0.25 --seed 1
./scripts/run_transfer_experiment.sh --model socialvae --stage finetune --checkpoint socialvae_transfer_v1_ptdpes_1 --target madras --fraction 1.0 --seed 1
```

Target fractions select namespaced source trajectories, retain all selected
training windows, are nested for a fixed data seed, and never alter validation
or test. Their persistent manifests live in `experiments/manifests/target_fractions/` and a
copy is saved with every run. Use `--transfer-config configs/transfer.yaml` or
individual overrides such as `--learning-rate`, `--epochs`, and `--batch-size`.
Python pretraining entry points also accept multiple sources, for example
`--pretrain-datasets ptdpes steps`; the wrapper equivalent is
`--sources ptdpes,steps`.
For an explicitly declared fine-tuning LR sweep, use `--learning-rate` with
SocialVAE values `5e-5`, `1e-4`, or `2e-4`, or MoFlow values `1e-5`, `2e-5`,
`5e-5`, or `1e-4`; the defaults remain `1e-4` and `2e-5`, respectively.

Aggregate matched seed results with standard error across independent model
seeds (not across overlapping windows):

```bash
python -m baselines.aggregate_transfer \
  experiments/downstream/finetune/moflow/moflow_transfer_v1_ptdpes_0/seed0/madras/fraction-1.0/test \
  experiments/downstream/finetune/moflow/moflow_transfer_v1_ptdpes_1/seed1/madras/fraction-1.0/test \
  experiments/downstream/finetune/moflow/moflow_transfer_v1_ptdpes_2/seed2/madras/fraction-1.0/test \
  --output experiments/downstream/finetune/moflow/ptdpes-to-madras-fraction-1.0.json
```

Each model uses the same stage convention:

```bash
# PTDPES pretraining
python -m baselines.train_moflow \
  --stage pretrain --seed 1 --data-seed 42 --device cuda \
  --run-dir experiments/moflow/seed_1/pretrain

# ETH/UCY fold fine-tuning from that seed's pretrained model
python -m baselines.train_moflow \
  --stage finetune --fold eth --seed 1 --data-seed 42 --device cuda \
  --checkpoint experiments/moflow/seed_1/pretrain/checkpoints/best.pt \
  --run-dir experiments/moflow/seed_1/eth/finetune

# Transfer training reserves one complete sequence from the four-fold fitting pool
# for validation; the held-out fold remains test-only.

# Held-out ETH evaluation
python -m baselines.evaluate_moflow \
  --fold eth --data-seed 42 --device cuda \
  --visual-scenes 20 \
  --checkpoint experiments/moflow/seed_1/eth/finetune/checkpoints/best.pt \
  --run-dir experiments/moflow/seed_1/eth/finetune_eval
```

Replace `moflow` with `trajectron` or `social_vae`. AgentFormer has separate VAE and DLow stages; exact
commands and checkpoint relationships are documented in
[`src/baselines/README.md`](src/baselines/README.md).

Aggregate any independently launched evaluation runs with:

```bash
python -m baselines.aggregate \
  --output experiments/moflow/eth_finetune_summary.json \
  experiments/moflow/seed_1/eth/finetune_eval \
  experiments/moflow/seed_2/eth/finetune_eval \
  experiments/moflow/seed_3/eth/finetune_eval
```

## Tests

Run the focused suite from the repository root:

```bash
uv run python -m pytest
```

Expensive model-training tests are marked `slow`, and [`pytest.ini`](pytest.ini) sets
`addopts = -m "not slow"`. **A bare `python -m pytest` therefore no longer runs
everything**, and neither does a targeted invocation: `python -m pytest
tests/test_train_moflow.py` reports green after running only that file's cheap
argument-parser test. Run the marked tests explicitly:

```bash
uv run python -m pytest -m slow                     # only the expensive tests
uv run python -m pytest -m "slow or not slow"       # everything, in one pass
```

The `slow` tests build the production-sized MoFlow model and train it on CPU against a
tiny synthetic dataset (roughly a minute in total). Nothing runs them automatically, so
run them yourself after changing [`src/baselines/train_moflow.py`](src/baselines/train_moflow.py)
or [`src/baselines/moflow.py`](src/baselines/moflow.py).

## References

Sources for the datasets this repository converts and the upstream implementations it pins. Each
link is the source recorded in the corresponding adapter README or in [`.gitmodules`](.gitmodules);
where the repository records no published source, the adapter README documents the raw files the
converter expects instead.

| Dataset | Recorded source |
| --- | --- |
| ATC | <https://dil.atr.jp/ISL/crest2010_HRI/ATC_dataset/> — [adapter](src/datasets_preprocessing/atc/README.md) |
| ETH / BIWI Walking Pedestrians | <https://vision.ee.ethz.ch/datsets/index.en.html> — [adapter](src/datasets_preprocessing/eth/README.md) |
| MADRAS | <https://doi.org/10.5281/zenodo.13830435> — [adapter](src/datasets_preprocessing/madras/README.md) |
| UCY Crowds-by-Example | supplied RAR archives; no source URL recorded in this repository — [adapter](src/datasets_preprocessing/ucy/README.md) |
| Grand Central | supplied annotation text files; no source URL recorded in this repository — [adapter](src/datasets_preprocessing/grand_central/README.md) |
| PTDPES | published final modified trajectory CSVs; no source URL recorded in this repository — [adapter](src/datasets_preprocessing/PTDPES/README.md) |
| STEPS | recorded for this project from local tracker exports — [adapter](src/datasets_preprocessing/steps/README.md) |

Reference implementations are pinned as git submodules under `references/` and are used unmodified:

| Project | Upstream repository |
| --- | --- |
| AgentFormer | <https://github.com/Khrylx/AgentFormer> |
| MoFlow | <https://github.com/DSL-Lab/MoFlow> |
| Trajectron++ | <https://github.com/StanfordASL/Trajectron-plus-plus> |
| TrajNet++ baselines / tools | <https://github.com/vita-epfl/trajnetplusplusbaselines>, <https://github.com/vita-epfl/trajnetplusplustools> |
| Social GAN | <https://github.com/agrimgupta92/sgan> |
| Schöller constant-velocity baseline | <https://github.com/cschoeller/constant_velocity_pedestrian_motion> |
| SocialVAE | <https://github.com/xupei0610/SocialVAE> |
| OpenTraj | <https://github.com/crowdbotp/OpenTraj> (pinned commit and MIT licence recorded in [`references/opentraj.json`](references/opentraj.json)) |
