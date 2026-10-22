# Shell launchers

## Purpose

`scripts/` holds the Bash entry points for the two expensive workflows in this repository:
regenerating a dataset end to end, and training a baseline. Each launcher is a thin, auditable
wrapper that resolves the repository root from its own location and calls the Python modules under
[`src/`](../src); the ones that invoke a Python entry point directly also export `PYTHONPATH`, so a
detached job does not depend on the invoking shell. The scientific settings live in the Python entry points and in
[`configs/`](../configs); a launcher only fixes the recipe (which model, which dataset, which seed,
which batch size) used for a particular set of runs.

Setup (virtual environment, editable install, submodules) is documented once in the
[project README](../README.md); do not repeat it here. Every command below assumes the repository
root as the working directory and `.venv` active. The commands can instead be prefixed with
`uv run`, for example `uv run ./scripts/regenerate_canonical_dataset.sh eth 42`.

## Organization

Three groups of files:

- **Dataset regeneration:** `regenerate_canonical_dataset.sh`, `regenerate_dataset_phase1.sh`.
- **Training launchers:** `pretrain_baseline.sh`, `pretrain_transfer_model.sh`,
  `run_transfer_experiment.sh` — these take the full argument set and are what a manual run uses.
- **Detached tmux launchers and their seed workers:** a `*_tmux.sh` script creates a tmux session
  with one window per seed or per model/seed. Each window runs a `*_seed.sh` worker that pins one
  physical GPU, tees a log under `logs/experiments/`, and calls a training launcher or a trainer
  module.
- **TACC Horizon:** `horizon/gpu_jobs.slurm` runs one to four commands on a GB200 node, one GPU
  each; see [`docs/horizon.md`](../docs/horizon.md).

## Usage

### Dataset regeneration

| Script | Purpose | Prerequisites | Representative command | Status |
| --- | --- | --- | --- | --- |
| [`regenerate_canonical_dataset.sh`](regenerate_canonical_dataset.sh) | Run one dataset's converter, then chain into `regenerate_dataset_phase1.sh` for the same dataset and data seed. Accepts `eth`, `ucy`, `atc`, `steps`, `madras`, `grand_central`, `ptdpes`. | Raw downloads already placed in that dataset's `raw/` directory; MADRAS and PTDPES fail fast if `raw/` is missing. | `./scripts/regenerate_canonical_dataset.sh eth 42` | Current |
| [`regenerate_dataset_phase1.sh`](regenerate_dataset_phase1.sh) | Post-conversion pipeline for one dataset: dense 8+12 forecast-window index, OpenTraj analysis view and trajlets, publication artifacts or OpenTraj figures, `stats`, and the consolidated OpenTraj dataset table. | Converted canonical Parquet already present under `datasets/<dataset>/`. | `SKIP_ANALYSIS=1 PREPROCESS_WORKERS=8 ./scripts/regenerate_dataset_phase1.sh steps 42` | Current |

Both take `DATASET [DATA_SEED]` positionally and default the data seed to 42. Worker counts come from
the environment: `ATC_WORKERS` for the converter script,
`PREPROCESS_WORKERS` for the phase-1 script, which also honours `SKIP_FORECAST=1` and
`SKIP_ANALYSIS=1`. Both tee a timestamped log into `logs/preprocess/`.

### Publication artifact regeneration

| Script | Purpose | Prerequisites | Representative command | Status |
| --- | --- | --- | --- | --- |
| [`finish_density_regeneration.sh`](finish_density_regeneration.sh) | Final step of the 0.05 m duplicate-filter change: rebuilds the published dataset-summary artifacts by running `src/opentraj_dataset_stats.py` then `src/paper_dataset_metrics.py`, rewriting `paper/stats/` and the paper tables. | Every dataset's indicator report already carries the `local_density_max_deduplicated` column. | `./scripts/finish_density_regeneration.sh` | Current |

The script's header calls it safe to re-run: it only reads indicator parquets and rewrites
`paper/stats/` plus the paper tables, so a repeat run does not accumulate side effects. It aborts on the
first failing step (`set -e`, and the second command's failure is checked and re-raised explicitly), so
a partial rewrite is reported as a non-zero exit rather than a logged warning. As of this writing it
cannot reach `=== FINISHED ===`: `src/paper_dataset_metrics.py` rejects the PTDPES publication artifact
on an `indicator_implementation_hash` mismatch (tracked as issue L-003), so running this script
surfaces that abort. Re-run once L-003's remedy (a PTDPES indicator re-run or a metadata refresh, per
that issue) has landed.

### Training launchers

| Script | Purpose | Prerequisites | Representative command | Status |
| --- | --- | --- | --- | --- |
| [`pretrain_baseline.sh`](pretrain_baseline.sh) | Single-dataset pretraining for `trajectron`, `moflow`, `agentformer`, `socialvae` on `atc`, `ptdpes`, or `steps`; registers the finished checkpoint. | GPU; converted dataset and scene index. Trajectron++ builds its shard cache on first use; pre-build it with `python -m scripts.cache_trajectron_datasets` to avoid a long first run. | `./scripts/pretrain_baseline.sh --model trajectron --dataset ptdpes --seed 0` | Current |
| [`pretrain_transfer_model.sh`](pretrain_transfer_model.sh) | Stage-aware transfer pretraining for `moflow` and `socialvae` over one or more `--sources` datasets, under an explicit `--run-version` (default `transfer_v1`). Refuses to overwrite an existing run directory unless `--resume` is passed. | GPU; converted source datasets. | `./scripts/pretrain_transfer_model.sh --model moflow --sources ptdpes --seed 0` | Current |
| [`run_transfer_experiment.sh`](run_transfer_experiment.sh) | Downstream training *and* evaluation for one `moflow`/`socialvae` condition: `--stage finetune` or `--stage direct` (both require `--checkpoint`; `direct` skips training and evaluates the checkpoint as-is, so `--fraction` must be unset or `1.0`) or `--stage scratch`, on `--target eth_ucy` (with `--fold`), `grand_central`, or `madras`, at a given `--fraction`. `--seed` is optional for `finetune`/`direct` and defaults to the pretrained checkpoint's registered/config seed; it is still required for `scratch`. Refuses to overwrite an existing run's training or evaluation output unless `--resume` is passed. | GPU; converted target dataset; for `finetune`/`direct`, a registered pretrained checkpoint. | `./scripts/run_transfer_experiment.sh --model moflow --stage scratch --target eth_ucy --fold eth --fraction 0.25 --seed 1` | Current |

`--resume`/`--resuming` is accepted by all three and forwarded to the trainer; without a path it
resumes `RUN_DIR/checkpoints/latest.pt`. The root README's "Copy-paste pretraining commands" section
is the authoritative description of the pretraining and resume workflow.

### Detached tmux launchers

| Script | Purpose | Default session | Worker it runs | Prerequisites | Representative command | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [`pretrain_ptdpes_transfer_tmux.sh`](pretrain_ptdpes_transfer_tmux.sh) | Start MoFlow and SocialVAE PTDPES pretraining for seeds 0, 1, 2 in a new session, one window per model/seed, seed `s` pinned to GPU `s % gpu_count`. | `ptdpes-transfer-v2` | `pretrain_ptdpes_transfer_seed.sh` | `tmux`, `nvidia-smi`, at least one visible GPU, and no session of that name yet. | `./scripts/pretrain_ptdpes_transfer_tmux.sh ptdpes-transfer-v2` | Current |
| [`add_trajectron_ptdpes_tmux.sh`](add_trajectron_ptdpes_tmux.sh) | Add Trajectron++ seeds 0, 1, 2 at batch size 128 to an **already existing** session (seed `s` to GPU `s`), plus a `gpu-monitor` window logging one-minute samples to `logs/experiments/<session>-gpu.log`. | `ptdpes-transfer-v2` | `pretrain_trajectron_ptdpes_seed.sh` | The named session must already exist, three GPUs must be visible, and no window may collide. | `./scripts/add_trajectron_ptdpes_tmux.sh ptdpes-transfer-v2` | Current |

Each launcher takes the session name as its single optional positional argument, defaulting as
listed. Attach with `tmux attach-session -t SESSION_NAME`. The first two share the default
`ptdpes-transfer-v2` by design: `pretrain_ptdpes_transfer_tmux.sh` creates that session and
`add_trajectron_ptdpes_tmux.sh` adds windows to it, so run them in that order.

### Seed workers (internal)

These two are invoked by the tmux launchers above, one process per seed per GPU. They can be run
directly for a single seed, but they are not the documented entry point for a full experiment.

| Worker | Invoked by | Arguments |
| --- | --- | --- |
| [`pretrain_ptdpes_transfer_seed.sh`](pretrain_ptdpes_transfer_seed.sh) | `pretrain_ptdpes_transfer_tmux.sh` | `MODEL SEED GPU_INDEX`; calls `pretrain_transfer_model.sh` with `--run-version transfer_v2` |
| [`pretrain_trajectron_ptdpes_seed.sh`](pretrain_trajectron_ptdpes_seed.sh) | `add_trajectron_ptdpes_tmux.sh` | `SEED GPU_INDEX`; calls `baselines.train_trajectron` and registers the result |

## Conventions and constraints

- **Every training launcher here starts expensive GPU work,** and the regeneration scripts run the
  full converter plus analysis pipeline over whole recordings. Nothing in this directory is a dry
  run or a cheap check.
- Launchers resolve `PROJECT_DIR` from `BASH_SOURCE`, so they work from any working directory, but
  `pretrain_baseline.sh` resolves its default `--output` (`experiments/pretrained`) against the
  *caller's* working directory. Run from the repository root.
- `PYTHON_BIN` overrides the interpreter in the regeneration scripts, in `pretrain_baseline.sh`,
  `pretrain_transfer_model.sh`, `run_transfer_experiment.sh`, and in both seed workers; it defaults
  to whatever `python` the active environment provides.
- Model tags are not always the model name: `pretrain_baseline.sh` registers MoFlow and SocialVAE
  under a `_transfer_v1` tag, and `pretrain_transfer_model.sh` uses the `--run-version` value
  (`transfer_v2` for the current detached runs). Expect run directories such as
  `experiments/pretrained/moflow_transfer_v2_ptdpes_0`.
- Resume handling: `pretrain_ptdpes_transfer_seed.sh` and `pretrain_trajectron_ptdpes_seed.sh`
  skip a completed run, pass `--resume` when `latest.pt` exists, and exit 3 on unresumable run
  metadata.

## Related documentation

- [Project README](../README.md) — setup, task definition, copy-paste pretraining commands, resume
  semantics, and the current detached-run workflow.
- [`src/baselines/README.md`](../src/baselines/README.md) — model adapters, trainers, and evaluation.
- [`src/datasets_preprocessing/README.md`](../src/datasets_preprocessing/README.md) — converter
  contract for the datasets the regeneration scripts drive.
- [`configs/README.md`](../configs/README.md) — task, transfer, and protocol configuration.
- [`experiments/README.md`](../experiments/README.md) — run directories, manifests, and the model
  registry these launchers write into.
