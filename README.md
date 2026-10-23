# STEPS trajectory forecasting

**STEPS** (*Scene Trajectories of Everyday Pedestrians*) is an automated pipeline that turns raw
pedestrian tracker recordings into a common trajectory format, plus the forecasting baselines,
evaluation and dataset analysis built on it. The current release is one indoor university atrium
recorded over 19 days (198 recordings). This repository also extends STARS into a multi-agent
forecaster and simulator ([`docs/stars_multiagent.md`](docs/stars_multiagent.md)).

Datasets: STEPS (`steps`), ETH, UCY, ATC, MADRAS, Grand Central and PTDPES.

## Quick start

```bash
git clone --recurse-submodules https://github.com/michaelmunje/stars_steps_extension.git
cd stars_steps_extension
uv sync                                  # creates .venv from uv.lock
uv run python -m pytest -q               # fast suite; slow tests: -m slow
```

Datasets live in the gitignored `datasets/` directory. On TACC Horizon, copy them from the shared
read-only location (use `rsync -a` so file times are kept):

```bash
rsync -a /scratch/09865/mmunje/shared/datasets/ datasets/          # all, 23 GB
rsync -a /scratch/09865/mmunje/shared/datasets/steps datasets/     # STEPS only, 462 MB
```

Running on Horizon (aarch64 GB200 nodes, Slurm, no internet on compute nodes):
[`docs/horizon.md`](docs/horizon.md).

## Task

Given each pedestrian's past positions, predict their future positions, using the other visible
people as context.

- Input and output: `x, y` in metres, in each dataset's metric world frame
- Window: 8 observed + 12 future steps at 2.5 Hz (3.2 s → 4.8 s)
- Canonical Parquet is never modified; smoothing, normalization and windowing are derived at load
  time

## Common commands

```bash
# Index a converted dataset (splits, categories, sizing)
python -m preprocess datasets/steps --name steps --data-seed 42

# Multi-agent STARS: self-supervised encoder, then forecaster (see docs/stars_multiagent.md)
python -m baselines.train_stars_ma --phase ssl --dataset steps --run-dir experiments/stars_ma/ssl
python -m baselines.train_stars_ma --phase forecast --dataset steps --decoder cvae \
  --encoder-checkpoint experiments/stars_ma/ssl/checkpoints/best.pt --encoder-mode finetune \
  --run-dir experiments/stars_ma/cvae_finetune

# A baseline: pretrain MoFlow on STEPS (see docs/baselines.md)
./scripts/pretrain_transfer_model.sh --model moflow --sources steps --seed 0
```

Prefix with `uv run`, or `source .venv/bin/activate` once per shell.

## Layout

```text
configs/        task definition, transfer defaults, fixed paper/benchmark protocols
src/            data.py (splits, loader), baselines/ (models, training, evaluation),
                datasets_preprocessing/ (one converter per dataset), paper/ (tables)
scripts/        shell launchers; scripts/horizon/ for Slurm
references/     pinned, unmodified upstream implementations (git submodules)
experiments/    manifests and aggregate results (run outputs are gitignored)
paper/          generated tables and statistics
docs/           guides (below)
tests/
```

## Documentation

| Topic | Where |
| --- | --- |
| **Latest forecaster** (results, figures) | [`docs/latest_forecaster/`](docs/latest_forecaster/README.md) |
| **Latest navigation policy** (results, animations) | [`docs/latest_policy/`](docs/latest_policy/README.md) |
| All results and the reasoning behind them | [`docs/results/README.md`](docs/results/README.md), [`docs/decisions.md`](docs/decisions.md) |
| SEAN-JAX simulator and policy tuning | [`seanjax/README.md`](seanjax/README.md) |
| TACC Horizon setup and jobs | [`docs/horizon.md`](docs/horizon.md) |
| Multi-agent STARS design, phases, ablations | [`docs/stars_multiagent.md`](docs/stars_multiagent.md) |
| Loader, preprocessing, OpenTraj analysis, paper tables, visualization | [`docs/data_pipeline.md`](docs/data_pipeline.md) |
| Forecasting baselines: pretraining, transfer, evaluation | [`docs/baselines.md`](docs/baselines.md), [`src/baselines/README.md`](src/baselines/README.md) |
| Shell launchers | [`scripts/README.md`](scripts/README.md) |
| Dataset converters | [`src/datasets_preprocessing/README.md`](src/datasets_preprocessing/README.md) |
| STEPS release and raw fields | [`docs/steps_release.md`](docs/steps_release.md), [`docs/steps_raw_fields.md`](docs/steps_raw_fields.md) |

## Tests

`uv run python -m pytest` runs the fast suite. Training tests marked `slow` are excluded by
[`pytest.ini`](pytest.ini); run them with `-m slow` after changing MoFlow code.

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