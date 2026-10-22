# Experiments

This directory is the home for training and evaluation artifacts.

- `pretrained/`, `downstream/`, and other run directories contain checkpoints, curves,
  copied parameters, and per-run dataset snapshots. These large artifacts are gitignored.
- `model_registry/` maps reusable model IDs to local checkpoints and is gitignored.
- `manifests/` records exact seeded track/sample selections. These small reproducibility
  records are version controlled.
- `results/runs.parquet` is the consolidated machine-readable results table; it holds the
  constant-velocity/paper baseline rows written by `src/paper/generate_tables.py`. Transfer
  and downstream results are consolidated per condition separately, by
  `python -m baselines.aggregate_transfer`.

A manifest is not a configuration preset: it records the concrete population realized
after applying a configuration and seed, allowing that exact subset to be reconstructed.
