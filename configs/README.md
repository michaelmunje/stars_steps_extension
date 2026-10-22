# Configuration

This directory contains version-controlled inputs that define how the pipeline behaves:

- `task.json` defines the analysis and forecasting task, including sampling frequency,
  window lengths, and dataset-specific stride policy.
- The canonical trajectory schema (`trajectory.json` was retired) lives solely in
  [`src/canonical.py`](../src/canonical.py) (`CANONICAL_SCHEMA_VERSION`, `CANONICAL_HZ`,
  `CANONICAL_SCHEMA`); every preprocessing, analysis, and run-status consumer imports these
  constants directly instead of reading a config file.
- `transfer.yaml` defines pretraining and fine-tuning defaults.
- `protocols/` contains fixed scientific membership policies, such as publication-table
  datasets and ETH/UCY held-out folds.

Protocols are configuration, not generated results. Changing one changes the scientific
population or evaluation policy and should therefore be reviewed like a code change.
