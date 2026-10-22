#!/bin/bash
# Final step of the 0.05 m duplicate-filter change (revise cycle
# 2026-09-11-local-density): once every dataset's indicator report carries the
# `local_density_max_deduplicated` column, rebuild the published summary
# artifacts. Safe to re-run; it only reads indicator parquets and rewrites
# paper/stats + paper tables.
set -e
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python}"
echo "=== opentraj_dataset_stats $(date -Is) ==="
"$python_bin" src/opentraj_dataset_stats.py
echo "=== paper_dataset_metrics $(date -Is) ==="
"$python_bin" src/paper_dataset_metrics.py || { rc=$?; echo "paper_dataset_metrics failed with $rc (see output above)"; exit "$rc"; }
echo "=== FINISHED $(date -Is) ==="
