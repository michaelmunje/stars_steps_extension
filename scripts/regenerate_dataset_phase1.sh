#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: regenerate_dataset_phase1.sh DATASET [DATA_SEED]}"
data_seed="${2:-42}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
preprocess_workers="${PREPROCESS_WORKERS:-8}"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"

case "${dataset,,}" in
  ptdpes) dataset_name="ptdpes"; canonical_dataset_name="PTDPES"; dataset_dir="$project_dir/datasets/PTDPES" ;;
  atc) dataset_name="atc"; dataset_dir="$project_dir/datasets/atc" ;;
  *) dataset_name="${dataset,,}"; canonical_dataset_name="${dataset,,}"; dataset_dir="$project_dir/datasets/${dataset,,}" ;;
esac
canonical_dataset_name="${canonical_dataset_name:-$dataset_name}"

mkdir -p "$project_dir/logs/preprocess"
if [[ -z "${STEPS_PHASE1_LOGGED:-}" ]]; then
  exec > >(tee -a "$project_dir/logs/preprocess/${dataset_name}-$(date +%Y%m%dT%H%M%S).log") 2>&1
fi

cd "$project_dir"
echo "[$(date --iso-8601=seconds)] preprocessing $dataset_name seed=$data_seed"
if [[ "${SKIP_FORECAST:-0}" != "1" ]]; then
  echo "[$(date --iso-8601=seconds)] building dense 8+12 forecast-window index"
  "$python_bin" -m preprocess "$dataset_dir" --name "$dataset_name" --data-seed "$data_seed" --workers "$preprocess_workers"
else
  echo "[$(date --iso-8601=seconds)] reusing existing forecast-window index"
fi
if [[ "${SKIP_ANALYSIS:-0}" != "1" ]]; then
  echo "[$(date --iso-8601=seconds)] building OpenTraj analysis view and exact 4.8 s trajlets"
  "$python_bin" -m analysis_pipeline "$dataset_dir" --data-seed "$data_seed" --workers "$preprocess_workers"
  if "$python_bin" - <<PY
from paper_dataset_metrics import load_paper_datasets
raise SystemExit(0 if any(source.canonical_dataset == "$canonical_dataset_name" for row in load_paper_datasets() for source in row.sources) else 1)
PY
  then
    echo "[$(date --iso-8601=seconds)] building publication artifacts, indicators, and figures"
    "$python_bin" -m scripts.prepare_publication_analysis --datasets "$canonical_dataset_name" --workers "$preprocess_workers"
  else
    echo "[$(date --iso-8601=seconds)] computing OpenTraj-compatible indicators and figures"
    "$python_bin" -m opentraj_report "$dataset_dir" --plot-dir "$project_dir/paper/figures/opentraj/$dataset_name"
  fi
fi
echo "[$(date --iso-8601=seconds)] computing statistics $dataset_name"
"$python_bin" -m stats "$dataset_dir" --name "$dataset_name" --data-seed "$data_seed"
echo "[$(date --iso-8601=seconds)] refreshing consolidated OpenTraj dataset table"
"$python_bin" -m opentraj_dataset_stats
"$python_bin" -m paper_dataset_metrics --allow-partial
echo "[$(date --iso-8601=seconds)] complete $dataset_name"
