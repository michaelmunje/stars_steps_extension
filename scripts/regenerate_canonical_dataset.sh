#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: regenerate_canonical_dataset.sh DATASET [DATA_SEED]}"
data_seed="${2:-42}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
atc_workers="${ATC_WORKERS:-8}"
dataset_name="${dataset,,}"

mkdir -p "$project_dir/logs/preprocess"
exec > >(tee -a "$project_dir/logs/preprocess/${dataset_name}-$(date +%Y%m%dT%H%M%S).log") 2>&1

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
case "${dataset,,}" in
  eth)
    "$python_bin" -m datasets_preprocessing.eth.convert datasets/eth
    ;;
  ucy)
    "$python_bin" -m datasets_preprocessing.ucy.convert datasets/ucy
    ;;
  atc)
    "$python_bin" -m datasets_preprocessing.atc.convert datasets/atc --workers "$atc_workers"
    ;;
  steps)
    "$python_bin" -m datasets_preprocessing.steps.convert datasets/steps
    ;;
  madras)
    [[ -d datasets/madras/raw ]] || {
      echo "MADRAS raw data is missing at datasets/madras/raw" >&2
      exit 1
    }
    "$python_bin" -m datasets_preprocessing.madras.convert datasets/madras
    ;;
  ptdpes)
    [[ -d datasets/PTDPES/raw ]] || {
      echo "PTDPES raw data is missing at datasets/PTDPES/raw" >&2
      exit 1
    }
    "$python_bin" -m datasets_preprocessing.PTDPES.convert datasets/PTDPES
    ;;
  grand_central)
    "$python_bin" -m datasets_preprocessing.grand_central.convert datasets/grand_central
    ;;
  *)
    echo "Unknown dataset '$dataset'; choose eth, ucy, atc, steps, madras, grand_central, or ptdpes" >&2
    exit 2
    ;;
esac

export STEPS_PHASE1_LOGGED=1
"$project_dir/scripts/regenerate_dataset_phase1.sh" "${dataset,,}" "$data_seed"
