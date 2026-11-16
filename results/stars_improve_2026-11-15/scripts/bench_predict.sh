#!/usr/bin/env bash
# Forecast-benchmark predictions for a STARS method from stars-improve ($1), all 12 window sets.
set -euo pipefail
cd /scratch/09865/mmunje/stars-forecast-benchmark
export PYTHONPATH=/scratch/09865/mmunje/stars-improve/src:/scratch/09865/mmunje/stars-forecast-benchmark/src
SETS=ethucy_eth,ethucy_hotel,ethucy_univ,ethucy_zara1,ethucy_zara2,pooled_steps,pooled_ptdpes,pooled_madras,pooled_grand_central,pooled_eth,pooled_ucy,pooled_atc
python -m forecast_bench.predict --skip-existing --method $1 --sets $SETS
