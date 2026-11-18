#!/usr/bin/env bash
# Open-loop predictions of protocol rows on the 5 ETH/UCY test folds. Usage: predict.sh METHOD...
set -euo pipefail
cd /scratch/09865/mmunje/stars-ethucy-protocol
for m in "$@"; do
  python -m forecast_bench.predict --method $m --sets ethucy_eth,ethucy_hotel,ethucy_univ,ethucy_zara1,ethucy_zara2 --skip-existing
done
