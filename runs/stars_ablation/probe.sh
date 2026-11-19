#!/usr/bin/env bash
# bash runs/stars_ablation/probe.sh <seed> <suffix ("" = 15-epoch encoders, "_long" = 40-epoch)> <name tag> <rung>...
# Runs the L0-L3 linear probes of the given rungs in parallel on this GPU (L3 = multi-agent head on E3).
S=$1; SUF=$2; TAG=$3; shift 3
declare -A ENC=([L0]=E0 [L1]=E1 [L2]=E2 [L3]=E3) MODE=([L0]=single [L1]=single [L2]=single [L3]=multi)
for L in "$@"; do
  python -m forecast_bench.stars_ablation --encoder experiments/stars_ablation/ssl/${ENC[$L]}_s$S$SUF/checkpoints/best.pt \
    --mode ${MODE[$L]} --seed $S --method abl_${L}${TAG}_s$S > experiments/stars_ablation/probe_${L}${TAG}_s$S.log 2>&1 &
done
wait
for L in "$@"; do [[ -f experiments/forecast_bench/ablation/abl_${L}${TAG}_s$S/summary.json ]] || { echo "$L failed"; exit 1; }; done
