#!/usr/bin/env bash
# One GB200 node: STARS leave-one-out fine-tunes (GPUs 0-2) and open-loop STARS predictions (GPU 3).
#   sbatch --time=03:00:00 --job-name=fc_bench scripts/horizon/gpu_jobs.slurm \
#     "bash runs/forecast_bench/stars_gpu.sh loo eth hotel" "bash runs/forecast_bench/stars_gpu.sh loo univ" \
#     "bash runs/forecast_bench/stars_gpu.sh loo zara1 zara2" "bash runs/forecast_bench/stars_gpu.sh predict"
set -euo pipefail
ROOT=experiments/forecast_bench
ETH=ethucy_eth,ethucy_hotel,ethucy_univ,ethucy_zara1,ethucy_zara2
POOLED=${POOLED:-pooled_steps,pooled_ptdpes,pooled_madras,pooled_grand_central,pooled_eth,pooled_ucy,pooled_atc}
case "$1" in
  loo)
    shift
    for fold in "$@"; do
      python -m forecast_bench.finetune_stars_loo --fold "$fold" --out "$ROOT/stars_loo/$fold"
    done ;;
  predict)
    python -m forecast_bench.predict --method cv --sets "$ETH,$POOLED"
    for m in stars_steps stars_pooled stars_noethucy stars_noethucy_goal stars_steps_diffusion stars_steps_joint_diffusion; do
      python -m forecast_bench.predict --method "$m" --sets "$ETH,$POOLED"
    done ;;
  predict_loo)
    python -m forecast_bench.predict --method stars_loo --sets "$ETH" ;;
esac
