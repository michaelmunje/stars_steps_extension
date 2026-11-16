#!/usr/bin/env bash
# One GB200 node: STARS leave-one-out fine-tunes + open-loop STARS predictions + closed-loop rollouts.
#   sbatch --time=04:00:00 --job-name=fc_bench scripts/horizon/gpu_jobs.slurm \
#     "bash runs/forecast_bench/stars_gpu.sh gpu0" "bash runs/forecast_bench/stars_gpu.sh gpu1" \
#     "bash runs/forecast_bench/stars_gpu.sh gpu2" "bash runs/forecast_bench/stars_gpu.sh gpu3"
set -euo pipefail
ROOT=experiments/forecast_bench
ETH=ethucy_eth,ethucy_hotel,ethucy_univ,ethucy_zara1,ethucy_zara2
POOLED=${POOLED:-pooled_steps,pooled_ptdpes,pooled_madras,pooled_grand_central,pooled_eth,pooled_ucy,pooled_atc}
CL="python -m forecast_bench.closed_loop --device cuda --out $ROOT/closed_loop --methods"
loo() { for fold in "$@"; do
  python -m forecast_bench.finetune_stars_loo --fold "$fold" --out "$ROOT/stars_loo/$fold"
  touch "$ROOT/stars_loo/$fold/done"; done; }
case "$1" in
  gpu0) loo eth hotel; $CL socialvae_official@4,moflow_official@4 ;;
  gpu1) loo univ; $CL stars_steps_native,stars_pooled_native,stars_noethucy_native,stars_noethucy_goal_native ;;
  gpu2) loo zara1 zara2; $CL ground_truth,cv,stars_noethucy@4,trajectron_official@4,agentformer_official@4,sgan_official@4 ;;
  gpu3)
    python -m forecast_bench.predict --method cv --sets "$ETH,$POOLED"
    for m in stars_steps stars_pooled stars_noethucy stars_noethucy_goal stars_steps_diffusion stars_steps_joint_diffusion; do
      python -m forecast_bench.predict --method "$m" --sets "$ETH,$POOLED"
    done
    until [ -f $ROOT/stars_loo/eth/done ] && [ -f $ROOT/stars_loo/hotel/done ] && [ -f $ROOT/stars_loo/univ/done ] \
          && [ -f $ROOT/stars_loo/zara1/done ] && [ -f $ROOT/stars_loo/zara2/done ]; do sleep 30; done
    python -m forecast_bench.predict --method stars_loo --sets "$ETH"
    $CL stars_loo_native,stars_loo@4 ;;
esac
