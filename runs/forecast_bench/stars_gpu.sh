#!/usr/bin/env bash
# Forecast benchmark GPU work, split into short jobs (one mode per GPU; 4 per node):
#   preds   (30 min): sbatch --time=00:30:00 scripts/horizon/gpu_jobs.slurm "bash runs/forecast_bench/stars_gpu.sh preds stars_noethucy stars_steps" \
#                       "... preds stars_pooled stars_noethucy_goal" "... preds stars_steps_diffusion" "... preds stars_steps_joint_diffusion"
#   loo     (1 h):    "... loo eth hotel" "... loo univ" "... loo zara1" "... loo zara2"   (then predict_loo)
#   closed  (1 h):    "... cl_baselines" "... cl_stars" "... cl_misc" "... cl_learned"
#   cl_loo  (30 min, after loo): "... cl_loo"
set -euo pipefail
ROOT=experiments/forecast_bench
ETH=ethucy_eth,ethucy_hotel,ethucy_univ,ethucy_zara1,ethucy_zara2
POOLED=pooled_steps,pooled_ptdpes,pooled_madras,pooled_grand_central,pooled_eth,pooled_ucy,pooled_atc
CL="python -m forecast_bench.closed_loop --device cuda --out $ROOT/closed_loop --methods"
mode=$1; shift
case "$mode" in
  preds) for m in "$@"; do python -m forecast_bench.predict --skip-existing --method "$m" --sets "$ETH,$POOLED"; done ;;
  loo) for fold in "$@"; do
         python -m forecast_bench.finetune_stars_loo --fold "$fold" --epochs "${EPOCHS:-15}" --out "$ROOT/stars_loo/$fold"
         touch "$ROOT/stars_loo/$fold/done"
         python -m forecast_bench.predict --method stars_loo --sets "ethucy_$fold"
       done ;;
  cl_baselines) $CL socialvae_official@4,moflow_official@4 ;;
  cl_stars) $CL stars_steps_native,stars_pooled_native,stars_noethucy_native,stars_noethucy_goal_native ;;
  cl_misc) $CL ground_truth,cv,stars_noethucy@4,sgan_official@4,agentformer_official@4 ;;
  cl_learned) $CL trajectron_official@4 ;;
  cl_loo) $CL stars_loo_native,stars_loo@4 ;;
  cl_fold) for fold in "${@:2}"; do $CL "$1" --folds "$fold"; done ;;   # e.g. cl_fold moflow_official@4 univ
esac
