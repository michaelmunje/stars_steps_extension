#!/usr/bin/env bash
# Closed-loop fine-tune of a trained all-7 STARS CVAE. Usage: train.sh NAME INIT_RUN EPOCHS [extra train_stars_ma flags]
# Resumes from checkpoints/latest.pt; skips if complete. Checkpoints: experiments/stars_ma/runs/rft_NAME.
set -euo pipefail
cd /scratch/09865/mmunje/stars-rollout-ft
name=$1; init=$2; epochs=$3; shift 3
run=experiments/stars_ma/runs/rft_$name
resume=""; [ -f $run/checkpoints/latest.pt ] && resume=--resume
if [ -f $run/status.json ] && grep -q '"status": "complete"' $run/status.json; then echo "complete: $run"; exit 0; fi
python -m baselines.train_stars_ma --phase forecast --config configs/stars_ma/all7.yaml --encoder-mode scratch \
  --init-checkpoint experiments/stars_ma/runs/$init/checkpoints/best.pt --knn 24 --radius 10 \
  --epochs $epochs --max-nodes 2048 --learning-rate 5e-5 --lr-scaling none --warmup-epochs 0.2 \
  --kl-weight 0.0799 --kl-warmup-epochs 0 --weight-decay 0.00217 \
  --rft-eval-steps 12 --selection closedloop --samples 20 --train-batches 300 --val-batches 150 \
  --seed 42 --data-seed 42 --cache-workers 32 --eval-every 1 "$@" $resume --run-dir $run
