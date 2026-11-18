#!/usr/bin/env bash
# Protocol A pretraining: the stars_improved recipe on STEPS, PTDPES, MADRAS, Grand Central (no ATC, no ETH/UCY).
# Usage: pretrain.sh SEED [EPOCHS]   (resumable from checkpoints/latest.pt)
set -euo pipefail
cd /scratch/09865/mmunje/stars-ethucy-protocol
seed=$1; epochs=${2:-60}
run=experiments/stars_ma/runs/ep_pre4_s$seed
resume=""; [ -f $run/checkpoints/latest.pt ] && resume=--resume
[ -f $run/status.json ] && grep -q '"status": "complete"' $run/status.json && exit 0
python -m baselines.train_stars_ma --phase forecast --config configs/stars_ma/four_no_ethucy.yaml \
  --encoder-mode scratch --decoder cvae --cvae-latent-dim 64 --decoder-hidden-dim 512 --decoder-dropout 0.0633 \
  --dropout 0.0905 --kl-weight 0.0799 --weight-decay 0.00217 --warmup-epochs 2 --epochs $epochs --max-nodes 4096 \
  --knn 24 --radius 10 --seed $seed --data-seed 42 --selection macro --cache-workers 32 --eval-every 3 \
  $resume --run-dir $run
