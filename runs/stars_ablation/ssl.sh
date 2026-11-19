#!/usr/bin/env bash
# bash runs/stars_ablation/ssl.sh <E0|E1|E2|E3> <seed> [extra flags]: one self-supervised STARS encoder of the ladder.
# Budget-capped STARS "homo" recipe: 15 epochs (STARS: <= 1000, patience 20), beta warm-up 3 epochs (STARS: 50),
# 4096 nodes per step (lr sqrt-scaled from STARS's 1.16e-4 at 32 graphs ~ 600 nodes: 3.0e-4), cosine LR,
# best epoch by validation loss at beta = 0.9 (beta_max). Resumable.
source runs/stars_ablation/ssl_flags.sh
E=$1; S=$2; shift 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m baselines.train_stars_ma --phase ssl $DATA ${!E} --max-nodes 4096 --epochs 15 --beta-warmup-epochs 3 \
  --ssl-val-beta 0.9 --seed $S --run-dir experiments/stars_ablation/ssl/${E}_s$S --resume --cache-workers 16 "$@" 2>&1 | grep --line-buffered -v "batch/s"
exit ${PIPESTATUS[0]}
