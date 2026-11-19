#!/usr/bin/env bash
# bash runs/stars_ablation/ssl_long.sh <E> <seed> [extra]: warm restart of a 15-epoch encoder for 25 more epochs
# (new cosine schedule, beta fixed at 0.9: 40 epochs in total) into ssl/<E>_s<seed>_long.
E=$1; S=$2; shift 2
RUN=${E}_s${S}_long bash runs/stars_ablation/ssl.sh $E $S --init-checkpoint experiments/stars_ablation/ssl/${E}_s$S/checkpoints/best.pt \
  --epochs 25 --beta-warmup-epochs 0 "$@"
