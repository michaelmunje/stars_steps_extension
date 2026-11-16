#!/usr/bin/env bash
# One all-dataset CVAE run with the pooled recipe of cvae_all_lr3e-4_seed42 plus extra flags, then its
# open-loop validation report. Usage: train.sh NAME EPOCHS [extra train_stars_ma flags...]
set -euo pipefail
cd /scratch/09865/mmunje/stars-improve
name=$1; epochs=$2; shift 2
run=experiments/stars_ma/runs/si_$name
resume=""; [ -f $run/checkpoints/latest.pt ] && resume=--resume
[ -f $run/status.json ] && grep -q '"status": "complete"' $run/status.json && skip=1 || skip=0
[ $skip = 1 ] || python -m baselines.train_stars_ma --phase forecast --config configs/stars_ma/all7.yaml \
  --encoder-mode scratch --decoder cvae --cvae-latent-dim 64 --decoder-hidden-dim 512 --decoder-dropout 0.0633 \
  --dropout 0.0905 --kl-weight 0.0799 --weight-decay 0.00217 --warmup-epochs 2 --epochs $epochs --max-nodes 4096 \
  --knn 24 --radius 10 --seed 42 --data-seed 42 --selection macro --cache-workers 32 --eval-every 3 \
  "$@" $resume --run-dir $run
python -m baselines.stars_ma_openloop --split val --checkpoint $name=$run/checkpoints/best.pt \
  --datasets steps,ptdpes,madras,grand_central,ucy,atc --output results/stars_improve_2026-11-15/val/$name.json
