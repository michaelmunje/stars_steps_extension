#!/usr/bin/env bash
# One no-ETH/UCY pooled CVAE (pooled recipe of cvae_all_lr3e-4_seed42) then its test metrics. $1 = goal|nogoal
set -euo pipefail
name=cvae_noethucy_$1
run=experiments/stars_ma/runs/$name
python -m baselines.train_stars_ma --phase forecast --config configs/stars_ma/all_no_ethucy.yaml \
  --encoder-mode scratch --decoder cvae --cvae-latent-dim 64 --decoder-hidden-dim 512 --decoder-dropout 0.0633 \
  --dropout 0.0905 --kl-weight 0.0799 --weight-decay 0.00217 --warmup-epochs 4.31 --epochs 60 --max-nodes 4096 \
  --knn 24 --radius 10 --seed 42 --data-seed 42 --selection macro --cache-workers 32 \
  $([ "$1" = goal ] && echo --goal) --run-dir $run
python -m baselines.evaluate_stars_ma --checkpoint $run/checkpoints/best.pt --config configs/stars_ma/all_no_ethucy.yaml \
  --split test --output $run/eval_test_noethucy.json
