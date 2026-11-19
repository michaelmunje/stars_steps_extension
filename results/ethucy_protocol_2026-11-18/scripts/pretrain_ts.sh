#!/usr/bin/env bash
# Two-stage protocol A': the protocol-A forecaster recipe (pretrain.sh) with the STARS self-supervised encoder
# E3_s<SEED>_long (branch stars-ablation; same 4 datasets, graph 24-NN / 10 m, native 8 steps) as initialisation.
# Usage: pretrain_ts.sh SEED MODE   (MODE: finetune -> ep_ts_pre4_s<SEED>, frozen -> ep_tsf_pre4_s<SEED>)
set -euo pipefail
cd /scratch/09865/mmunje/stars-ethucy-protocol
seed=$1; mode=$2; tag=$([ $mode = frozen ] && echo tsf || echo ts)
enc=/scratch/09865/mmunje/stars-ablation/experiments/stars_ablation/ssl/E3_s${seed}_long/checkpoints/best.pt
run=experiments/stars_ma/runs/ep_${tag}_pre4_s$seed
resume=""; [ -f $run/checkpoints/latest.pt ] && resume=--resume
[ -f $run/status.json ] && grep -q '"status": "complete"' $run/status.json && exit 0
python -m baselines.train_stars_ma --phase forecast --config configs/stars_ma/four_no_ethucy.yaml \
  --encoder-checkpoint $enc --encoder-mode $mode --decoder cvae --cvae-latent-dim 64 --decoder-hidden-dim 512 \
  --decoder-dropout 0.0633 --dropout 0.0905 --kl-weight 0.0799 --weight-decay 0.00217 --warmup-epochs 2 --epochs 60 \
  --max-nodes 4096 --knn 24 --radius 10 --seed $seed --data-seed 42 --selection macro --cache-workers 32 --eval-every 3 \
  $resume --run-dir $run
