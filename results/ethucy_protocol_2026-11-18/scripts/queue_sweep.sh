#!/usr/bin/env bash
# Queue the LOSO fine-tuning sweep for pretraining seed $1 over the 8 worker queues.
cd /scratch/09865/mmunje/stars-ethucy-protocol/runs/ethucy_protocol
seed=$1; i=${2:-0}
for lr in 3e-5 1e-4 3e-4; do for enc in 1 0.1; do for ep in 10 30; do for f in eth hotel univ zara1 zara2; do
  echo "bash runs/ethucy_protocol/loso.sh loso $seed $f lr${lr}_enc${enc}_e$ep" > q$((i % 8))/30_loso_s${seed}_lr${lr}_enc${enc}_e${ep}_$f.req; i=$((i+1))
done; done; done; done
