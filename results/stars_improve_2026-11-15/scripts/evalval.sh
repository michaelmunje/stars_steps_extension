#!/usr/bin/env bash
# Open-loop validation report of si_$1 (all six validation datasets).
cd /scratch/09865/mmunje/stars-improve
python -m baselines.stars_ma_openloop --split val --checkpoint $1=experiments/stars_ma/runs/si_$1/checkpoints/best.pt \
  --datasets steps,ptdpes,madras,grand_central,ucy,atc --output results/stars_improve_2026-11-15/val/$1.json
