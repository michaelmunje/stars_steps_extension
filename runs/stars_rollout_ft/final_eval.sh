#!/usr/bin/env bash
# After training: (1) closed-loop ETH/UCY validation (students003) of every fine-tuned best.pt, 3 seeds;
# (2) ETH/UCY benchmark (all windows) appended to a copy of results/ethucy_2026-11-16_improved, 1 seed;
# (3) open-loop test accuracy (as runs/stars_improve/test_final_os1.sh).
set -uo pipefail
cd /scratch/09865/mmunje/stars-rollout-ft
R=experiments/stars_ma/runs
part=${1:-all}
if [[ $part == val || $part == all ]]; then
python runs/stars_rollout_ft/cl_val.py --out results/rft_val/final.json --seeds 3 cv \
  base=$R/si_final_base_s2/checkpoints/best.pt social=$R/si_final_social_s42/checkpoints/best.pt \
  rft_base=$R/rft_base_noatc/checkpoints/best.pt rft_base_progress=$R/rft_base_progress_noatc/checkpoints/best.pt \
  rft_social_coll=$R/rft_social_coll_noatc/checkpoints/best.pt rft_social_coll_progress=$R/rft_social_coll_progress_noatc/checkpoints/best.pt
fi
if [[ $part == bench || $part == all ]]; then
python simplenavsim/scripts/ethucy_benchmark.py run --run results/ethucy_rft_eval --append --methods stars_rft stars_rft_os4 stars_rft_social --seeds 1
python simplenavsim/scripts/ethucy_benchmark.py aggregate --run results/ethucy_rft_eval
fi
if [[ $part == openloop || $part == all ]]; then
python -m baselines.stars_ma_openloop --split test --datasets steps,ptdpes,madras,grand_central,ucy,eth,atc --cache-workers 32 \
  --checkpoint base_s2=$R/si_final_base_s2/checkpoints/best.pt --checkpoint social_s42=$R/si_final_social_s42/checkpoints/best.pt \
  --checkpoint rft_base=$R/rft_base_noatc/checkpoints/best.pt --checkpoint rft_base_progress=$R/rft_base_progress_noatc/checkpoints/best.pt \
  --checkpoint rft_social_coll_progress=$R/rft_social_coll_progress_noatc/checkpoints/best.pt \
  --output results/rft_val/openloop_test_os1.json
fi
