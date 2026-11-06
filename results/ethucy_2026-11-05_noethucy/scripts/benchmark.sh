#!/usr/bin/env bash
# Wait for both trainings, then add their rows to the ETH/UCY benchmark (same windows/seeds/CIs as 4981b49).
set -euo pipefail
R=results/ethucy_2026-11-05_noethucy
for g in goal nogoal; do
  until [ -f experiments/stars_ma/runs/cvae_noethucy_$g/eval_test_noethucy.json ]; do
    grep -q '"status": "failed"' experiments/stars_ma/runs/cvae_noethucy_$g/status.json 2>/dev/null && { echo "$g failed"; exit 1; }
    sleep 120
  done
done
M="stars_noethucy stars_noethucy_goal stars_noethucy_goal_adapter"
python simplenavsim/scripts/ethucy_benchmark.py calibrate --append --run $R --methods $M --device cuda
python simplenavsim/scripts/ethucy_benchmark.py run --append --run $R --methods $M --device cuda
for s in aggregate plot report; do python simplenavsim/scripts/ethucy_benchmark.py $s --run $R; done
