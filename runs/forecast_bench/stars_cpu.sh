#!/usr/bin/env bash
# Open-loop STARS predictions on CPU (login node), smallest sets first; skips existing files.
set -uo pipefail
SETS=ethucy_eth,ethucy_hotel,ethucy_zara1,ethucy_zara2,pooled_madras,pooled_eth,pooled_ucy,ethucy_univ,pooled_ptdpes,pooled_atc,pooled_steps,pooled_grand_central
for m in stars_noethucy stars_pooled stars_steps stars_noethucy_goal stars_steps_joint_diffusion stars_steps_diffusion; do
  python -m forecast_bench.predict --skip-existing --device cpu --method "$m" --sets "$SETS"
done
