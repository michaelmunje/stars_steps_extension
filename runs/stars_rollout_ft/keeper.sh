#!/usr/bin/env bash
# Keep up to $1 (default 1) short stars_rft jobs queued/running while queued work remains.
cd /scratch/09865/mmunje/stars-rollout-ft
want=${1:-1}
while [ -n "$(find runs/stars_rollout_ft -path "*/q[0-9]/*" \( -name "*.req" -o -name "*.running.*" \) | head -1)" ]; do
  n=$(squeue -u $USER -h -n stars_rft | wc -l)
  for ((i=n; i<want; i++)); do sbatch --time=00:30:00 runs/stars_rollout_ft/node.slurm >/dev/null 2>&1 && echo "$(date +%H:%M) submitted"; done
  sleep 300
done
echo "queues empty $(date)"
