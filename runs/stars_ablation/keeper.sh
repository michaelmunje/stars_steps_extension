#!/usr/bin/env bash
# Keep up to $1 (default 2) stars_abl jobs queued/running while runnable requests remain.
cd /scratch/09865/mmunje/stars-ablation
MAX=${1:-2}
runnable() {
  for r in runs/stars_ablation/queue/*.req; do
    [[ -e $r ]] || continue
    ok=1; for need in $(sed -n 's/^# needs: //p' "$r"); do [[ -e $need ]] || { ok=0; break; }; done
    [[ $ok == 1 ]] && { echo $r; return; }
  done
}
while [[ ! -f runs/stars_ablation/STOP_KEEPER ]]; do
  n=$(squeue -u $USER -h -n stars_abl | wc -l)
  if [[ -n "$(runnable)" ]]; then
    for ((i=n; i<MAX; i++)); do sbatch runs/stars_ablation/node.slurm >/dev/null 2>&1 && echo "$(date +%H:%M) submitted"; done
  fi
  sleep 120
done
