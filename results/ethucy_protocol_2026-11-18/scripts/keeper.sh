#!/usr/bin/env bash
# Keep up to N (default 2: one running, one waiting via singleton) short ethucy_proto jobs queued/running while queued or running requests remain.
cd /scratch/09865/mmunje/stars-ethucy-protocol
N=${1:-2}
while [ -n "$(find runs/ethucy_protocol -path "*/q[0-9]/*" \( -name "*.req" -o -name "*.running.*" \) | head -1)" ] || pgrep -f "bash pipeline.sh" >/dev/null; do
  [ -f runs/ethucy_protocol/KEEPER_N ] && N=$(cat runs/ethucy_protocol/KEEPER_N)
  n=$(squeue -u $USER -h -n ethucy_proto | wc -l)
  for ((i=n; i<N; i++)); do sbatch --time=00:30:00 --dependency=singleton runs/ethucy_protocol/node1h.slurm >/dev/null 2>&1 && echo "$(date +%H:%M) submitted"; done
  sleep 300
done
echo "queues empty $(date)"
