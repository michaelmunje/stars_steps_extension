#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --account=CDA26011
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --job-name=ethucy_proto
#SBATCH --output=runs/ethucy_protocol/logs/node-%j.out
# One-hour node: eight workers (two per GPU) on runs/ethucy_protocol/q{0..7}. Two and a half minutes before the
# limit, unfinished requests go back to their queue (training resumes from checkpoints/latest.pt).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; unset TMUX TMUX_PANE
source /scratch/09865/mmunje/stars-forecasting/.venv/bin/activate
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 CC=gcc CXX=g++
S=runs/ethucy_protocol
requeue() {
  echo "requeue at $(date)"
  for f in $S/q*/*.running.$SLURM_JOB_ID; do [ -e "$f" ] && mv "$f" "${f%.running.$SLURM_JOB_ID}.req" && echo "requeued $f"; done
  pkill -TERM -f "baselines\.|forecast_bench\." ; sleep 20; pkill -KILL -f "baselines\.|forecast_bench\."
  exit 0
}
trap requeue USR1
echo "node $(hostname) job $SLURM_JOB_ID $(date)"; nvidia-smi -L
for q in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$((q % 4)) bash $S/worker.sh $S/q$q & done   # 2 workers per GPU
wait
