#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --account=CDA26011
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@150
#SBATCH --job-name=stars_abl
#SBATCH --output=runs/stars_ablation/logs/node-%j.out
# Four GPU workers on the shared queue runs/stars_ablation/queue. 2.5 min before the limit, unfinished
# requests go back to the queue (every trainer resumes from its latest.pt). Workers exit when no runnable
# request is left, so the node never idles.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; unset TMUX TMUX_PANE
source /scratch/09865/mmunje/stars-forecasting/.venv/bin/activate
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 CC=gcc CXX=g++
S=runs/stars_ablation
requeue() {
  echo "requeue at $(date)"
  for f in $S/queue/*.running.$SLURM_JOB_ID; do [ -e "$f" ] && mv "$f" "${f%.running.$SLURM_JOB_ID}.req" && echo "requeued $f"; done
  pkill -TERM -f "python -m " ; sleep 20; pkill -KILL -f "python -m "
  exit 0
}
trap requeue USR1
echo "node $(hostname) job $SLURM_JOB_ID $(date)"; nvidia-smi -L
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g bash $S/worker.sh $S/queue & done
wait
echo "workers done $(date)"
