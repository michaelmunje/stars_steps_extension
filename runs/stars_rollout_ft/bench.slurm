#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --account=CDA26011
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --job-name=stars_rft
#SBATCH --output=runs/stars_rollout_ft/logs/bench-%j.out
# Throughput / memory of the closed-loop fine-tuning step (K = 12) at several batch sizes, plus a GPU smoke run.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; unset TMUX TMUX_PANE
source /scratch/09865/mmunje/stars-forecasting/.venv/bin/activate
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 CC=gcc CXX=g++
S=runs/stars_rollout_ft
b() { g=$1; shift; CUDA_VISIBLE_DEVICES=$g bash $S/train.sh bench_$g si_final_base_s2 1 --benchmark 30 "$@" > $S/logs/bench_$g.log 2>&1; }
b 0 --rft-steps 12 --max-nodes 2048 &
b 1 --rft-steps 12 --max-nodes 4096 &
b 2 --rft-steps 12 --max-nodes 4096 --activation-checkpointing &
b 3 --rft-steps 12 --max-nodes 2048 --rft-latent posterior &
wait
for g in 0 1 2 3; do echo "== $g"; grep -v "it/s\]\|s/it\]" $S/logs/bench_$g.log | tail -3; done
# GPU smoke run: two short epochs with eval (ucy only, capped)
CUDA_VISIBLE_DEVICES=0 bash $S/train.sh gpusmoke si_final_base_s2 2 --rft-steps 4 --rft-start-steps 2 --rft-ramp-epochs 1 \
  --train-batches 20 --val-batches 20 > $S/logs/gpusmoke.log 2>&1
tail -3 $S/logs/gpusmoke.log
