#!/usr/bin/env bash
# One multi-agent STARS training run, data-parallel over every GPU of the allocated node(s).
#
#   sbatch scripts/horizon/stars_ma.slurm --config configs/stars_ma/steps.yaml --phase ssl \
#       --run-dir experiments/stars_ma/ssl
#   sbatch --nodes=2 scripts/horizon/stars_ma.slurm ...        # 8 GPUs across two nodes
#
# Arguments go to `python -m baselines.train_stars_ma` (one process per GPU via torchrun). Set
# STARS_MA_MODULE=baselines.evaluate_stars_ma to run the evaluator the same way. Submit from the
# repository root; the log is logs/horizon/<job id>/train.log.
#SBATCH --partition=debug
#SBATCH --account=CDA26011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=stars_ma
#SBATCH --output=%x-%j.out

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"
if [[ ! -f pyproject.toml || ! -x .venv/bin/python ]]; then
  echo "Submit from the repository root after 'uv sync' (no pyproject.toml or .venv in $PROJECT_DIR)" >&2
  exit 2
fi
unset TMUX TMUX_PANE
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=8
# The default nvc compiler cannot build Triton/torch.compile helpers or C extensions.
export CC=gcc CXX=g++

GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
LOG_DIR="logs/horizon/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
echo "job $SLURM_JOB_ID: $SLURM_NNODES node(s) x $GPUS_PER_NODE GPUs, master $MASTER_ADDR, log $LOG_DIR/train.log"

srun --ntasks-per-node=1 --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29511" --rdzv-id="$SLURM_JOB_ID" \
  -m "${STARS_MA_MODULE:-baselines.train_stars_ma}" "$@" 2>&1 | tee "$LOG_DIR/train.log"
