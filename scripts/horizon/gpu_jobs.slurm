#!/usr/bin/env bash
# Run up to four independent commands on one TACC Horizon GB200 node, one GPU each.
#
# Horizon allocates whole nodes (4 GPUs), so put up to four runs in one job. Every argument is
# one shell command, run from the repository root with .venv active and CUDA_VISIBLE_DEVICES set
# to its own GPU. Logs: logs/horizon/<job id>/gpu<N>.log. The job fails if any command fails.
#
#   sbatch scripts/horizon/gpu_jobs.slurm \
#     "python -m baselines.train_stars_ma --phase ssl --dataset steps --run-dir experiments/stars_ma/ssl" \
#     "python -m baselines.train_stars_ma --phase ssl --dataset steps --edge-only --run-dir experiments/stars_ma/ssl_edge_only"
#
# Submit from the repository root (sbatch writes the Slurm log to the submission directory).
# Override the defaults on the command line, e.g. `sbatch --time=12:00:00 --account=<yours> ...`.
#SBATCH --partition=debug
#SBATCH --account=CDA26011
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=stars_ma
#SBATCH --output=%x-%j.out

set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: sbatch $0 CMD [CMD ...]   (1 to 4 commands, one per GPU)" >&2
  exit 2
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"
if [[ ! -f pyproject.toml || ! -x .venv/bin/python ]]; then
  echo "Submit from the repository root after 'uv sync' (no pyproject.toml or .venv in $PROJECT_DIR)" >&2
  exit 2
fi

# sbatch exports the submitter's environment; an inherited $TMUX confuses tools started here.
unset TMUX TMUX_PANE
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# Compute nodes have no internet access.
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
# The default nvc compiler cannot build Triton/torch.compile helpers or C extensions.
export CC=gcc CXX=g++

LOG_DIR="logs/horizon/${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"
echo "node $(hostname)  job ${SLURM_JOB_ID:-local}  logs $LOG_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

pids=()
gpu=0
for command in "$@"; do
  echo "gpu$gpu: $command" | tee "$LOG_DIR/gpu$gpu.log"
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$command" >>"$LOG_DIR/gpu$gpu.log" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "gpu$index: done"
  else
    echo "gpu$index: FAILED (exit $?), see $LOG_DIR/gpu$index.log" >&2
    status=1
  fi
done
exit $status
