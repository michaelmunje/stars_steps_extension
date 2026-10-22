#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: pretrain_trajectron_ptdpes_seed.sh SEED GPU_INDEX}"
GPU_INDEX="${2:?usage: pretrain_trajectron_ptdpes_seed.sh SEED GPU_INDEX}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_VERSION="${RUN_VERSION:-transfer_v2}"
MODEL_ID="trajectron_${RUN_VERSION}_ptdpes_${SEED}"
RUN_DIR="$PROJECT_DIR/experiments/pretrained/$MODEL_ID"
LOG_DIR="$PROJECT_DIR/logs/experiments"
LOG="$LOG_DIR/trajectron-${RUN_VERSION}-ptdpes-seed${SEED}.log"

mkdir -p "$LOG_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export TRAJECTORY_MODEL_REGISTRY="$PROJECT_DIR/experiments/model_registry"
exec > >(tee -a "$LOG") 2>&1

resume_args=()
if [[ -s "$RUN_DIR/checkpoints/best.pt" && -f "$RUN_DIR/status.json" ]] \
    && [[ "$(jq -r '.status // empty' "$RUN_DIR/status.json")" == "complete" ]]; then
  echo "[$(date --iso-8601=seconds)] SKIP completed model=trajectron seed=$SEED run=$RUN_DIR"
  exit 0
elif [[ -s "$RUN_DIR/checkpoints/latest.pt" ]]; then
  resume_args+=(--resume)
elif [[ -e "$RUN_DIR/config.json" || -e "$RUN_DIR/status.json" ]]; then
  echo "Existing run metadata has no resumable latest checkpoint: $RUN_DIR" >&2
  exit 3
fi

echo "[$(date --iso-8601=seconds)] START model=trajectron seed=$SEED physical_gpu=$GPU_INDEX batch_size=128"
echo "[$(date --iso-8601=seconds)] PYTHON executable=$($PYTHON_BIN -c 'import sys; print(sys.executable)')"
"$PYTHON_BIN" -c 'import pykalman; print("pykalman preflight: OK")'
nvidia-smi --id="$GPU_INDEX" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
cd "$PROJECT_DIR"
"$PYTHON_BIN" -m baselines.train_trajectron \
  --stage pretrain \
  --pretrain-dataset ptdpes \
  --seed "$SEED" \
  --data-seed 42 \
  --epochs 100 \
  --batch-size 128 \
  --patience 10 \
  --min-delta 0.0001 \
  --status-every 100 \
  --device cuda \
  --run-dir "$RUN_DIR" \
  "${resume_args[@]}"

"$PYTHON_BIN" -m baselines.registry add \
  "$MODEL_ID" "$RUN_DIR/checkpoints/best.pt" \
  --model trajectron --dataset ptdpes --seed "$SEED"
echo "[$(date --iso-8601=seconds)] COMPLETE model=trajectron seed=$SEED"
