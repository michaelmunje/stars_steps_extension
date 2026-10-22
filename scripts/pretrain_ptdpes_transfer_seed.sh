#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: pretrain_ptdpes_transfer_seed.sh MODEL SEED GPU_INDEX}"
SEED="${2:?usage: pretrain_ptdpes_transfer_seed.sh MODEL SEED GPU_INDEX}"
GPU_INDEX="${3:?usage: pretrain_ptdpes_transfer_seed.sh MODEL SEED GPU_INDEX}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_VERSION="${RUN_VERSION:-transfer_v2}"
MODEL_TAG="${MODEL/social_vae/socialvae}"
RUN_DIR="$PROJECT_DIR/experiments/pretrained/${MODEL_TAG}_${RUN_VERSION}_ptdpes_${SEED}"
LOG_DIR="$PROJECT_DIR/logs/experiments"
LOG="$LOG_DIR/${MODEL_TAG}-${RUN_VERSION}-ptdpes-seed${SEED}.log"

mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHON_BIN
export TRAJECTORY_MODEL_REGISTRY="$PROJECT_DIR/experiments/model_registry"
exec > >(tee -a "$LOG") 2>&1

resume_args=()
if [[ -s "$RUN_DIR/checkpoints/best.pt" && -f "$RUN_DIR/status.json" ]] \
    && [[ "$(jq -r '.status // empty' "$RUN_DIR/status.json")" == "complete" ]]; then
  echo "[$(date --iso-8601=seconds)] SKIP completed model=$MODEL seed=$SEED run=$RUN_DIR"
  exit 0
elif [[ -s "$RUN_DIR/checkpoints/latest.pt" ]]; then
  resume_args+=(--resume)
elif [[ -e "$RUN_DIR/config.json" || -e "$RUN_DIR/status.json" ]]; then
  echo "Existing run metadata has no resumable latest checkpoint: $RUN_DIR" >&2
  exit 3
fi

echo "[$(date --iso-8601=seconds)] START model=$MODEL seed=$SEED physical_gpu=$GPU_INDEX run_version=$RUN_VERSION"
cd "$PROJECT_DIR"
"$PROJECT_DIR/scripts/pretrain_transfer_model.sh" \
  --model "$MODEL" \
  --sources ptdpes \
  --seed "$SEED" \
  --data-seed 42 \
  --device cuda \
  --num-workers 2 \
  --run-version "$RUN_VERSION" \
  --python "$PYTHON_BIN" \
  "${resume_args[@]}"
echo "[$(date --iso-8601=seconds)] COMPLETE model=$MODEL seed=$SEED"
