#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=""
DATASET=""
SEED=""
DATA_SEED=42
DEVICE="cuda"
EPOCHS=""
BATCH_SIZE=""
NUM_WORKERS=4
ITERATIONS=""
MAX_VAL_BATCHES=""
PATIENCE=""
MIN_DELTA=0.0001
FORECAST_INPUT_PREPROCESSING="raw"
FORECAST_TARGET_PREPROCESSING="raw"
OUTPUT="experiments/pretrained"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --data-seed) DATA_SEED="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --iterations) ITERATIONS="$2"; shift 2;;
    --max-val-batches) MAX_VAL_BATCHES="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --min-delta) MIN_DELTA="$2"; shift 2;;
    --forecast-input-preprocessing) FORECAST_INPUT_PREPROCESSING="$2"; shift 2;;
    --forecast-target-preprocessing) FORECAST_TARGET_PREPROCESSING="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --resume|--resuming)
      if [[ $# -gt 1 && "$2" != --* ]]; then RESUME="$2"; shift 2; else RESUME="auto"; shift; fi;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

[[ -n "$MODEL" ]] || { echo "--model is required" >&2; exit 2; }
[[ -n "$DATASET" ]] || { echo "--dataset is required (atc, ptdpes, or steps)" >&2; exit 2; }
[[ -n "$SEED" ]] || { echo "--seed is required" >&2; exit 2; }
case "$DATASET" in
  atc|ptdpes|steps) ;;
  *) echo "Unknown dataset '$DATASET'; choose atc, ptdpes, or steps" >&2; exit 2;;
esac
if [[ -z "$BATCH_SIZE" ]]; then
  case "$MODEL" in
    trajectron) ;;  # no shared config object; defer to its own argparse defaults
    moflow) BATCH_SIZE=48;;
    agentformer) BATCH_SIZE=1;;
    socialvae|social_vae) BATCH_SIZE=128;;
  esac
fi
if [[ -z "$EPOCHS" ]]; then
  case "$MODEL" in
    socialvae|social_vae) EPOCHS=600;;
    moflow) EPOCHS=150;;
    *) EPOCHS=100;;
  esac
fi
mkdir -p "$OUTPUT"
export TRAJECTORY_MODEL_REGISTRY="${TRAJECTORY_MODEL_REGISTRY:-experiments/model_registry}"

MODEL_TAG="$MODEL"
case "$MODEL" in
  moflow|socialvae|social_vae) MODEL_TAG="${MODEL}_transfer_v1";;
esac
run_dir="$OUTPUT/${MODEL_TAG}_${DATASET}_${SEED}"
mkdir -p "$run_dir"
printf 'MODEL=%q\nDATASET=%q\nSEED=%q\nDATA_SEED=%q\nDEVICE=%q\nEPOCHS=%q\nBATCH_SIZE=%q\nNUM_WORKERS=%q\nITERATIONS=%q\nMAX_VAL_BATCHES=%q\nPATIENCE=%q\nMIN_DELTA=%q\nFORECAST_INPUT_PREPROCESSING=%q\nFORECAST_TARGET_PREPROCESSING=%q\nOUTPUT=%q\nPYTHON_BIN=%q\nRESUME=%q\n' \
  "$MODEL" "$DATASET" "$SEED" "$DATA_SEED" "$DEVICE" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$ITERATIONS" "$MAX_VAL_BATCHES" "$PATIENCE" "$MIN_DELTA" "$FORECAST_INPUT_PREPROCESSING" "$FORECAST_TARGET_PREPROCESSING" "$OUTPUT" "$PYTHON_BIN" "$RESUME" > "$run_dir/pretrain_params.env"

optional_training_args=()
iteration_args=()
early_stopping_args=()
resume_args=()
if [[ "$RESUME" == "auto" ]]; then resume_args+=(--resume); elif [[ -n "$RESUME" ]]; then resume_args+=(--resume "$RESUME"); fi
if [[ -n "$PATIENCE" ]]; then early_stopping_args+=(--patience "$PATIENCE" --min-delta "$MIN_DELTA"); fi
preprocessing_args=(--forecast-input-preprocessing "$FORECAST_INPUT_PREPROCESSING" --forecast-target-preprocessing "$FORECAST_TARGET_PREPROCESSING")
# BATCH_SIZE is left empty for trajectron when the user does not pass --batch-size, so its
# own argparse default (train_trajectron.py) takes effect instead of a shell fallback.
if [[ "$MODEL" != "agentformer" && -n "$BATCH_SIZE" ]]; then optional_training_args+=(--batch-size "$BATCH_SIZE"); fi
epoch_args=()
if [[ -n "$EPOCHS" ]]; then epoch_args+=(--epochs "$EPOCHS"); fi
if [[ -n "$ITERATIONS" ]]; then
  optional_training_args+=(--iterations "$ITERATIONS")
  iteration_args+=(--iterations "$ITERATIONS")
fi

case "$MODEL" in
  trajectron)
    if [[ "$FORECAST_INPUT_PREPROCESSING" != "raw" || "$FORECAST_TARGET_PREPROCESSING" != "raw" ]]; then echo "Trajectron++'s pinned environment adapter supports only raw leakage-safe canonical inputs/targets" >&2; exit 2; fi
    "$PYTHON_BIN" -m baselines.train_trajectron --stage pretrain --pretrain-dataset "$DATASET" --seed "$SEED" --data-seed "$DATA_SEED" "${epoch_args[@]}" "${optional_training_args[@]}" "${early_stopping_args[@]}" "${resume_args[@]}" --device "$DEVICE" --run-dir "$run_dir"
    "$PYTHON_BIN" -m baselines.registry add "trajectron_${DATASET}_${SEED}" "$run_dir/checkpoints/best.pt" --model trajectron --dataset "$DATASET" --seed "$SEED"
    ;;
  moflow)
    "$PYTHON_BIN" -m baselines.train_moflow --stage pretrain --pretrain-dataset "$DATASET" --seed "$SEED" --data-seed "$DATA_SEED" --epochs "$EPOCHS" "${optional_training_args[@]}" "${early_stopping_args[@]}" "${preprocessing_args[@]}" "${resume_args[@]}" --device "$DEVICE" --run-dir "$run_dir"
    "$PYTHON_BIN" -m baselines.registry add "moflow_transfer_v1_${DATASET}_${SEED}" "$run_dir/checkpoints/best.pt" --model moflow --dataset "$DATASET" --seed "$SEED"
    ;;
  agentformer)
    vae_dir="$run_dir/vae"; dlow_dir="$run_dir/dlow"
    [[ "$BATCH_SIZE" == "1" ]] || { echo "AgentFormer requires --batch-size 1 (upstream batching is one scene)" >&2; exit 2; }
    "$PYTHON_BIN" -m baselines.train_agentformer --stage pretrain --pretrain-dataset "$DATASET" --seed "$SEED" --data-seed "$DATA_SEED" --epochs "$EPOCHS" "${iteration_args[@]}" "${early_stopping_args[@]}" "${preprocessing_args[@]}" --device "$DEVICE" --run-dir "$vae_dir"
    "$PYTHON_BIN" -m baselines.registry add "agentformer_vae_${DATASET}_${SEED}" "$vae_dir/checkpoints/best.pt" --model agentformer_vae --dataset "$DATASET" --seed "$SEED"
    "$PYTHON_BIN" -m baselines.train_agentformer_dlow --stage pretrain --pretrain-dataset "$DATASET" --seed "$SEED" --data-seed "$DATA_SEED" --epochs "$EPOCHS" "${iteration_args[@]}" "${early_stopping_args[@]}" "${preprocessing_args[@]}" --device "$DEVICE" --predictor-checkpoint "agentformer_vae_${DATASET}_${SEED}" --run-dir "$dlow_dir"
    "$PYTHON_BIN" -m baselines.registry add "agentformer_dlow_${DATASET}_${SEED}" "$dlow_dir/checkpoints/best.pt" --model agentformer_dlow --dataset "$DATASET" --seed "$SEED"
    ;;
  socialvae|social_vae)
    val_args=(); [[ -z "$MAX_VAL_BATCHES" ]] || val_args+=(--max-val-batches "$MAX_VAL_BATCHES")
    "$PYTHON_BIN" -m baselines.train_social_vae --stage pretrain --pretrain-dataset "$DATASET" --seed "$SEED" --data-seed "$DATA_SEED" --epochs "$EPOCHS" --num-workers "$NUM_WORKERS" "${optional_training_args[@]}" "${val_args[@]}" "${early_stopping_args[@]}" "${preprocessing_args[@]}" "${resume_args[@]}" --device "$DEVICE" --run-dir "$run_dir"
    "$PYTHON_BIN" -m baselines.registry add "socialvae_transfer_v1_${DATASET}_${SEED}" "$run_dir/checkpoints/best.pt" --model social_vae --dataset "$DATASET" --seed "$SEED"
    ;;
  *) echo "Unknown model '$MODEL'; choose trajectron, moflow, agentformer, or socialvae" >&2; exit 2;;
esac

echo "Pretraining complete: $run_dir"
