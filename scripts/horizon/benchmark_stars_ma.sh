#!/usr/bin/env bash
# Throughput sweep for multi-agent STARS training on one GB200 node; run inside an allocation:
#   sbatch --time=02:00:00 scripts/horizon/gpu_jobs.slurm "scripts/horizon/benchmark_stars_ma.sh"
# Builds every dataset's train/val window cache first (configs/stars_ma/all.yaml), then times
# --benchmark steps for each phase/decoder at several per-GPU node budgets, bf16 vs fp32,
# torch.compile, and 4-GPU DDP. One JSON line per configuration goes to $OUT.
set -uo pipefail
OUT="${OUT:-experiments/stars_ma/benchmark/$(date +%Y%m%d-%H%M%S).jsonl}"
STEPS="${BENCH_STEPS:-40}"
ROOT_RUN="$(dirname "$OUT")/runs-$(basename "$OUT" .jsonl)"
mkdir -p "$(dirname "$OUT")" "$ROOT_RUN"
export CUDA_VISIBLE_DEVICES="${ALL_GPUS:-0,1,2,3}"

run() {  # name, launcher ("1" or "4" GPUs), trainer args...
  local name="$1" gpus="$2"; shift 2
  local dir="$ROOT_RUN/$name"
  local start=$(date +%s)
  if [[ "$gpus" == 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 python -m baselines.train_stars_ma --run-dir "$dir" --benchmark "$STEPS" "$@" > "$dir.log" 2>&1
  else
    torchrun --standalone --nproc-per-node "$gpus" -m baselines.train_stars_ma --run-dir "$dir" --benchmark "$STEPS" "$@" > "$dir.log" 2>&1
  fi
  local code=$?
  python - "$name" "$gpus" "$dir" "$code" "$(( $(date +%s) - start ))" "$*" >> "$OUT" <<'PY'
import json, sys
name, gpus, path, code, wall, argv = sys.argv[1:]
try:
    status = json.load(open(f"{path}/status.json"))
except Exception:
    status = {}
keys = ("nodes_per_second", "edges_per_second", "steps_per_second", "peak_memory_gb", "train_loss", "status", "error")
print(json.dumps({"name": name, "gpus": int(gpus), "exit": int(code), "wall_seconds": int(wall), "args": argv,
                  **{k: status.get(k) for k in keys}}))
PY
  tail -1 "$OUT"
}

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 1. Window caches for every dataset (also times all-dataset batches with large scenes).
run cache_all 1 --config configs/stars_ma/all.yaml --phase ssl --max-nodes 16384
# 2. Single GPU: node budget sweep per training mode.
SSL="--config configs/stars_ma/steps.yaml --phase ssl"
DET="--config configs/stars_ma/steps.yaml --phase forecast --encoder-mode scratch"
CVAE="$DET --decoder cvae"
for nodes in 4096 16384 65536; do
  run "ssl_n$nodes" 1 $SSL --max-nodes $nodes
  run "det_n$nodes" 1 $DET --max-nodes $nodes
  run "cvae_n$nodes" 1 $CVAE --max-nodes $nodes
  run "cvae_rollout4_n$nodes" 1 $CVAE --rollout-exec-steps 4 --max-nodes $nodes
done
run ssl_n16384_fp32 1 $SSL --max-nodes 16384 --no-amp
run ssl_n16384_compile 1 $SSL --max-nodes 16384 --compile
run cvae_n16384_compile 1 $CVAE --max-nodes 16384 --compile
run ssl_fully_connected_n16384 1 $SSL --max-nodes 16384 --fully-connected
# 3. Four GPUs, data parallel.
run ssl_n16384_ddp4 4 $SSL --max-nodes 16384
run cvae_n16384_ddp4 4 $CVAE --max-nodes 16384
run cvae_rollout4_n16384_ddp4 4 $CVAE --rollout-exec-steps 4 --max-nodes 16384
run all_ssl_n16384_ddp4 4 --config configs/stars_ma/all.yaml --phase ssl --max-nodes 16384
echo "results: $OUT"
