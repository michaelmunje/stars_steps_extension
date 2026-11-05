#!/usr/bin/env bash
# Judged CMA-ES on Horizon. Compute nodes have no internet, so per generation the rollouts and pictures
# run as a GPU job (`ask`, sbatch --wait) and the judging and CMA-ES update run here, on the login node
# (`judge`, `tell`: no GPU, no STARS weights). Resumable: rerun the same command after an interruption.
#
#   tmux new -s judged          # from the repository root, after `uv sync` and the weights bundle
#   scripts/horizon/judged_loop.sh runs/simplenavsim/judged_stars
#
# Environment overrides: CONFIG, POLICY (start point; empty = controller defaults), JUDGE (claude-code |
# claude | file), MODEL (judge model), SBATCH_ARGS (e.g. "--account=XYZ --time=01:00:00").
set -euo pipefail

OUT=${1:?usage: $0 OUT_DIR}
CONFIG=${CONFIG:-simplenavsim/configs/judged_cmaes_stars.yaml}
POLICY=${POLICY-checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml}
JUDGE=${JUDGE:-claude-code}
S=simplenavsim/scripts/judged_cmaes.py
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
model_args=(); [[ -n "${MODEL:-}" ]] && model_args=(--model "$MODEL")
start_args=(--config "$CONFIG"); [[ -n "$POLICY" ]] && start_args+=(--policy "$POLICY")
gpu() { sbatch --wait ${SBATCH_ARGS:-} --job-name=judged scripts/horizon/gpu_jobs.slurm "$1"; }

[[ -f "$OUT/state.pkl" ]] || python "$S" status --out-dir "$OUT" "${start_args[@]}" --device cpu >/dev/null
while true; do
  read -r told pending planned < <(python "$S" status --out-dir "$OUT")
  if [[ "$pending" == 0 ]]; then
    (( told >= planned )) && break
    echo "== generation $((told + 1)) / $planned: rollouts on a GPU node"
    gpu "python $S ask --out-dir $OUT --device cuda"
  fi
  read -r told pending planned < <(python "$S" status --out-dir "$OUT")
  if [[ ! -f "$OUT/gen_$(printf %02d "$pending")/labels.json" ]]; then
    echo "== generation $pending: $JUDGE labels the pictures"
    python "$S" judge --out-dir "$OUT" --judge "$JUDGE" "${model_args[@]}"
  fi
  python "$S" tell --out-dir "$OUT"
done
echo "== report (held-out test scenes on a GPU node)"
gpu "python $S report --out-dir $OUT --device cuda"
echo "done: $OUT/summary.json, $OUT/convergence.png, $OUT/test/"
