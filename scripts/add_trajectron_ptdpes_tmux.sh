#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-ptdpes-transfer-v2}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="$PROJECT_DIR/scripts/pretrain_trajectron_ptdpes_seed.sh"
LOG_DIR="$PROJECT_DIR/logs/experiments"

command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
tmux has-session -t "=${SESSION_NAME}" 2>/dev/null || {
  echo "tmux session does not exist: $SESSION_NAME" >&2
  exit 1
}

# Batch-128 probing added about 0.48 GiB.  The other active experiments have
# variable memory footprints, so keep one Trajectron++ process per physical GPU
# rather than stacking two on a card whose free memory can change during a run.
gpu_assignments=(0 1 2)
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "$gpu_count" -ge 3 ]] || { echo "This conservative assignment requires three visible GPUs." >&2; exit 1; }

for seed in 0 1 2; do
  window="trajectron-s${seed}"
  if tmux list-windows -t "=${SESSION_NAME}" -F '#{window_name}' | grep -Fxq "$window"; then
    echo "tmux window already exists: $SESSION_NAME:$window" >&2
    exit 1
  fi
  gpu_index="${gpu_assignments[$seed]}"
  printf -v command 'cd %q && exec %q %q %q' \
    "$PROJECT_DIR" "$WORKER" "$seed" "$gpu_index"
  tmux new-window -d -t "=${SESSION_NAME}" -n "$window"
  tmux set-option -w -t "=${SESSION_NAME}:=${window}" remain-on-exit on
  tmux send-keys -t "=${SESSION_NAME}:=${window}" -l "$command"
  tmux send-keys -t "=${SESSION_NAME}:=${window}" Enter
done

mkdir -p "$LOG_DIR"
monitor_window="gpu-monitor"
if ! tmux list-windows -t "=${SESSION_NAME}" -F '#{window_name}' | grep -Fxq "$monitor_window"; then
  monitor_log="$LOG_DIR/${SESSION_NAME}-gpu.log"
  printf -v monitor_command \
    'while pgrep -f %q >/dev/null; do date --iso-8601=seconds; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader; sleep 60; done 2>&1 | tee -a %q' \
    '[b]aselines.train_(moflow|social_vae|trajectron)' "$monitor_log"
  tmux new-window -d -t "=${SESSION_NAME}" -n "$monitor_window"
  tmux set-option -w -t "=${SESSION_NAME}:=${monitor_window}" remain-on-exit on
  tmux send-keys -t "=${SESSION_NAME}:=${monitor_window}" -l "$monitor_command"
  tmux send-keys -t "=${SESSION_NAME}:=${monitor_window}" Enter
fi

echo "Added Trajectron++ PTDPES seeds 0, 1, and 2 to '$SESSION_NAME'."
echo "Assignments: seed0->GPU0, seed1->GPU1, seed2->GPU2; batch size 128."
echo "Attach with: tmux attach-session -t '$SESSION_NAME'"
