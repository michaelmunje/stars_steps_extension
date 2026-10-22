#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-ptdpes-transfer-v2}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="$PROJECT_DIR/scripts/pretrain_ptdpes_transfer_seed.sh"

command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "$gpu_count" -gt 0 ]] || { echo "No NVIDIA GPUs are visible." >&2; exit 1; }

first=true
for model in moflow socialvae; do
  for seed in 0 1 2; do
    gpu_index="$(( seed % gpu_count ))"
    window="${model}-s${seed}"
    printf -v command 'cd %q && exec %q %q %q %q' \
      "$PROJECT_DIR" "$WORKER" "$model" "$seed" "$gpu_index"
    if "$first"; then
      tmux new-session -d -s "$SESSION_NAME" -n "$window"
      first=false
    else
      tmux new-window -d -t "=${SESSION_NAME}" -n "$window"
    fi
    tmux set-option -w -t "=${SESSION_NAME}:=${window}" remain-on-exit on
    tmux send-keys -t "=${SESSION_NAME}:=${window}" -l "$command"
    tmux send-keys -t "=${SESSION_NAME}:=${window}" Enter
  done
done

echo "Started MoFlow and SocialVAE PTDPES seeds 0, 1, and 2 in '$SESSION_NAME'."
echo "Each physical GPU runs one MoFlow and one SocialVAE process."
echo "Attach with: tmux attach-session -t '$SESSION_NAME'"
