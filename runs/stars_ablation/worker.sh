#!/usr/bin/env bash
# Run *.req command files of queue dir $1 in name order on this GPU. A request whose lines "# needs: <path>"
# name missing files is skipped until they exist. Running files are *.running.<job>; *.done holds the exit code.
# Exits after 90 s without a runnable request.
Q=$1; J=${SLURM_JOB_ID:-local}; idle=0
while [[ ! -f $(dirname $Q)/STOP ]]; do
  ran=0
  for r in $(ls $Q/*.req 2>/dev/null | sort); do
    ok=1
    for need in $(sed -n 's/^# needs: //p' "$r"); do [[ -e $need ]] || { ok=0; break; }; done
    [[ $ok == 1 ]] || continue
    mv "$r" "${r%.req}.running.$J" 2>/dev/null || continue
    echo "== $(date) $(hostname) gpu=$CUDA_VISIBLE_DEVICES job=$J" >> "${r%.req}.log"
    bash "${r%.req}.running.$J" >> "${r%.req}.log" 2>&1; code=$?
    [[ -f "${r%.req}.running.$J" ]] || exit 0   # requeued by the end trap
    echo $code > "${r%.req}.done"; rm -f "${r%.req}.running.$J"
    ran=1; break
  done
  if [[ $ran == 0 ]]; then idle=$((idle+5)); [[ $idle -ge 90 ]] && exit 0; sleep 5; else idle=0; fi
done
