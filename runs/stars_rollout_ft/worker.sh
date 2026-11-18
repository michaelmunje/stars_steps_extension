#!/usr/bin/env bash
# Run the *.req command files in queue dir $1, oldest first, on this GPU; *.done gets the exit code.
# A file being run is renamed *.running.<job id>; the job's end trap puts unfinished ones back as *.req.
# Exits when the queue is empty (so an idle node ends instead of waiting for its time limit).
Q=$1; J=${SLURM_JOB_ID:-local}
while [[ ! -f $(dirname $Q)/STOP ]]; do
  r=$(ls -tr $Q/*.req 2>/dev/null | head -1)
  [[ -z $r ]] && exit 0
  mv "$r" "${r%.req}.running.$J" 2>/dev/null || continue
  bash "${r%.req}.running.$J" >> "${r%.req}.log" 2>&1; code=$?
  [[ -f "${r%.req}.running.$J" ]] || exit 0   # requeued by the end trap
  echo $code > "${r%.req}.done"; rm -f "${r%.req}.running.$J"
done
