#!/usr/bin/env bash
# Experiment 1, continued: judged CMA-ES until END (epoch seconds), checking subscription usage after every
# generation (stop at SESSION_PCT of the 5-hour session or WEEK_PCT of the week); COST_CAP (estimated USD) is a backstop.
set -uo pipefail
cd /scratch/09865/mmunje/stars-forecasting
source .venv/bin/activate; export PYTHONPATH="$PWD/src"
A=runs/simplenavsim/agent_judge; E=$A/exp1_judged; S=simplenavsim/scripts/judged_cmaes.py
END=${END:?}; SESSION_PCT=${SESSION_PCT:-60}; WEEK_PCT=${WEEK_PCT:-20}; COST_CAP=${COST_CAP:-15}; MODEL=${MODEL:-sonnet}
while (( $(date +%s) < END )) && [[ ! -f $A/STOP_EXP1 ]]; do
  read -r told pending planned < <(python $S status --out-dir $E)
  (( told >= planned )) && break
  [[ "$pending" == 0 ]] && python $S ask --out-dir $E --device cpu >/dev/null
  read -r told pending planned < <(python $S status --out-dir $E)
  python $S judge --out-dir $E --judge claude-code --model "$MODEL" --judge-timeout 900 || { echo "judge failed"; break; }
  python $S tell --out-dir $E
  cost=$(python $A/usage.py $E | python -c "import sys,json; print(json.load(sys.stdin)['cost_usd'])")
  guard=$(python $A/usage_guard.py "$SESSION_PCT" "$WEEK_PCT"); code=$?
  echo "$(date +%T) generation $pending told; judge cost so far \$$cost; usage $guard" | tee -a $A/usage_log.jsonl >/dev/null
  echo "$(date +%T) generation $pending told; judge cost so far \$$cost; usage $guard"
  (( code == 1 )) && { echo "usage limit reached: stopping both experiments"; touch $A/exp2_costfn/STOP; break; }
  python -c "import sys; sys.exit(0 if $cost < $COST_CAP else 1)" || { echo "cost backstop \$$COST_CAP reached"; break; }
done
echo "$(date +%T) exp1 stopped"
