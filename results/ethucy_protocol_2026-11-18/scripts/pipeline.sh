#!/usr/bin/env bash
# Login-node orchestration: when a pretraining seed completes, queue its zero-shot predictions and LOSO sweep;
# when every sweep run is done, select configs on val and queue the LOSO predictions.
cd /scratch/09865/mmunje/stars-ethucy-protocol
source /scratch/09865/mmunje/stars-forecasting/.venv/bin/activate; export PYTHONPATH=src
R=runs/ethucy_protocol; SEEDS="42 1 2 3"; queued=""; k=0
while :; do
  for s in $SEEDS; do
    [[ " $queued " == *" $s "* ]] && continue
    if grep -q '"status": "complete"' experiments/stars_ma/runs/ep_pre4_s$s/status.json 2>/dev/null; then
      echo "bash $R/predict.sh ep_zs_s$s ep_zs_s${s}_mode ep_zs_s${s}_os4" > $R/q$((k % 8))/25_pred_zs_s$s.req
      bash $R/queue_sweep.sh $s $((k + 1)); queued="$queued $s"; k=$((k + 3)); echo "$(date +%H:%M) queued seed $s"
    fi
  done
  n=$(ls experiments/forecast_bench/loso/sweep/loso_s*/*/*/summary.json 2>/dev/null | wc -l)
  [ $n -ge 240 ] && [ $(echo $queued | wc -w) -eq 4 ] && break
  sleep 120
done
python -m forecast_bench.protocol select > $R/logs/select.log 2>&1
i=0; for s in $SEEDS; do echo "bash $R/predict.sh ep_loso_s$s ep_loso_s${s}_mode ep_loso_s${s}_os4" > $R/q$i/40_pred_loso_s$s.req; i=$((i+1)); done
echo "bash $R/predict.sh ep_scratch_s0 ep_scratch_s0_mode ep_scratch_s0_os4" > $R/q4/40_pred_scratch_s0.req
echo "$(date +%H:%M) queued LOSO predictions"
