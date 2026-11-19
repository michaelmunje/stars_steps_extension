#!/usr/bin/env bash
# Two-stage vs end-to-end (high priority): A' seed s starts when the SSL encoder E3_s<s>_long is complete
# (fine-tuned encoder for s = 1, 2, 3 on q0-q2; frozen-encoder variant for s = 1 on q3); when an A' run
# completes, its zero-shot predictions and LOSO sweep (B') are queued; scratch seeds 1, 2 predictions once their
# sweeps finish; after all B' sweeps, select on val and queue B' predictions.
cd /scratch/09865/mmunje/stars-ethucy-protocol
source /scratch/09865/mmunje/stars-forecasting/.venv/bin/activate; export PYTHONPATH=src
R=runs/ethucy_protocol; SSL=/scratch/09865/mmunje/stars-ablation/experiments/stars_ablation/ssl
started=""; done_=""; scratch_done=0
done_ssl() { grep -q '"status": "complete"' $SSL/E3_s$1_long/status.json 2>/dev/null; }
done_run() { grep -q '"status": "complete"' experiments/stars_ma/runs/$1/status.json 2>/dev/null; }
while :; do
  for s in 1 2 3; do
    if [[ " $started " != *" $s "* ]] && done_ssl $s; then
      echo "bash $R/pretrain_ts.sh $s finetune" > $R/q$((s - 1))/05_pre_ts_s$s.req
      [ $s = 1 ] && echo "bash $R/pretrain_ts.sh 1 frozen" > $R/q3/05_pre_tsf_s1.req
      started="$started $s"; echo "$(date +%H:%M) started A' seed $s"
    fi
    if [[ " $done_ " != *" $s "* ]] && done_run ep_ts_pre4_s$s; then
      echo "bash $R/predict.sh ep_zs_ts_s$s ep_zs_ts_s${s}_mode ep_zs_ts_s${s}_os4" > $R/q$((s - 1))/06_pred_zs_ts_s$s.req
      bash $R/queue_sweep.sh $s $((4 + s)) loso_ts; done_="$done_ $s"; echo "$(date +%H:%M) A' seed $s done, B' sweep queued"
    fi
  done
  if [ $scratch_done = 0 ] && [ $(ls experiments/forecast_bench/loso/sweep/scratch_s{1,2}/*/*/summary.json 2>/dev/null | wc -l) -ge 30 ]; then
    python -m forecast_bench.protocol select > $R/logs/select3.log 2>&1
    echo "bash $R/predict.sh ep_scratch_s1 ep_scratch_s1_mode ep_scratch_s1_os4 ep_scratch_s2 ep_scratch_s2_mode ep_scratch_s2_os4" > $R/q4/23_pred_scratch12.req
    scratch_done=1; echo "$(date +%H:%M) scratch seeds 1, 2 predictions queued"
  fi
  if done_run ep_tsf_pre4_s1 && [ ! -f $R/.tsf_pred ]; then
    echo "bash $R/predict.sh ep_zs_tsf_s1 ep_zs_tsf_s1_mode ep_zs_tsf_s1_os4" > $R/q3/06_pred_zs_tsf_s1.req; touch $R/.tsf_pred
  fi
  [ $(ls experiments/forecast_bench/loso/sweep/loso_ts_s*/*/*/summary.json 2>/dev/null | wc -l) -ge 180 ] && break
  sleep 120
done
python -m forecast_bench.protocol select > $R/logs/select4.log 2>&1
i=0; for s in 1 2 3; do echo "bash $R/predict.sh ep_loso_ts_s$s ep_loso_ts_s${s}_mode ep_loso_ts_s${s}_os4" > $R/q$i/07_pred_loso_ts_s$s.req; i=$((i+1)); done
echo "$(date +%H:%M) queued B' predictions"
