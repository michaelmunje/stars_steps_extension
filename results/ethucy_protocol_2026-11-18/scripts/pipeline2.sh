#!/usr/bin/env bash
# Stage 2 (after pipeline.sh): closed-loop rollout fine-tuning (RFT) of the best pretraining seed (A + RFT, on the
# 4 pretraining datasets) and of its LOSO checkpoints (B + RFT, each on its fold's 4 training scenes only),
# their predictions, then the closed-loop ETH/UCY realism runs of A, B, A + RFT, B + RFT.
cd /scratch/09865/mmunje/stars-ethucy-protocol
source /scratch/09865/mmunje/stars-forecasting/.venv/bin/activate; export PYTHONPATH=src
R=runs/ethucy_protocol
while pgrep -f "bash pipeline.sh" >/dev/null; do sleep 120; done
B=$(python -c "
import json
s={k: json.load(open(f'experiments/stars_ma/runs/ep_pre4_s{k}/status.json'))['best_selection'] for k in (42,1,2,3)}
print(min(s, key=s.get))")
echo "$(date +%H:%M) best pretraining seed $B" ; echo $B > $R/BEST_SEED
echo "bash $R/rft.sh experiments/stars_ma/runs/ep_pre4_s$B/checkpoints/best.pt experiments/stars_ma/runs/ep_rft_zs_s$B --config configs/stars_ma/four_no_ethucy.yaml" > $R/q5/50_rft_zs.req
i=0; for f in eth hotel univ zara1 zara2; do echo "bash $R/loso.sh loso_rft $B $f lr5e-5_e12" > $R/q$((6 + i % 2))/50_loso_rft_$f.req; i=$((i+1)); done
until grep -q '"status": "complete"' experiments/stars_ma/runs/ep_rft_zs_s$B/status.json 2>/dev/null && \
      [ $(ls experiments/forecast_bench/loso/sweep/loso_rft_s$B/*/*/summary.json 2>/dev/null | wc -l) -ge 5 ]; do sleep 120; done
python -m forecast_bench.protocol select > $R/logs/select2.log 2>&1
echo "bash $R/predict.sh ep_zs_rft_s$B ep_zs_rft_s${B}_mode ep_zs_rft_s${B}_os4 ep_loso_rft_s$B ep_loso_rft_s${B}_mode ep_loso_rft_s${B}_os4" > $R/q0/60_pred_rft.req
cat > $R/q1/60_closed_loop.req <<EOF
cd /scratch/09865/mmunje/stars-ethucy-protocol
EP_SEED=$B python simplenavsim/scripts/ethucy_benchmark.py run --run results/ethucy_2026-11-16_improved --append --device cuda --seeds 5 \
  --methods stars_ep_zs stars_ep_zs_mode stars_ep_loso stars_ep_loso_mode stars_ep_zs_rft stars_ep_loso_rft
EOF
echo "$(date +%H:%M) queued RFT predictions and closed loop"
