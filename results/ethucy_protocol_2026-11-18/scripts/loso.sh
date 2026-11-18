#!/usr/bin/env bash
# One LOSO run. Usage: loso.sh KIND SEED FOLD CONFIG
#   KIND=loso: fine-tune the protocol-A checkpoint ep_pre4_s<SEED>; KIND=scratch: same architecture, re-initialised.
#   KIND=loso_rft: closed-loop rollout fine-tuning of the selected LOSO checkpoint loso/ep_loso_s<SEED>/<FOLD>
#   on the same fold's training scenes (selection: 12-step closed-loop ADE on the fold's val split).
#   CONFIG: lr<LR>_enc<MULT>_e<EPOCHS> (fine-tune) or lr<LR>_e<EPOCHS> (scratch, loso_rft).
set -euo pipefail
cd /scratch/09865/mmunje/stars-ethucy-protocol
kind=$1; seed=$2; fold=$3; cfg=$4
lr=$(sed -E 's/^lr([^_]+)_.*/\1/' <<< $cfg); ep=$(sed -E 's/.*_e([0-9]+)$/\1/' <<< $cfg)
out=experiments/forecast_bench/loso/sweep/${kind}_s${seed}/$cfg/$fold
if [ $kind = loso ]; then
  enc=$(sed -E 's/.*_enc([^_]+)_.*/\1/' <<< $cfg)
  python -m forecast_bench.finetune_stars_loo --fold $fold --init experiments/stars_ma/runs/ep_pre4_s$seed/checkpoints/best.pt \
    --out $out --lr $lr --encoder-lr-mult $enc --epochs $ep --warmup-epochs 1 --patience 8 --kl-weight 0.0799 \
    --weight-decay 1e-4 --seed $seed
elif [ $kind = loso_rft ]; then
  python -m forecast_bench.finetune_stars_loo --fold $fold --init experiments/forecast_bench/loso/ep_loso_s$seed/$fold/best.pt \
    --out $out --lr $lr --epochs $ep --warmup-epochs 0.2 --kl-weight 0.0799 --weight-decay 0.00217 --seed $seed \
    --rft-steps 12 --rft-start-steps 4 --rft-ramp-epochs 3 --rft-speed-weight 1.0
else   # architecture/recipe only from the pretraining recipe's checkpoint; weights re-initialised
  python -m forecast_bench.finetune_stars_loo --fold $fold --init experiments/stars_ma/runs/si_final_base_s2/checkpoints/best.pt \
    --scratch --out $out --lr $lr --epochs $ep --warmup-epochs 2 --kl-warmup-epochs 10 --patience 30 --kl-weight 0.0799 \
    --weight-decay 0.00217 --seed $seed
fi
