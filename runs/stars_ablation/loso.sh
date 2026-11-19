#!/usr/bin/env bash
# bash runs/stars_ablation/loso.sh <run name> <seed> <E0..E3> [finetune_stars_loo flags]: the five LOSO folds of one
# ladder configuration in parallel on this GPU, then K=20 predictions (+ prior-mean "_mode" for CVAEs) on the test scenes.
# Shared recipe (stars_improved): AdamW lr 3e-4 (override with --lr), weight decay 0.00217, 2 warm-up epochs, cosine
# to 0 over 100 epochs, early stopping after 20 epochs without a validation gain, best epoch by validation
# single-sample ADE (mean over 20 draws), decoder 512 hidden / dropout 0.0633 / CVAE latent 64. Resumable.
N=$1; S=$2; E=$3; shift 3
ENC=experiments/stars_ablation/ssl/${E}_s$S${ENC_SUFFIX-_long}/checkpoints/best.pt   # decoders use the 40-epoch encoders
OUT=experiments/forecast_bench/ablation/loso/${N}_s$S
mkdir -p $OUT
for f in eth hotel univ zara1 zara2; do
  python -m forecast_bench.finetune_stars_loo --fold $f --encoder $ENC --seed $S --out $OUT/$f --epochs 100 --patience 20 \
    --lr 3e-4 --weight-decay 0.00217 --warmup-epochs 2 --select mean_ade "$@" > $OUT/$f.log 2>&1 &
done
wait
for f in eth hotel univ zara1 zara2; do [[ -f $OUT/$f/summary.json ]] || { echo "fold $f incomplete"; exit 1; }; done
SETS=ethucy_eth,ethucy_hotel,ethucy_univ,ethucy_zara1,ethucy_zara2
python -m forecast_bench.predict --method abl_${N}_s$S --sets $SETS --skip-existing || exit 1
if python -c "import torch,sys; sys.exit(0 if torch.load('$OUT/eth/best.pt',map_location='cpu',weights_only=False)['decoder_recipe']['latent_dim'] else 1)"; then
  python -m forecast_bench.predict --method abl_${N}_s${S}_mode --sets $SETS --skip-existing || exit 1
fi
