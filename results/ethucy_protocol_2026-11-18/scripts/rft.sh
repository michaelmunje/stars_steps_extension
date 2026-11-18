#!/usr/bin/env bash
# Reusable closed-loop rollout fine-tuning of ANY multi-agent STARS forecast checkpoint.
# Usage: rft.sh INIT_CHECKPOINT RUN_DIR [extra train_stars_ma flags, e.g. --datasets ucy,eth or --config X --epochs N]
# Defaults: data configs/stars_ma/rft_no_atc.yaml (override with --config/--datasets/--val-datasets), the
# checkpoint's own graph (knn/radius read from it), 12 epochs x 300 batches of 2048 pedestrians, LR 5e-5,
# K = 4 -> 12 closed-loop steps over 3 epochs, prior latents with AR(1) noise, progress-matching term 1.0,
# selection on the 12-step closed-loop validation ADE (epoch 0 = the input checkpoint, so best.pt is never worse).
# Resumable (re-run the same command); a finished run is skipped.
set -euo pipefail
cd /scratch/09865/mmunje/stars-ethucy-protocol
init=$1; run=$2; shift 2
resume=""; [ -f $run/checkpoints/latest.pt ] && resume=--resume
if [ -f $run/status.json ] && grep -q '"status": "complete"' $run/status.json; then echo "complete: $run"; exit 0; fi
graph=$(python -c "import torch,sys; g=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['graph']; \
print(f\"--knn {g['knn']} --radius {g['radius']}\" if g['knn'] is not None else '--fully-connected')" $init)
# The checkpoint's own open-loop loss terms (KL weight, collision hinge, best-of-many) carry over to the regulariser.
inherit=$(python -c "import torch,sys; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); c=p.get('run_config',{}); \
print(f\"--kl-weight {p['recipe']['kl_weight']} --collision-weight {c.get('collision_weight',0.0)} --collision-radius {c.get('collision_radius',0.4)} \
--bom-weight {c.get('bom_weight',0.0)} --bom-samples {c.get('bom_samples',4)}\" + ('' if c.get('collision_fixed_prior',True) else ' --no-collision-fixed-prior'))" $init)
python -m baselines.train_stars_ma --phase forecast --config configs/stars_ma/rft_no_atc.yaml --encoder-mode scratch \
  --init-checkpoint $init $graph $inherit --kl-warmup-epochs 0 \
  --epochs 12 --max-nodes 2048 --learning-rate 5e-5 --lr-scaling none --warmup-epochs 0.2 --weight-decay 0.00217 \
  --rft-steps 12 --rft-start-steps 4 --rft-ramp-epochs 3 --rft-speed-weight 1.0 \
  --rft-eval-steps 12 --selection closedloop --samples 20 --train-batches 300 --val-batches 150 \
  --seed 42 --data-seed 42 --cache-workers 32 --eval-every 1 "$@" $resume --run-dir $run
