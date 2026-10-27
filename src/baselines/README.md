# Baseline integrations

Reference repositories are cloned under references/ and remain unmodified. Exact
commits are recorded in upstream.json. Integration code is restricted to this
directory, the unified dataset protocol, and run artifacts.

## ETH/UCY protocol

configs/protocols/eth_ucy_five_fold.json defines the five held-out locations:

- eth: seq_eth
- hotel: seq_hotel
- univ: students001 and students003
- zara1: crowds_zara01
- zara2: crowds_zara02

Only these five benchmark locations are used for forecasting. `arxiepiskopi1`,
`crowds_zara03`, and `uni_examples` are canonical and are included in the independent
UCY publication-characterization row, but are excluded from five-fold forecasting
training/evaluation. Publication membership never changes a forecasting fold.

## Forecast preprocessing presets

The common loader resolves one structured recipe with the precedence preset < config
file < explicit CLI. Use `python -m forecasting_config --list-forecast-presets` to see
the concise registry. Trajectron++ defaults to `trajectron_pp`; native project adapters
default to `steps_default`. The Social GAN data adapter is available in
`social_gan_data.py`. SocialVAE is a full train/evaluate integration using
`social_vae_data.py` and the pinned official model, without creating separately
converted datasets. Schöller's deterministic last-observed-displacement implementation is in
`scholler.py`.

Every run saves both the requested preset and every resolved field, a preprocessing
hash, whether the recipe is literature-exact, and the pinned upstream commit. Any
explicit override marks the recipe modified unless it resolves back to the exact
preset values. Sharded Trajectron++ pretraining also saves complete train/validation
shard manifests with each shard's realized example budget; the first in-memory shard
is only a type-template and is not mistaken for the full training population.

SocialVAE and MoFlow fine-tuning default to `--validation-mode grouped`: one complete
non-test scene sequence is reserved for validation and the held-out fold remains test
only. This is required for validation-selected checkpoints and early stopping. Scratch
and pretrained initialization use the identical target split and optimization recipe.

## Trajectron++ CVAE (ready, K=20)

The adapter constructs the official `Environment`/`Scene`/`Node` objects from
canonical 2.5 Hz Parquet tracks and calls the pinned model, CVAE loss, annealers,
gradient clipping, and exponential learning-rate schedule directly.

    python -m baselines.train_trajectron \
      --stage pretrain \
      --run-dir experiments/trajectron_ptdpes_pretrain \
      --device cuda

    python -m baselines.train_trajectron \
      --stage finetune --fold eth \
      --checkpoint experiments/trajectron_ptdpes_pretrain/checkpoints/best.pt \
      --run-dir experiments/trajectron_ptdpes_to_eth \
      --device cuda

    python -m baselines.train_trajectron \
      --stage scratch --fold eth \
      --run-dir experiments/trajectron_scratch_eth \
      --device cuda

Use `--max-recordings 1 --iterations 1 --device cpu` only as an integration
diagnostic. Full runs require a CUDA-visible process.

Evaluate the held-out location with `python -m baselines.evaluate_trajectron`.
Long PTDPES recordings are divided into overlapping 100-timestep scenes before
the official O(T*N^2) offline graph is built; the overlap preserves all 20-step
windows.

The realized training population admits 2-8 observed steps per node
(upstream-faithful `minimum_history_length`/`maximum_history_length` from the
pinned `eth_vel` reference config), while evaluation fixes an 8-step observed
history.

`--stage finetune` refuses a checkpoint unless it is `checkpoints/best.pt` from
a completed pretraining run with a matching `forecast_preprocessing_hash` and
pinned `upstream_commit`, and a recorded checkpoint-selection metric; pass
`--allow-legacy-checkpoint` only for an explicit compatibility run. Fine-tuning
also defaults `--seed` to the pretraining run's seed unless `--seed` is given
explicitly; pretraining and scratch training keep the previous default of 0.

## MoFlow (ready, teacher K=20)

The reported MoFlow implementation uses the official conditional flow-matching
teacher directly. It retains the upstream 20 proposal heads, tied noise, input
scaling, AdamW optimizer, 5% linear warmup followed by cosine decay, and flow sampler.
Pretraining defaults to 150 epochs, LR `1e-4`, weight decay `0.01`, batch 48, and no
early stopping. Fine-tuning and scratch share a 50-epoch target recipe with LR `2e-5`,
fresh optimizer/scheduler state, full-model training, and patience 10.

    python -m baselines.train_moflow --stage pretrain \
      --run-dir experiments/moflow_ptdpes_pretrain --device cuda
    python -m baselines.train_moflow --stage finetune \
      --target-dataset eth_ucy --fold eth --target-fraction 1.0 \
      --checkpoint experiments/moflow_ptdpes_pretrain/checkpoints/best.pt \
      --run-dir experiments/moflow_ptdpes_to_eth --device cuda
    python -m baselines.train_moflow --stage scratch \
      --target-dataset eth_ucy --fold eth --target-fraction 1.0 \
      --run-dir experiments/moflow_scratch_eth --device cuda
    python -m baselines.evaluate_moflow --fold eth \
      --checkpoint experiments/moflow_ptdpes_to_eth/checkpoints/best.pt \
      --run-dir experiments/moflow_ptdpes_to_eth_eval --device cuda

The optional IMLE distillation described by MoFlow is not required to evaluate
the flow model and is not currently the reported model. It is a speed-distilled
student of the trained teacher, not a separate forecasting architecture.

## SocialVAE (ready, K=20)

The adapter imports the unmodified model from the pinned official checkout. It preserves
the reference 8+12 position/velocity/acceleration state, per-step (not per-second)
derivatives, 2 m social radius, variable neighbor padding, Adam, best-of-20
evaluation, and training-only flip/continuous-rotation/`N(1, 0.05)` scale augmentation.
The upstream schedule uses 100 batches per epoch. Pretraining defaults to 600 such epochs
at LR `5e-4`, batch 128, without early stopping. Fine-tuning and scratch share a
200-epoch target recipe at LR `1e-4`, with fresh Adam state, all parameters trainable,
validation-ADE checkpoint selection, and patience 20.

    python -m baselines.train_social_vae --stage pretrain \
      --pretrain-dataset ptdpes --seed 0 \
      --run-dir experiments/socialvae_ptdpes_pretrain --device cuda
    python -m baselines.train_social_vae --stage finetune \
      --target-dataset eth_ucy --fold eth --target-fraction 1.0 \
      --checkpoint socialvae_transfer_v1_ptdpes_0 --seed 0 \
      --run-dir experiments/socialvae_ptdpes_to_eth --device cuda
    python -m baselines.train_social_vae --stage scratch \
      --target-dataset eth_ucy --fold eth --target-fraction 1.0 \
      --seed 0 --run-dir experiments/socialvae_scratch_eth --device cuda
    python -m baselines.evaluate_social_vae --fold eth \
      --checkpoint experiments/socialvae_ptdpes_to_eth/checkpoints/best.pt \
      --run-dir experiments/socialvae_ptdpes_to_eth_eval --device cuda

Pretraining through `scripts/pretrain_transfer_model.sh --model socialvae ...` registers IDs
such as `socialvae_transfer_v1_ptdpes_0`; one pretrained checkpoint can be fine-tuned independently
on every ETH/UCY held-out fold.

## AgentFormer (ready, VAE + DLow K=20)

AgentFormer follows the official two-stage procedure. First pretrain/fine-tune
the VAE predictor, then train/fine-tune DLow with that predictor checkpoint.
One upstream ETH architecture and `traj_scale=2` are frozen across folds because
the published fold configs vary architecture, which prevents cross-dataset
checkpoint transfer. Fold names select data only.

    python -m baselines.train_agentformer --stage pretrain \
      --run-dir experiments/agentformer_vae_ptdpes --device cuda
    python -m baselines.train_agentformer --stage finetune --fold eth \
      --checkpoint experiments/agentformer_vae_ptdpes/checkpoints/best.pt \
      --run-dir experiments/agentformer_vae_ptdpes_to_eth --device cuda
    python -m baselines.train_agentformer_dlow --stage pretrain \
      --predictor-checkpoint experiments/agentformer_vae_ptdpes/checkpoints/best.pt \
      --run-dir experiments/agentformer_dlow_ptdpes --device cuda
    python -m baselines.train_agentformer_dlow --stage finetune --fold eth \
      --predictor-checkpoint experiments/agentformer_vae_ptdpes_to_eth/checkpoints/best.pt \
      --checkpoint experiments/agentformer_dlow_ptdpes/checkpoints/best.pt \
      --run-dir experiments/agentformer_dlow_ptdpes_to_eth --device cuda
    python -m baselines.evaluate_agentformer --model dlow --fold eth \
      --predictor-checkpoint experiments/agentformer_vae_ptdpes_to_eth/checkpoints/best.pt \
      --checkpoint experiments/agentformer_dlow_ptdpes_to_eth/checkpoints/best.pt \
      --run-dir experiments/agentformer_dlow_ptdpes_to_eth_eval --device cuda

## Evaluation

All evaluation commands use the held-out location for the selected five-fold
split. Trajectron++, SocialVAE, MoFlow, and AgentFormer report best-of-20 ADE/FDE.
Validation data comes only from the
non-held-out training locations.

Each evaluator accepts `--visual-scenes N` (default 10, maximum 30) and writes
individual PNG/PDF trajectory comparisons, a contact sheet, and
`visuals/metrics.json`. Legends include coherent model and constant-velocity
ADE/FDE values for the exact trajectories drawn.

CUDA is the default for every trainer and evaluator. There is no automatic CPU
fallback; `--device cpu` must be explicit and is intended only for diagnostics.

## Reference integrity

Reference checkouts remain unmodified. Compatibility shims for modern NumPy,
PyTorch, and Python live only in these adapters. `external_runner.py` remains
available for reproducing upstream commands against their native released data.

## Named pretrained models

`--checkpoint`, and AgentFormer's `--predictor-checkpoint`, accept either a file
path or a registry ID such as `trajectron_ptdpes_0`. Register manually produced models
with `python -m baselines.registry add MODEL_ID CHECKPOINT --model MODEL
--dataset ptdpes --seed SEED`. Pretraining registers seeded
outputs automatically and reuses each ID across all requested downstream folds.

## Configurable pretraining and transfer

SocialVAE and MoFlow resolve optimization settings in this order: model-native stage
recipe, `--transfer-config`, then explicit CLI options. See `configs/transfer.yaml`.
Pretraining accepts one source with `--pretrain-dataset ptdpes`, or a union such as
`--pretrain-datasets ptdpes steps`. Target fractions are selected at the
namespaced-trajectory level and persisted under `experiments/manifests/target_fractions/`; the same
data seed gives nested 5/10/25/50/100% populations.

Run one complete target train/evaluate job with:

    ./scripts/run_transfer_experiment.sh --model moflow --stage finetune \
      --checkpoint MODEL_ID --target eth_ucy --fold eth --fraction 0.25

`--seed` is optional for `--stage finetune` (and `--stage direct`, below): when omitted, it
resolves from the pretrained checkpoint's registered seed (or its `config.json` seed, for
an explicit checkpoint path), is printed, and is recorded in `params.env`; pass `--seed`
explicitly to override it.

For the matched scratch control, omit the checkpoint and pass `--seed` explicitly:

    ./scripts/run_transfer_experiment.sh --model moflow --stage scratch \
      --target eth_ucy --fold eth --fraction 0.25 --seed 1

To evaluate a pretrained or fine-tuned checkpoint directly on a target holdout with no
further training, use `--stage direct` (requires `--checkpoint`; `--fraction` must be
unset or `1.0`, since there is no target training split to subsample):

    ./scripts/run_transfer_experiment.sh --model moflow --stage direct \
      --checkpoint MODEL_ID --target eth_ucy --fold eth

Replace the target with `madras` or `grand_central` (without `--fold`). Every run stores
the exact target manifest, source checkpoint and validation metric, resolved transfer
recipe/hash, best target-validation epoch, test ADE/FDE, curves, and qualitative plots.
Fine-tuning accepts only a protocol-aware `checkpoints/best.pt` by default; the explicit
`--allow-legacy-checkpoint` trainer flag is for compatibility diagnostics, not paper runs.
`run_transfer_experiment.sh` refuses to overwrite an existing run's training or evaluation
output unless `--resume`/`--resuming` is passed. `--num-workers` forwards to the trainer for
`finetune`/`scratch`, and to `evaluate_social_vae` whenever it runs (`evaluate_moflow` has no
such flag); `--validation-mode` (default `grouped`) forwards to the trainer only.

## STARS (work in progress, K=1)

`train_stars.py` / `evaluate_stars.py` run the pinned STARS homogeneous GNN-VAE
(`references/stars` at `13d840c`, `HGNNCombinedVAE`, unmodified) as a forecaster on the same
`steps_default` samples as MoFlow (identical preprocessing hash). `stars_data.py` builds one
graph per benchmark sample: the target pedestrian in STARS's robot slot (node 0), neighbours
seen at least twice in the 8 observed steps as human nodes, STARS's 8-channel SE(2) edge
features over the observed steps only, resampled to 40 steps; edges whose endpoints share no
valid step are dropped (STARS's edge transformer returns NaN for fully masked edges in eval).

Two phases per run: the VAE is trained self-supervised (STARS paper `homo` recipe; effective
beta warm-up 50) on the training split, then frozen; a linear head on
`[z_target ‖ mean z_neighbours ‖ target history]` predicts all 12 steps (STARS forecasting-probe
recipe), selected on validation ADE. `pretrain` = source datasets; `finetune` = head refit on
the target with the pretrained encoder frozen; `scratch` = both phases on the target.
Deterministic (K=1); `--latent-samples K` reports best-of-K from posterior samples.
`--max-edges` splits batches by edge count with whole-batch loss denominators, so memory is
bounded without changing the objective.

    python -m baselines.train_stars --stage pretrain --pretrain-dataset steps \
      --run-dir experiments/pretrained/stars_steps_0 --device cuda --seed 0
    python -m baselines.train_stars --stage finetune --target-dataset eth_ucy --fold eth \
      --checkpoint experiments/pretrained/stars_steps_0 --run-dir experiments/stars_steps_to_eth --device cuda
    python -m baselines.evaluate_stars --target-dataset eth_ucy --fold eth \
      --checkpoint experiments/stars_steps_to_eth --run-dir experiments/stars_steps_to_eth_eval --device cuda

Status (2026-10-21): CPU smoke of pretrain -> finetune -> evaluate passes on tiny subsets;
`tests/test_stars_integration.py` covers edge-feature parity with STARS's helpers, no future
leakage, world mapping, and chunked-loss/gradient equality with STARS's full-batch loss. Open:
no GPU run yet (GPU timing and a `--max-edges` budget for the target GPU still to measure; STEPS
graphs are median 32 / max 69 nodes vs STARS's <=18, so a 1000-epoch VAE run may be long);
no `--resume`; not wired into `scripts/pretrain_transfer_model.sh` / `run_transfer_experiment.sh`
or the aggregate tables; no qualitative renders in evaluation. The heterogeneous STARS model
was not used: it hard-codes SEAN/SNSB inputs.

## Multi-agent STARS (work in progress)

`train_stars_ma.py` / `evaluate_stars_ma.py` extend STARS into a multi-agent pedestrian dynamics
model on STEPS: node motion features alongside STARS's edge features at the native rate, kNN/radius
graphs, a per-agent deterministic or CVAE decoder that forecasts every pedestrian, and closed-loop
rollouts for evaluation and simulation. The single-target STARS probe above is
unchanged. Design, commands, ablation flags and open items: [docs/stars_multiagent.md](../../docs/stars_multiagent.md).
No GPU run yet.
