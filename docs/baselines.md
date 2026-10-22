# Forecasting baselines

Commands run from the repository root with `.venv` active. See also [`src/baselines/README.md`](../src/baselines/README.md).

The integrated baselines are Trajectron++ (K=20), SocialVAE (K=20), MoFlow (K=20), and
AgentFormer VAE+DLow (K=20). CUDA is the default and CPU is
used only when explicitly requested.

SocialVAE and MoFlow use explicit, stage-aware transfer recipes. Pretraining and
fine-tuning never share optimizer or scheduler state, all parameters are fine-tuned,
and scratch/pretrained target runs use identical fractions and target optimization.
The completed September 9 SocialVAE/MoFlow PTDPES runs predate this protocol and are
legacy diagnostics; rerun them before using transfer results in the paper.

## Copy-paste pretraining commands

Run these from the repository root. Change only the final seed number.
ATC currently has converted trajectories but still needs its seed-42 scene index;
run `python -m preprocess datasets/atc --data-seed 42` once before its commands.

Trajectron++:

```bash
./scripts/pretrain_baseline.sh --model trajectron --dataset atc --seed 0 --batch-size 512
./scripts/pretrain_baseline.sh --model trajectron --dataset ptdpes --seed 0 --batch-size 512
./scripts/pretrain_baseline.sh --model trajectron --dataset steps --seed 0 --batch-size 512
```

SocialVAE:

```bash
./scripts/pretrain_transfer_model.sh --model socialvae --sources atc --seed 0
./scripts/pretrain_transfer_model.sh --model socialvae --sources ptdpes --seed 0
./scripts/pretrain_transfer_model.sh --model socialvae --sources steps --seed 0
```

MoFlow:

```bash
./scripts/pretrain_transfer_model.sh --model moflow --sources atc --seed 0
./scripts/pretrain_transfer_model.sh --model moflow --sources ptdpes --seed 0
./scripts/pretrain_transfer_model.sh --model moflow --sources steps --seed 0
```

AgentFormer:

```bash
./scripts/pretrain_baseline.sh --model agentformer --dataset atc --seed 0 --batch-size 1
./scripts/pretrain_baseline.sh --model agentformer --dataset ptdpes --seed 0 --batch-size 1
./scripts/pretrain_baseline.sh --model agentformer --dataset steps --seed 0 --batch-size 1
```

Each command pretrains on the selected dataset using CUDA, saves all logs/manifests/curves and
parameters under `experiments/pretrained/`, and registers a reusable ID such as
`trajectron_ptdpes_0` or `moflow_transfer_v1_atc_0`. AgentFormer automatically runs and registers both its VAE and
DLow stages. Optional overrides include `--epochs`, `--device`, `--data-seed`,
`--batch-size`, `--num-workers`, `--iterations`, `--patience`, `--min-delta`,
`--output`, and `--python`. SocialVAE pretraining defaults to 600 epochs × 100
batches, Adam at `5e-4`, and no early stopping. MoFlow pretraining defaults to
150 epochs, AdamW at `1e-4`, batch 48, 5% warmup plus cosine decay, and no
early stopping.

`--iterations` provides a fixed optimizer-step budget independent of dataset size. Increase
the batch size until GPU memory is well utilized; 128 is a practical starting point, while
256 is worth trying if it fits. Keep the chosen batch size and iteration budget identical
across comparable experiment arms. The validation cap applies only during checkpoint
selection; held-out evaluation still uses the complete test split. AgentFormer retains
the upstream one-scene batching.

Measured on the default 11 GB RTX 2080 Ti, the largest useful batches were:

| Model | Recommended batch | Observed result |
|---|---:|---|
| Trajectron++ | 512 | 1024 fit but provided essentially no throughput improvement over 512 |
| SocialVAE | 128 | Official batch size; increase only after checking GPU memory and validation parity |
| MoFlow | 48 for the literature protocol | Larger batches are explicitly labeled throughput ablations |
| AgentFormer VAE/DLow | 1 scene | Required by the upstream scene-oriented data/model contract |

Trajectron++ caches fully prepared interaction graphs as bounded temporal shards
under each dataset's `.cache/trajectron/` directory. The first run can spend substantial
time building them; later model seeds and reruns load them directly. Writes are atomic,
locked, resumable, and use streaming Zstandard compression. Training loads one shard at
a time and deterministically caps its examples to the accepted unified-index count. Each
shard contains at most two overlapping 100-timestep scenes, so a dense or long recording
cannot create an unbounded in-memory environment. This also prevents native expansion to
tens of millions of examples.

Build caches for every currently available dataset before training with:

```bash
python -m scripts.cache_trajectron_datasets --datasets all --data-seed 42
```

Use `--datasets PTDPES,atc` for selected datasets. Add `steps` after its converted files and
scene index exist. Completed temporal shards are skipped automatically.

For the current detached PTDPES transfer-pretraining runs, one launcher starts
MoFlow and SocialVAE for seeds 0, 1, and 2. It pins the two models for a seed to
the same physical GPU and persists one terminal log per model/seed:

```bash
./scripts/pretrain_ptdpes_transfer_tmux.sh ptdpes-transfer-v2
./scripts/add_trajectron_ptdpes_tmux.sh ptdpes-transfer-v2
tmux attach-session -t ptdpes-transfer-v2
tail -f logs/experiments/moflow-transfer_v2-ptdpes-seed0.log
tail -f logs/experiments/socialvae-transfer_v2-ptdpes-seed0.log
```

Attach to a still-running session with `tmux attach -t SESSION_NAME`.

MoFlow and SocialVAE checkpoints contain model, optimizer, scheduler,
early-stopping, RNG, epoch, within-epoch batch, and DataLoader generator state.
Restart an interrupted run with the identical command plus `--resume` (or its
alias `--resuming`); without a path it uses `RUN_DIR/checkpoints/latest.pt`:

```bash
./scripts/pretrain_transfer_model.sh --model moflow --sources ptdpes --seed 0 --run-version transfer_v2 --resume
./scripts/run_transfer_experiment.sh --model moflow --stage finetune --checkpoint moflow_transfer_v2_ptdpes_0 --target eth_ucy --fold eth --seed 0 --resume
```

Resume validates the original model, stage, seeds, data sources, preprocessing,
and optimization recipe. It refuses a changed contract rather than silently
continuing a different experiment. Transfer initialization remains separate:
fine-tuning loads pretrained model weights but starts a fresh optimizer and
scheduler, while `--resume` restores an interrupted stage's optimizer and
scheduler.

The Trajectron++ launcher adds seeds 0--2 to the same tmux session at batch 128.
Its worker reuses the bounded PTDPES shard cache, automatically resumes an
unfinished `transfer_v2` checkpoint, and skips a completed run. A `gpu-monitor`
window records one-minute VRAM/utilization samples in
`logs/experiments/ptdpes-transfer-v2-gpu.log`.

Pretraining and downstream training are disjoint. Completed pretrained models
are registered under stable IDs:

```text
trajectron_ptdpes_0
socialvae_transfer_v1_ptdpes_0
moflow_transfer_v1_ptdpes_0
agentformer_vae_ptdpes_0
agentformer_dlow_ptdpes_0
```

The same pretrained ID can be supplied to any number of downstream runs:

```bash
python -m baselines.train_trajectron \
  --stage finetune --fold eth --checkpoint trajectron_ptdpes_0 \
  --data-seed 42 --device cuda \
  --run-dir experiments/downstream/trajectron_ptdpes_0/eth

python -m baselines.train_trajectron \
  --stage finetune --fold zara1 --checkpoint trajectron_ptdpes_0 \
  --data-seed 42 --device cuda \
  --run-dir experiments/downstream/trajectron_ptdpes_0/zara1
```

SocialVAE/MoFlow ETH/UCY target training reserves one complete non-test scene
sequence for validation by default. `best.pt` is selected by target validation
ADE/minADE with early stopping; the held-out test fold is never used for
checkpoint selection.

Set `TRAJECTORY_MODEL_REGISTRY` when using a non-default registry directory.
Inspect or resolve registered models with:

```bash
python -m baselines.registry list
python -m baselines.registry resolve trajectron_ptdpes_0
```

Checkpoint arguments accept a registered ID, a checkpoint file, or a run
directory containing `checkpoints/best.pt`.

Every run contains:

- `config.json` and atomic `status.json`
- append-only `training.log` and `metrics.jsonl`
- `datasets/train.parquet`, `val.parquet`, or `test.parquet`; streamed
  Trajectron++ pretraining additionally records the complete
  `*_trajectron_shards.parquet` population and realized per-shard budgets
- copied `configs/task.json`, the five-fold policy, and hashes of source scene indices
- `checkpoints/latest.pt` and `checkpoints/best.pt`
- `curves/training_curves.png` and `training_curves.pdf`
- `evaluation.json` for evaluation runs
- `visuals/scene_XX.png` and `.pdf`, `contact_sheet.png`, and per-scene metrics

Dataset preparation stages are printed immediately, and each training epoch has a live
batch progress bar with iteration and loss. The same lifecycle events and periodic metrics
are persisted in `training.log`, `status.json`, and `metrics.jsonl`, so progress remains
inspectable when a job is detached or its terminal output is unavailable.

Evaluation renders 10 deterministic held-out scenes by default. Set
`--visual-scenes N` on an evaluator; values are capped at
30. Each plot shows observed and context trajectories, future ground truth,
constant velocity, and the model forecast. The legend reports ADE/FDE for both
CV and the displayed model trajectory. For K=20 models, the displayed trajectory
is the sample with the best ADE, and its matching FDE is reported.

## Individual commands

For SocialVAE or MoFlow, one command trains one target condition and immediately
evaluates its selected checkpoint. The same command with `--stage scratch`
omits `--checkpoint` and uses the identical target fraction, split, maximum
epochs, checkpoint metric, and early-stopping rule:

```bash
./scripts/run_transfer_experiment.sh --model moflow --stage finetune --checkpoint moflow_transfer_v1_ptdpes_1 --target eth_ucy --fold eth --fraction 0.25 --seed 1
./scripts/run_transfer_experiment.sh --model moflow --stage scratch --target eth_ucy --fold eth --fraction 0.25 --seed 1
./scripts/run_transfer_experiment.sh --model socialvae --stage finetune --checkpoint socialvae_transfer_v1_ptdpes_1 --target madras --fraction 1.0 --seed 1
```

Target fractions select namespaced source trajectories, retain all selected
training windows, are nested for a fixed data seed, and never alter validation
or test. Their persistent manifests live in `experiments/manifests/target_fractions/` and a
copy is saved with every run. Use `--transfer-config configs/transfer.yaml` or
individual overrides such as `--learning-rate`, `--epochs`, and `--batch-size`.
Python pretraining entry points also accept multiple sources, for example
`--pretrain-datasets ptdpes steps`; the wrapper equivalent is
`--sources ptdpes,steps`.
For an explicitly declared fine-tuning LR sweep, use `--learning-rate` with
SocialVAE values `5e-5`, `1e-4`, or `2e-4`, or MoFlow values `1e-5`, `2e-5`,
`5e-5`, or `1e-4`; the defaults remain `1e-4` and `2e-5`, respectively.

Aggregate matched seed results with standard error across independent model
seeds (not across overlapping windows):

```bash
python -m baselines.aggregate_transfer \
  experiments/downstream/finetune/moflow/moflow_transfer_v1_ptdpes_0/seed0/madras/fraction-1.0/test \
  experiments/downstream/finetune/moflow/moflow_transfer_v1_ptdpes_1/seed1/madras/fraction-1.0/test \
  experiments/downstream/finetune/moflow/moflow_transfer_v1_ptdpes_2/seed2/madras/fraction-1.0/test \
  --output experiments/downstream/finetune/moflow/ptdpes-to-madras-fraction-1.0.json
```

Each model uses the same stage convention:

```bash
# PTDPES pretraining
python -m baselines.train_moflow \
  --stage pretrain --seed 1 --data-seed 42 --device cuda \
  --run-dir experiments/moflow/seed_1/pretrain

# ETH/UCY fold fine-tuning from that seed's pretrained model
python -m baselines.train_moflow \
  --stage finetune --fold eth --seed 1 --data-seed 42 --device cuda \
  --checkpoint experiments/moflow/seed_1/pretrain/checkpoints/best.pt \
  --run-dir experiments/moflow/seed_1/eth/finetune

# Transfer training reserves one complete sequence from the four-fold fitting pool
# for validation; the held-out fold remains test-only.

# Held-out ETH evaluation
python -m baselines.evaluate_moflow \
  --fold eth --data-seed 42 --device cuda \
  --visual-scenes 20 \
  --checkpoint experiments/moflow/seed_1/eth/finetune/checkpoints/best.pt \
  --run-dir experiments/moflow/seed_1/eth/finetune_eval
```

Replace `moflow` with `trajectron` or `social_vae`. AgentFormer has separate VAE and DLow stages; exact
commands and checkpoint relationships are documented in
[`src/baselines/README.md`](../src/baselines/README.md).

Aggregate any independently launched evaluation runs with:

```bash
python -m baselines.aggregate \
  --output experiments/moflow/eth_finetune_summary.json \
  experiments/moflow/seed_1/eth/finetune_eval \
  experiments/moflow/seed_2/eth/finetune_eval \
  experiments/moflow/seed_3/eth/finetune_eval
```
