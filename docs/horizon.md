# Running on TACC Horizon

Setup and training on TACC Horizon (Rocky Linux, aarch64, 4x NVIDIA GB200 per node, Slurm). The
general setup in the [project README](../README.md) still applies; this page covers what differs
on Horizon. Model commands are in [`stars_multiagent.md`](stars_multiagent.md).

Things that are different on Horizon:

- **Compute nodes have no internet.** Clone, `uv sync` and any download run on a login node.
- **Keep everything under `$SCRATCH`.** `$HOME` has a small quota; the code, `datasets/`,
  `experiments/` and the uv cache all go on `$SCRATCH`.
- **Nodes are allocated whole** (4 GPUs, exclusive), so put up to four runs in one job
  ([`scripts/horizon/gpu_jobs.slurm`](../scripts/horizon/gpu_jobs.slurm)).
- **The default compiler (`nvc`) cannot build some C extensions** (`ncls`); use `gcc`.
- **PyPI's aarch64 torch is CPU-only.** `pyproject.toml` already pulls torch / torchvision from
  the PyTorch CUDA 13.0 index on aarch64; nothing to do, but don't remove that block.

## 1. One-time setup (login node)

```bash
# uv, installed into ~/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Keep uv's cache (several GB of CUDA wheels) off $HOME. Add both lines to your shell rc.
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export CC=gcc CXX=g++

cd "$SCRATCH"
git clone --recurse-submodules https://github.com/michaelmunje/stars_steps_extension.git
cd stars_steps_extension
uv sync --locked
```

`uv sync` downloads Python 3.12 and the locked dependencies into `.venv/`. If it was interrupted,
rerun it; don't pipe its output through `head`, which kills it partway.

Check the environment (CPU, login node). The full suite takes a few minutes and should report
`561 passed, 10 skipped`; `tests/test_stars_ma.py` alone is a quick check:

```bash
uv run python -m pytest tests/test_stars_ma.py -q
uv run python -m pytest -q
```

## 2. Datasets

`datasets/` is gitignored. A read-only copy of every dataset is shared on Horizon at
`/scratch/09865/mmunje/shared/datasets` (23 GB: `PTDPES` 20 GB, `atc_subset` 2 GB, `steps` 462 MB,
the rest small). The multi-agent STARS runs only need `steps`. From the clone's root:

```bash
# Only what multi-agent STARS needs (462 MB)
mkdir -p datasets && rsync -a /scratch/09865/mmunje/shared/datasets/steps datasets/

# Or everything (23 GB)
rsync -a /scratch/09865/mmunje/shared/datasets/ datasets/
```

Use `rsync -a` (or `cp -a`), not plain `cp`: the split manifests record file modification times,
and a copy that resets them makes the loader regenerate and warn about "stale" split manifests.
The shared directory cannot be listed from above, so use the exact path. The files are used as
they are; no preprocessing is needed for `steps`.

## 3. Check a GPU node

Before long runs, run one short job and confirm CUDA works and the pipeline completes (about
a minute of work):

```bash
cd "$SCRATCH/stars_steps_extension"
sbatch --time=00:30:00 scripts/horizon/gpu_jobs.slurm \
  "python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())'
   python -m baselines.train_stars_ma --phase ssl --dataset steps --run-dir experiments/stars_ma/smoke \
     --max-scenes 64 --max-batches 4 --epochs 1"
squeue -u "$USER"
```

`sbatch` prints the job id. The Slurm log is `stars_ma-<job id>.out` in the repository root; each
command's own output is `logs/horizon/<job id>/gpu<N>.log`. The first line should print
`... True NVIDIA GB200`.

## 4. Training

Each argument to `gpu_jobs.slurm` is one command and gets its own GPU (at most four per job). The
script defaults to partition `debug`, account `CDA26011` and 24 hours; override any of them on
the `sbatch` command line (`--account=<yours>`, `--time=2-00:00:00`; 48 h is the queue limit).
Submit from the repository root.

**Phase 1** -- self-supervised encoder, node+edge and the edge-only ablation:

```bash
sbatch scripts/horizon/gpu_jobs.slurm \
  "python -m baselines.train_stars_ma --phase ssl --dataset steps --run-dir experiments/stars_ma/ssl" \
  "python -m baselines.train_stars_ma --phase ssl --dataset steps --edge-only --run-dir experiments/stars_ma/ssl_edge_only"
```

**Phase 2 and 3** -- after phase 1 has written `experiments/stars_ma/ssl/checkpoints/best.pt`:

```bash
SSL=experiments/stars_ma/ssl/checkpoints/best.pt
F="python -m baselines.train_stars_ma --phase forecast --dataset steps"
sbatch scripts/horizon/gpu_jobs.slurm \
  "$F --encoder-checkpoint $SSL --encoder-mode frozen   --run-dir experiments/stars_ma/det_frozen" \
  "$F --encoder-checkpoint $SSL --encoder-mode finetune --run-dir experiments/stars_ma/det_finetune" \
  "$F --encoder-mode scratch --run-dir experiments/stars_ma/det_scratch" \
  "$F --decoder cvae --encoder-checkpoint $SSL --encoder-mode finetune --run-dir experiments/stars_ma/cvae_finetune"
```

`$SSL` and `$F` are expanded by your shell when you submit, so the job receives the full commands.
To chain a job after phase 1 instead of waiting, add `--dependency=afterok:<phase 1 job id>`.

The ablations take the flags in
[`stars_multiagent.md`](stars_multiagent.md#commands) in the same way.

**Evaluation** (also on a GPU node; writes a JSON report):

```bash
sbatch --time=04:00:00 scripts/horizon/gpu_jobs.slurm \
  "python -m baselines.evaluate_stars_ma --checkpoint experiments/stars_ma/cvae_finetune/checkpoints/best.pt \
     --datasets steps --split test --output experiments/stars_ma/cvae_finetune/eval_test.json"
```

## 5. Monitoring

```bash
squeue -u "$USER"                                    # state, node, elapsed time
tail -f logs/horizon/<job id>/gpu0.log               # one run's output
ssh <node>                                           # while the job runs, e.g. to run nvidia-smi
scancel <job id>
```

Each run writes its checkpoints and metrics under its `--run-dir`
(`experiments/stars_ma/<name>/`); `checkpoints/best.pt` is the validation-selected model.

## Status

The Horizon environment (`uv sync`, the test suite, CPU runs of every phase) was checked on a
login node on 2026-10-21. **No GPU job has been run yet**: CUDA on the GB200 nodes, epoch times
and the default batch sizes are unverified, so run step 3 first. Phase 1 has no `--resume`; a run
that hits the time limit restarts from scratch.
