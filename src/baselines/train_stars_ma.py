"""Train multi-agent STARS (``stars_ma_model``): self-supervised encoder, then forecaster.

Phase 1, ``--phase ssl``: the extended STARS encoder (node motion + SE(2) edge features at the
native rate, neighbourhood graphs) is trained with STARS's self-supervised graph-VAE objective
plus node-motion reconstruction, selected on validation loss.

Phases 2-4, ``--phase forecast``: a multi-agent decoder forecasts every pedestrian of each scene
window. ``--encoder-mode`` chooses a frozen or jointly fine-tuned phase-1 encoder
(``--encoder-checkpoint``) or one trained from scratch; ``--decoder cvae`` adds the probabilistic
decoder (phase 3); ``--rollout-exec-steps`` adds a closed-loop multi-step loss (phase 4). Forecast
checkpoints are selected on validation best-of-K all-agent ADE.

Closed-loop fine-tuning, ``--init-checkpoint`` + ``--rft-steps K``: starting from a trained forecaster,
every batch is also unrolled in a simulator-faithful closed loop for K steps of 0.4 s
(``stars_ma_model.closed_loop_rollout``: re-encode from the model's own executed positions, execute
the first step, prior latents with AR(1) noise as in ``simplenavsim.realism.sim.Stars``), and the
executed positions are supervised against the ground truth with gradients through the rollout. The
open-loop CVAE loss stays as a regulariser (``--open-loop-weight``). ``--rft-eval-steps`` adds the
same closed loop (no gradients, one sample) to validation, and ``--selection closedloop`` selects on
its per-dataset mean ADE.

Datasets come from a YAML config (``--config``, see ``configs/stars_ma/``) or ``--datasets``.
Every option below can also be set in the config (same names, underscores); explicit flags win.

Scale: every split is extracted once into a cached ``WindowStore`` (``stars_ma_batch``) that
lives on the GPU; graphs are built batch-wise on the device; batches hold up to ``--max-nodes``
pedestrians of similar-sized windows; ``torchrun --nproc-per-node 4`` trains data-parallel
(DDP) across GPUs; ``--amp`` (default) runs the networks in bf16. ``--benchmark N`` times N
training steps and exits.
"""
from __future__ import annotations

import argparse
import math
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from baselines.device import training_device
from baselines.early_stopping import EarlyStopping
from baselines.stars import REFERENCE, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY, VAE_RECIPE
from baselines.stars_ma_batch import (WindowStore, build_batch, cached_store, concat_stores, graph_options,
                                      node_budget_batches, per_target_store, window_weights)
from baselines.stars_ma_graph import to_local, to_scene
from baselines.stars_ma_metrics import MetricAccumulator, batch_metrics
from baselines.stars_ma_model import (
    DECODER_RECIPE,
    ENCODER_RECIPE,
    StarsMotionVAE,
    StarsMultiAgentForecaster,
    closed_loop_rollout,
    rollout,
)
from baselines.status import RunStatus, checkpoint_metadata
from config import Config, DatasetSpec, SeedConfig
from data import CANONICAL_DT, SceneDataset
from forecasting_config import config_as_dict, resolve_forecasting_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRESET = "steps_default"
# Name -> location and split policy of every dataset this trainer can read.
DATASETS = {
    "steps": {"path": "datasets/steps"},
    "eth": {"path": "datasets/eth"},
    "ucy": {"path": "datasets/ucy"},
    "madras": {"path": "datasets/madras"},
    "grand_central": {"path": "datasets/grand_central", "split_strategy": "temporal"},
    "ptdpes": {"path": "datasets/PTDPES"},
    "atc": {"path": "datasets/atc_subset", "split_strategy": "held_out_recording"},
}
# STARS's phase-1 optimisation recipe (stars.VAE_RECIPE), at STARS's batch of 32 windows.
SSL_RECIPE = {key: VAE_RECIPE[key] for key in
              ("learning_rate", "weight_decay", "grad_clip", "max_epochs", "beta_max", "beta_warmup_epochs",
               "patience", "min_delta")}
FORECAST_RECIPE = {
    "learning_rate": 1e-3,
    "encoder_learning_rate": 1e-4,  # fine-tuning only, scaled with learning_rate
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "max_epochs": 200,
    "patience": 20,
    "min_delta": 1e-4,
    "kl_weight": 0.1,  # 1.0 collapses the latent on STEPS
    "kl_warmup_epochs": 10,
}
# The recipes' learning rates were set for 32 windows of ~19 STEPS pedestrians per step.
REFERENCE_NODES = 600


# ----------------------------------------------------------------------------- configuration


def dataset_entries(value) -> list[dict]:
    """Normalise a dataset list (names or ``{name, weight, path, split_strategy}``) against ``DATASETS``."""
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    entries = []
    for item in value or []:
        item = {"name": item} if isinstance(item, str) else dict(item)
        name = str(item["name"]).lower()
        if name == "all":
            entries += [{"name": key, **meta, "weight": 1.0} for key, meta in DATASETS.items()
                        if (ROOT / meta["path"]).exists() and key != "atc"]
            continue
        if name not in DATASETS and "path" not in item:
            raise ValueError(f"Unknown dataset {name!r}; choose {sorted(DATASETS)} or give a path")
        entries.append({**DATASETS.get(name, {}), **item, "name": name, "weight": float(item.get("weight", 1.0))})
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate datasets in {names}")
    return entries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML with `datasets` (and optionally any option below)")
    parser.add_argument("--phase", choices=("ssl", "forecast"))
    parser.add_argument("--datasets", help="Comma-separated training datasets, or 'all' (overrides the config)")
    parser.add_argument("--val-datasets", help="Validation datasets (default: the training datasets)")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue --run-dir from checkpoints/latest.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/.cache/stars_ma"))
    parser.add_argument("--cache-workers", type=int, default=min(32, os.cpu_count() or 1),
                        help="Processes used to extract windows the first time a split is cached")
    parser.add_argument("--max-scenes", type=int, help="Integration diagnostic: cap every split's window count")
    parser.add_argument("--max-batches", type=int, help="Integration diagnostic: cap batches per epoch and evaluation")
    parser.add_argument("--benchmark", type=int, help="Time this many training steps, report throughput, and exit")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float, help="Encoder dropout (scratch encoders / phase 1)")
    parser.add_argument("--max-nodes", type=int, default=16384, help="Pedestrians per batch on each GPU")
    parser.add_argument("--max-graphs", type=int, help="Cap windows per batch")
    parser.add_argument("--learning-rate", type=float, help="Peak LR (default: the recipe LR scaled by --lr-scaling)")
    parser.add_argument("--lr-scaling", choices=("sqrt", "linear", "none"), default="sqrt",
                        help=f"Scale the recipe LR with the global batch size relative to {REFERENCE_NODES} nodes")
    parser.add_argument("--max-learning-rate", type=float, default=3e-3, help="Cap for the scaled LR")
    parser.add_argument("--schedule", choices=("cosine", "plateau", "constant"), default="cosine")
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast on CUDA")
    parser.add_argument("--compile", action="store_true", help="torch.compile the temporal transformers")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Recompute temporal-transformer activations in backward: less memory, more compute")
    parser.add_argument("--eval-every", type=int, default=1, help="Validate every this many epochs")
    parser.add_argument("--selection", choices=("macro", "pooled", "closedloop"), default="macro",
                        help="Forecast selection: mean of per-dataset agent ADE (macro) or over all agents, or "
                             "(with --rft-eval-steps) mean of per-dataset closed-loop ADE")
    graph = parser.add_argument_group("graph")
    graph.add_argument("--knn", type=int, default=16, help="Sources kept per destination (default 16)")
    graph.add_argument("--radius", type=float, default=10.0, help="Neighbourhood radius in metres (default 10)")
    graph.add_argument("--fully-connected", action="store_true", help="STARS's topology: ignore --knn/--radius")
    graph.add_argument("--stars-grid", type=int, help="STARS fidelity: resample node/edge sequences to this many steps (STARS: 40)")
    graph.add_argument("--stars-heading", action="store_true", help="STARS fidelity: heading 0 until an agent first moves (no back-fill)")
    graph.add_argument("--target-slot", action="store_true",
                       help="STARS fidelity (ssl): one graph per (window, target), target in the robot slot (node type 0); "
                            "one seeded target per window")
    graph.add_argument("--max-graph-nodes", type=int, help="With --target-slot: keep the target and its nearest nodes (STARS: 18)")
    encoder = parser.add_argument_group("encoder")
    encoder.add_argument("--hidden-dim", type=int)
    encoder.add_argument("--latent-dim", type=int)
    encoder.add_argument("--edge-only", action="store_true", help="Ablation: STARS's edge-only encoder (no node motion)")
    encoder.add_argument("--beta-warmup-epochs", type=int, help="ssl: KL beta warm-up epochs (default STARS's 50)")
    encoder.add_argument("--ssl-val-beta", type=float,
                         help="ssl: validate (and select) at this fixed beta instead of the current warm-up beta")
    encoder.add_argument("--node-loss-weight", type=float, default=1.0, help="ssl: weight of node-motion reconstruction")
    forecast = parser.add_argument_group("forecast")
    forecast.add_argument("--encoder-checkpoint", type=Path, help="A --phase ssl checkpoint (frozen/finetune)")
    forecast.add_argument("--encoder-mode", choices=("frozen", "finetune", "scratch"), default="frozen")
    forecast.add_argument("--decoder-hidden-dim", type=int)
    forecast.add_argument("--decoder-dropout", type=float)
    forecast.add_argument("--decoder", choices=("deterministic", "cvae", "diffusion", "joint_diffusion"),
                          default="deterministic")
    forecast.add_argument("--diffusion-social-layers", type=int, default=2,
                          help="joint_diffusion: social layers inside the denoiser")
    forecast.add_argument("--diffusion-steps", type=int, default=DECODER_RECIPE["diffusion_steps"],
                          help="diffusion: DDPM training steps")
    forecast.add_argument("--sample-steps", type=int, default=DECODER_RECIPE["sample_steps"],
                          help="diffusion: DDIM steps when sampling")
    forecast.add_argument("--cvae-latent-dim", type=int, default=32)
    forecast.add_argument("--goal", action="store_true",
                          help="Goal-conditioned decoder: each agent's oracle goal (its track's last observed "
                               "position in the recording) joins the decoder context")
    forecast.add_argument("--goal-beyond-horizon", action="store_true",
                          help="With --goal: drop goals that are reached within the forecast horizon (leak removal)")
    forecast.add_argument("--no-social-layer", action="store_true", help="Decoder without its social message-passing round")
    forecast.add_argument("--representation", choices=("mu", "hidden"), default=DECODER_RECIPE["representation"])
    forecast.add_argument("--kl-weight", type=float, default=FORECAST_RECIPE["kl_weight"])
    forecast.add_argument("--rollout-exec-steps", type=int, default=0,
                          help="> 0 adds the closed-loop multi-step loss, executing this many steps per round")
    forecast.add_argument("--rollout-weight", type=float, default=1.0)
    forecast.add_argument("--samples", type=int, default=20, help="K for best-of-K validation (CVAE)")
    forecast.add_argument("--refine", type=int, default=0,
                          help="Future-aware social refinement rounds in the decoder (agents attend to the "
                               "neighbours' predicted futures and re-decode)")
    forecast.add_argument("--collision-weight", type=float, default=0.0,
                          help="> 0 adds a pairwise collision hinge on the decoded futures (CVAE: on a prior sample)")
    forecast.add_argument("--bom-weight", type=float, default=0.0,
                          help="> 0 (CVAE) adds a best-of-many loss: the min over --bom-samples prior samples of "
                               "each agent's ADE (prior samples cover the data instead of regressing to its mean)")
    forecast.add_argument("--bom-samples", type=int, default=4)
    forecast.add_argument("--collision-fixed-prior", action=argparse.BooleanOptionalAction, default=True,
                          help="Collision hinge on a prior sample with a gradient-free latent: it trains the "
                               "decoder only (--no-collision-fixed-prior also lets it widen the prior, which "
                               "inflated the sample spread)")
    rft = parser.add_argument_group("closed-loop fine-tuning (off by default)")
    rft.add_argument("--init-checkpoint", type=Path,
                     help="Start from this forecast checkpoint (its encoder/decoder recipes and weights)")
    rft.add_argument("--rft-steps", type=int, default=0,
                     help="> 0: closed-loop rollout loss over this many executed 0.4 s steps (<= prediction steps)")
    rft.add_argument("--rft-start-steps", type=int, help="Curriculum: rollout length at epoch 1 (default --rft-steps)")
    rft.add_argument("--rft-ramp-epochs", type=int, default=0, help="Curriculum: epochs to reach --rft-steps")
    rft.add_argument("--rft-weight", type=float, default=1.0)
    rft.add_argument("--open-loop-weight", type=float, default=1.0,
                     help="Weight of the open-loop (teacher-forced) forecast + KL loss kept as a regulariser")
    rft.add_argument("--rft-speed-weight", type=float, default=0.0,
                     help="> 0: add this x the progress error (m; distance from the start, simulated vs true) to the rollout loss")
    rft.add_argument("--rft-latent", choices=("prior", "posterior"), default="prior",
                     help="Latents in the training rollout: prior with AR(1) noise (as the simulator) or posterior")
    rft.add_argument("--rft-rho", type=float, default=0.8, help="AR(1) correlation of the latent noise (simulator: 0.8)")
    rft.add_argument("--rft-bptt", type=int, default=0, help="> 0: truncate backprop through the rollout every N steps")
    rft.add_argument("--rft-eval-steps", type=int, default=0,
                     help="> 0: validation also reports the simulator-faithful closed loop over this many steps")
    rft.add_argument("--train-batches", type=int, help="Cap training batches per epoch (fresh random subset each epoch)")
    rft.add_argument("--val-batches", type=int, help="Cap validation batches (fixed subset)")
    rft.add_argument("--kl-warmup-epochs", type=int, help="KL weight warm-up epochs (default: the recipe's 10)")
    forecast.add_argument("--collision-radius", type=float, default=0.4,
                          help="Collision hinge distance in metres (pairs closer in the ground truth are allowed "
                               "down to their own closest approach)")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Defaults < ``--config`` YAML < explicit flags."""
    import yaml

    parser = _parser()
    known, _ = parser.parse_known_args(argv)
    config = {}
    if known.config:
        config = yaml.safe_load(known.config.read_text()) or {}
        options = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - options)
        if unknown:
            parser.error(f"{known.config}: unknown options {unknown}")
        parser.set_defaults(**{key: value for key, value in config.items() if key not in {"datasets", "val_datasets"}})
    args = parser.parse_args(argv)
    args.train_entries = dataset_entries(args.datasets if args.datasets is not None else config.get("datasets"))
    val = args.val_datasets if args.val_datasets is not None else config.get("val_datasets")
    args.val_entries = dataset_entries(val) if val is not None else [dict(entry) for entry in args.train_entries]
    for key in ("run_dir", "encoder_checkpoint", "cache_dir", "init_checkpoint"):
        if isinstance(getattr(args, key), str):
            setattr(args, key, Path(getattr(args, key)))
    if not args.phase or not args.run_dir:
        parser.error("--phase and --run-dir are required (flags or config)")
    if not args.train_entries:
        parser.error("No training datasets: give --datasets or a config with `datasets`")
    if args.fully_connected:
        args.knn = args.radius = None
    if (args.target_slot or args.max_graph_nodes) and args.phase != "ssl":
        parser.error("--target-slot/--max-graph-nodes apply to --phase ssl (heads: forecast_bench.stars_ablation)")
    if args.max_graph_nodes and not args.target_slot:
        parser.error("--max-graph-nodes needs --target-slot")
    if args.phase == "forecast" and (args.stars_grid or args.stars_heading):
        parser.error("--stars-grid/--stars-heading apply to --phase ssl")
    if args.phase == "forecast":
        if args.encoder_mode != "scratch" and not args.encoder_checkpoint:
            parser.error(f"--encoder-mode {args.encoder_mode} needs --encoder-checkpoint")
        if args.encoder_mode == "scratch" and args.encoder_checkpoint:
            parser.error("--encoder-mode scratch does not take --encoder-checkpoint")
        if args.rollout_exec_steps < 0:
            parser.error("--rollout-exec-steps must be >= 0")
        if args.rollout_exec_steps and args.decoder in ("diffusion", "joint_diffusion"):
            parser.error("--rollout-exec-steps is not supported with --decoder diffusion")
        if args.init_checkpoint and (args.encoder_checkpoint or args.encoder_mode != "scratch"):
            parser.error("--init-checkpoint replaces --encoder-checkpoint (use --encoder-mode scratch)")
        if args.rft_steps < 0 or args.rft_eval_steps < 0:
            parser.error("--rft-steps/--rft-eval-steps must be >= 0")
        if args.selection == "closedloop" and not args.rft_eval_steps:
            parser.error("--selection closedloop needs --rft-eval-steps")
    elif args.encoder_checkpoint or args.rollout_exec_steps:
        parser.error("--encoder-checkpoint/--rollout-exec-steps apply to --phase forecast")
    return args


# ----------------------------------------------------------------------------- distributed


def setup_distributed(args) -> None:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.world_size > 1:
        backend = "nccl" if args.device.startswith("cuda") else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend)
        args.device = f"cuda:{args.local_rank}" if backend == "nccl" else args.device
    args.device_realized = training_device(args.device)
    args.main = args.rank == 0


def barrier(args) -> None:
    if args.world_size > 1:
        dist.barrier()


def _all_mean(value: torch.Tensor, args) -> torch.Tensor:
    if args.world_size > 1:
        value = value.clone()
        dist.all_reduce(value)
        value /= args.world_size
    return value


class _NullRun:
    """RunStatus stand-in for non-zero ranks."""

    def __getattr__(self, name):
        return lambda *a, **k: None

    def progress(self, iterable, **_):
        return iterable


# ----------------------------------------------------------------------------- data


def load_stores(entries: list[dict], split: str, args, forecasting, *, windows: str = "unique") -> WindowStore:
    """One store with every entry's split (rank 0 builds missing caches; the others wait and load)."""
    stores = []
    for entry in entries:
        path = Path(entry["path"])
        path = path if path.is_absolute() else ROOT / path
        cfg = Config(seeds=SeedConfig(args.data_seed, args.seed), forecasting=forecasting)
        if entry.get("split_strategy"):
            cfg = replace(cfg, split_strategy=entry["split_strategy"])
        source = SceneDataset(DatasetSpec(entry["name"], path), split, cfg)
        if not len(source):
            continue
        cap = entry.get("max_windows")
        cap = cap.get(split) if isinstance(cap, dict) else cap
        if cap and len(source.rows) > int(cap):
            # Seeded subset of the index rows, before window extraction (ATC has millions of heavily
            # overlapping windows). The cache key includes the row count.
            keep = np.sort(np.random.default_rng(args.data_seed).choice(len(source.rows), int(cap), replace=False))
            source.rows = [source.rows[i] for i in keep]
            args.run.log(f"{entry['name']}/{split}: capped to {int(cap)} windows")
        options = dict(windows=windows, forecasting=config_as_dict(forecasting), cache_dir=args.cache_dir,
                       workers=args.cache_workers, log=lambda message: args.run.log(message),
                       goals=bool(getattr(args, "goals", False)))
        if args.main:
            store = cached_store(source, entry["name"], split, args.obs, args.pred, **options)
        barrier(args)
        if not args.main:
            store = cached_store(source, entry["name"], split, args.obs, args.pred, **{**options, "workers": 0})
        if args.max_scenes:
            store = store.select(torch.arange(min(args.max_scenes, len(store))))
        if store.goal is not None and getattr(args, "goal_beyond_horizon", False):
            # Leak removal: a goal reached within the forecast horizon equals a supervised future
            # position; such agents get no goal (valid = 0), as at a benchmark goal beyond reach.
            store.goal[:, 2] = torch.where(store.goal[:, 3] <= args.pred, 0.0, store.goal[:, 2])
        stores.append(store)
    if not stores:
        raise RuntimeError(f"No {split} windows in {[entry['name'] for entry in entries]}")
    return concat_stores(stores)


def goal_stats(store: WindowStore) -> dict:
    """How often a goal is available and how often it is reached within the forecast horizon (the goal
    then equals a supervised future position), for benchmark targets and for all nodes."""
    goal, targets = store.goal.cpu(), store.ptr[:-1].cpu()
    future = store.future_valid.shape[1]
    stats = {}
    for name, rows in (("targets", goal[targets]), ("all_nodes", goal)):
        valid = rows[:, 2] > 0
        stats[name] = {"count": int(len(rows)), "goal_valid": float(valid.float().mean()),
                       "track_ends_in_future": float((valid & (rows[:, 3] <= future)).float().mean())}
    return stats


def epoch_batches(store: WindowStore, args, epoch: int, *, train: bool, weights=None) -> list[torch.Tensor]:
    """This rank's batches: the same global list on every rank (seeded), split round-robin."""
    generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
    batches = node_budget_batches(store.sizes, args.max_nodes, generator=generator, shuffle=train,
                                  weights=weights if train else None, max_graphs=args.max_graphs)
    if train and args.world_size > 1:
        # Every rank takes the same number of steps (DDP); tiny splits repeat batches.
        count = max(args.world_size, len(batches) // args.world_size * args.world_size)
        batches = (batches * math.ceil(count / len(batches)))[:count]
    batches = batches[args.rank::args.world_size]
    cap = getattr(args, "train_batches" if train else "val_batches", None)
    if cap:
        if not train:  # a fixed, evenly spread subset of the (unshuffled, size-sorted) validation batches
            batches = [batches[i] for i in np.linspace(0, len(batches) - 1, min(cap, len(batches))).astype(int)]
        batches = batches[:cap]
    return batches[: args.max_batches] if args.max_batches else batches


def _graph(args) -> dict:
    """The checkpoint's ``graph`` entry; the STARS-fidelity keys only when set (old checkpoints compare equal)."""
    graph = {"knn": args.knn, "radius": args.radius}
    extra = {"resample": getattr(args, "stars_grid", None), "heading_backfill": False if getattr(args, "stars_heading", False) else None,
             "target_slot": True if getattr(args, "target_slot", False) else None,
             "max_graph_nodes": getattr(args, "max_graph_nodes", None)}
    return {**graph, **{key: value for key, value in extra.items() if value is not None}}


def _batch(store, windows, args):
    return build_batch(store, windows.to(store.device), dt=CANONICAL_DT, **graph_options(_graph(args)))


# ----------------------------------------------------------------------------- losses


def _masked_distance(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean L2 error over valid (agent, step) pairs."""
    error = torch.linalg.norm(prediction.float() - torch.nan_to_num(target), dim=-1)
    return (error * valid).sum() / valid.sum().clamp(min=1)


def collision_loss(local: torch.Tensor, batch, radius: float) -> torch.Tensor:
    """Pairwise hinge ``relu(r_ij - d_ij(t))`` over the graph's edges and both agents' valid future
    steps, for local-frame predictions ``[N, P, 2]``; ``r_ij`` is ``radius`` or, if smaller, the
    pair's closest approach in the ground truth (groups that walk closer are not pushed apart).
    Normalised per supervised agent-step."""
    source, destination = batch.edge_index
    if not source.numel():
        return local.new_zeros(()).float()
    with torch.autocast(device_type=local.device.type, enabled=False):
        scene = to_scene(local.float(), batch.origin.float(), batch.heading.float())
        valid = batch.future_valid[source] & batch.future_valid[destination]                  # [E, P]
        future = torch.nan_to_num(batch.future.float())
        truth = torch.linalg.norm(future[source] - future[destination], dim=-1)
        allowed = torch.where(valid, truth, torch.full_like(truth, float("inf"))).min(dim=1, keepdim=True).values
        threshold = allowed.clamp(max=radius)
        distance = torch.linalg.norm(scene[source] - scene[destination] + 1e-9, dim=-1)
        violation = torch.relu(threshold - distance) * valid
        return violation.sum() / batch.future_valid.sum().clamp(min=1)


def forecast_loss(model, batch, args, kl_weight: float) -> dict[str, torch.Tensor]:
    """Teacher-forced loss, CVAE KL, optional collision hinge, and (optionally) the closed-loop
    multi-step rollout loss."""
    future_local = to_local(batch.future, batch.origin, batch.heading)
    valid = batch.future_valid.to(future_local.dtype)
    collide = getattr(args, "collision_weight", 0.0) > 0
    bom = getattr(args, "bom_weight", 0.0) > 0 and model.latent_dim
    fixed = collide and getattr(args, "collision_fixed_prior", True)
    if (collide or bom) and model.latent_dim:
        out = model.training_forward(batch, future_local, batch.future_valid,
                                     prior_samples=args.bom_samples if bom else (0 if fixed else 1),
                                     fixed_prior_samples=1 if fixed else 0)
    else:
        out = model.training_forward(batch, future_local, batch.future_valid)
    if "denoise" in out:  # diffusion decoder: the forecast loss is the denoising (epsilon) MSE
        zero = out["denoise"].new_zeros(())
        return {"forecast": out["denoise"], "kl": zero, "rollout": zero, "loss": out["denoise"]}
    losses = {"forecast": _masked_distance(out["prediction"], future_local, valid)}
    losses["kl"] = out["kl"].float().sum() / batch.future_valid.any(dim=-1).sum().clamp(min=1) \
        if out["kl"] is not None else losses["forecast"].new_zeros(())
    losses["rollout"] = losses["forecast"].new_zeros(())
    losses["collision"] = losses["forecast"].new_zeros(())
    losses["bom"] = losses["forecast"].new_zeros(())
    if bom:
        error = torch.linalg.norm(out["prior_prediction"].float() - torch.nan_to_num(future_local)[None], dim=-1)
        ade = (error * valid[None]).sum(-1) / valid.sum(-1).clamp(min=1)[None]        # [S, N]
        has = batch.future_valid.any(-1)
        losses["bom"] = (ade.min(0).values * has).sum() / has.sum().clamp(min=1)
    if collide:
        generated = (out["fixed_prior_prediction" if fixed else "prior_prediction"][0] if model.latent_dim
                     else out["prediction"])
        losses["collision"] = collision_loss(generated, batch, args.collision_radius)
    losses["rft"] = losses["forecast"].new_zeros(())
    steps = getattr(args, "rft_current_steps", 0)
    if steps:
        simulated = closed_loop_rollout(model, batch, steps, latent=args.rft_latent, rho=args.rft_rho,
                                        bptt=args.rft_bptt)
        losses["rft"] = _masked_distance(simulated, batch.future[:, :steps], valid[:, :steps])
        if getattr(args, "rft_speed_weight", 0.0) > 0:
            # Progress matching: | |p_k - p_0| - |g_k - p_0| | (distance travelled from the start, simulated vs
            # true) so the rollout loss cannot be lowered by hedging (slowing down under uncertainty). Unlike
            # a per-step speed match it cannot be met by zig-zagging (that raised turning ~5x).
            start = batch.obs_xy[:, -1:].float()
            reach = lambda path: torch.linalg.norm(path - start + 1e-9, dim=-1)
            gap = (reach(simulated) - reach(torch.nan_to_num(batch.future[:, :steps].float()))).abs()
            fv = batch.future_valid[:, :steps].float()
            losses["rft"] = losses["rft"] + args.rft_speed_weight * (gap * fv).sum() / fv.sum().clamp(min=1)
    if args.rollout_exec_steps:
        # Closed loop: each agent keeps the intent z inferred from its future for the whole rollout.
        rounds = math.ceil(model.prediction_steps / args.rollout_exec_steps)
        simulated = rollout(model, batch, rounds=rounds, exec_steps=args.rollout_exec_steps, knn=args.knn,
                            radius=args.radius, z=out["z"], keep_graph=True)[:, :model.prediction_steps]
        losses["rollout"] = _masked_distance(simulated, batch.future, valid)
    losses["loss"] = (getattr(args, "open_loop_weight", 1.0) * (losses["forecast"] + kl_weight * losses["kl"])
                      + getattr(args, "rft_weight", 1.0) * losses["rft"] + args.rollout_weight * losses["rollout"]
                      + getattr(args, "collision_weight", 0.0) * losses["collision"]
                      + getattr(args, "bom_weight", 0.0) * losses["bom"])
    return losses


class _Step(nn.Module):
    """One training loss as a module forward, so DistributedDataParallel can synchronise it."""

    def __init__(self, model: nn.Module, args) -> None:
        super().__init__()
        self.model, self.args = model, args

    def forward(self, batch, weight: float) -> dict[str, torch.Tensor]:
        if self.args.phase == "ssl":
            return self.model.loss(batch, weight, self.args.node_loss_weight)
        return forecast_loss(self.model, batch, self.args, weight)


def _autocast(args):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=args.amp and args.device_realized.type == "cuda")


# ----------------------------------------------------------------------------- evaluation


TURN_MIN_STEP = 0.08   # metres per 0.4 s step (0.2 m/s): turning is measured only between moving steps


@torch.no_grad()
def closed_loop_metrics(model, store: WindowStore, batches, args, steps: int) -> dict[str, float]:
    """Simulator-faithful closed loop (one prior sample, AR(1) latent noise, seeded per batch) over
    ``steps``: ADE / FDE per dataset (``cl_<name>/ade``) and pooled, mean speed of the simulated
    vs ground-truth agents over the same supervised steps, and mean absolute turning per step."""
    model.eval()
    names = store.names
    sums = torch.zeros(len(names), 11, dtype=torch.float64, device=store.device)
    for index, windows in enumerate(batches):
        batch = _batch(store, windows, args)
        generator = torch.Generator().manual_seed(1000003 * args.seed + index)
        with _autocast(args):
            sim = closed_loop_rollout(model, batch, steps, rho=args.rft_rho, generator=generator).float()
        truth, valid = torch.nan_to_num(batch.future[:, :steps].float()), batch.future_valid[:, :steps]
        error = torch.linalg.norm(sim - truth, dim=-1)
        last = batch.obs_xy[:, -1:].float()
        step_ok = valid & torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)
        speed = lambda path: torch.linalg.norm(torch.diff(torch.cat([last, path], 1), dim=1), dim=-1) / CANONICAL_DT
        def turning(path):
            d = torch.diff(torch.cat([last, path], 1), dim=1)
            a, b = d[:, :-1], d[:, 1:]
            angle = torch.atan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1)).abs()
            ok = step_ok[:, 1:] & (torch.linalg.norm(a, dim=-1) > TURN_MIN_STEP) & (torch.linalg.norm(b, dim=-1) > TURN_MIN_STEP)
            return angle, ok
        sim_turn, sim_ok = turning(sim)
        gt_turn, gt_ok = turning(truth)
        complete = valid.all(-1)
        graph_dataset = batch.dataset[batch.batch]
        per_node = torch.stack([(error * valid).sum(-1), valid.sum(-1), error[:, -1] * complete, complete.float(),
                                (speed(sim) * step_ok).sum(-1), (speed(truth) * step_ok).sum(-1), step_ok.sum(-1),
                                (sim_turn * sim_ok).sum(-1), sim_ok.sum(-1), (gt_turn * gt_ok).sum(-1), gt_ok.sum(-1)],
                               -1).double()
        sums.index_add_(0, graph_dataset, per_node)
    if args.world_size > 1:
        dist.all_reduce(sums)
    out = {}

    def summarise(row, prefix):
        ratio = lambda a, b: float(row[a] / row[b].clamp(min=1))
        out.update({f"{prefix}ade": ratio(0, 1), f"{prefix}fde": ratio(2, 3), f"{prefix}speed": ratio(4, 6),
                    f"{prefix}gt_speed": ratio(5, 6), f"{prefix}turn": ratio(7, 8), f"{prefix}gt_turn": ratio(9, 10)})

    summarise(sums.sum(0), "cl_")
    for i, name in enumerate(names):
        if sums[i, 1] > 0:
            summarise(sums[i], f"cl_{name}/")
    return out


@torch.no_grad()
def evaluate_forecaster(model, store: WindowStore, batches, args, *, samples: int, rollout_exec_steps: int = 0):
    """Teacher-forced best-of-K metrics (per dataset too) and, with ``rollout_exec_steps``, closed-loop ones."""
    model.eval()
    teacher, closed = MetricAccumulator(), MetricAccumulator()
    for windows in batches:
        batch = _batch(store, windows, args)
        with _autocast(args):
            local = model.predict(batch, samples=samples)
        batch_metrics(to_scene(local.float(), batch.origin, batch.heading), batch, teacher, dataset_names=store.names)
        if rollout_exec_steps:
            rounds = math.ceil(model.prediction_steps / rollout_exec_steps)
            runs = samples if model.latent_dim else 1
            with _autocast(args):
                simulated = torch.stack([
                    rollout(model, batch, rounds=rounds, exec_steps=rollout_exec_steps, knn=args.knn,
                            radius=args.radius)[:, :model.prediction_steps] for _ in range(runs)
                ])
            batch_metrics(simulated.float(), batch, closed)
    teacher.all_reduce()
    closed.all_reduce()
    result = teacher.result()
    result.update({f"rollout_{key}": value for key, value in closed.result().items()})
    return result


@torch.no_grad()
def evaluate_ssl(model, store, batches, args, beta) -> dict[str, float]:
    model.eval()
    sums = {key: torch.zeros((), device=store.device, dtype=torch.float64) for key in ("loss", "edge", "node", "kl")}
    nodes = torch.zeros((), device=store.device, dtype=torch.float64)
    for windows in batches:
        batch = _batch(store, windows, args)
        with _autocast(args):
            losses = model.loss(batch, beta, args.node_loss_weight)
        for key in sums:
            sums[key] += losses[key].double() * batch.num_nodes
        nodes += batch.num_nodes
    stacked = torch.stack([*sums.values(), nodes])
    if args.world_size > 1:
        dist.all_reduce(stacked)
    return {key: float(stacked[i] / stacked[-1].clamp(min=1)) for i, key in enumerate(sums)}


def selection_name(args, samples: int) -> str:
    if args.selection == "closedloop":
        return f"val_closedloop{args.rft_eval_steps}_ADE_macro"
    return f"val_top{samples}_agent_ADE_{args.selection}"


def selection_metric(metrics: dict[str, float], names: list[str], how: str) -> float:
    if how == "closedloop":
        return float(np.mean([metrics[f"cl_{name}/ade"] for name in names if f"cl_{name}/ade" in metrics]))
    if how == "macro":
        values = [metrics[f"{name}/agent_ade"] for name in names if f"{name}/agent_ade" in metrics]
        if values:
            return float(np.mean(values))
    return metrics["agent_ade"]


# ----------------------------------------------------------------------------- checkpoints


def _save(path: Path, model, args, *, selection, extra=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": "stars_ma",
        "phase": args.phase,
        "state_dict": model.state_dict(),
        "encoder_recipe": args.encoder_recipe,
        "decoder_recipe": args.decoder_recipe,
        "recipe": args.recipe,
        "graph": _graph(args),
        "window": {"observation_steps": args.obs, "prediction_steps": args.pred, "dt": CANONICAL_DT},
        "datasets": {"train": args.train_entries, "val": args.val_entries},
        "checkpoint_selection": selection,
        "run_config": args.run_config,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        **(extra or {}),
        **checkpoint_metadata(path.parents[1]),
    }, temporary)
    os.replace(temporary, path)


def load_forecaster(path: Path, device: torch.device) -> tuple[StarsMultiAgentForecaster, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != "stars_ma" or payload.get("phase") != "forecast":
        raise ValueError(f"{path} is not a multi-agent STARS forecaster checkpoint")
    window = payload["window"]
    model = StarsMultiAgentForecaster(window["observation_steps"], window["prediction_steps"], dt=window["dt"],
                                      encoder_recipe=payload["encoder_recipe"], decoder_recipe=payload["decoder_recipe"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


# ----------------------------------------------------------------------------- training


MAX_NONFINITE_STREAK = 100   # consecutive non-finite gradient steps before a run is stopped


def _scaled_lr(base: float, args) -> float:
    if args.learning_rate is not None:
        return args.learning_rate
    factor = args.world_size * args.max_nodes / REFERENCE_NODES
    factor = {"sqrt": math.sqrt(factor), "linear": factor, "none": 1.0}[args.lr_scaling]
    return min(base * factor, max(base, args.max_learning_rate))


def _scheduler(optimizer, args, steps_per_epoch: int):
    if args.schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                          patience=max(1, args.recipe["patience"] // 3))
    warmup = max(1, int(args.warmup_epochs * steps_per_epoch))
    total = max(warmup + 1, args.recipe["max_epochs"] * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if args.schedule == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / (total - warmup))
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _build_model(args, payload):
    if args.phase == "ssl":
        model = StarsMotionVAE(args.obs, args.encoder_recipe)
        groups = lambda lr: [{"params": model.parameters(), "lr": lr}]
        return model, groups
    model = StarsMultiAgentForecaster(args.obs, args.pred, dt=CANONICAL_DT,
                                      encoder_recipe=args.encoder_recipe, decoder_recipe=args.decoder_recipe)
    if getattr(args, "init_payload", None) is not None:
        model.load_state_dict(args.init_payload["state_dict"])
        args.run.log("forecaster initialised", checkpoint=str(args.init_checkpoint))
    if payload is not None:
        encoder_state = {key.removeprefix("encoder."): value for key, value in payload["state_dict"].items()
                         if key.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_state)
        args.run.log("phase-1 encoder loaded", checkpoint=str(args.encoder_checkpoint), mode=args.encoder_mode)
    model.freeze_encoder = args.encoder_mode == "frozen"
    if model.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        groups = lambda lr: [{"params": model.decoder.parameters(), "lr": lr}]
    elif args.encoder_mode == "finetune":
        ratio = FORECAST_RECIPE["encoder_learning_rate"] / FORECAST_RECIPE["learning_rate"]
        groups = lambda lr: [{"params": model.decoder.parameters(), "lr": lr},
                             {"params": model.encoder.parameters(), "lr": lr * ratio}]
    else:
        groups = lambda lr: [{"params": model.parameters(), "lr": lr}]
    return model, groups


def _maybe_compile(model, args):
    if not args.compile:
        return
    encoder = model.encoder
    encoder.base.edge_encoder = torch.compile(encoder.base.edge_encoder, dynamic=True)
    if encoder.use_node_features:
        encoder.node_encoder = torch.compile(encoder.node_encoder, dynamic=True)


def train(args) -> None:
    forecasting = resolve_forecasting_config(preset=DEFAULT_PRESET)
    args.obs, args.pred = forecasting.window.observation_steps, forecasting.window.prediction_steps
    payload = None
    encoder_overrides = {"hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
                         "use_node_features": False if args.edge_only else None, "dropout": args.dropout,
                         "timesteps": args.stars_grid}
    args.encoder_recipe = {**ENCODER_RECIPE, **{k: v for k, v in encoder_overrides.items() if v is not None}}
    if args.encoder_checkpoint:
        payload = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("model") != "stars_ma" or payload.get("phase") != "ssl":
            raise ValueError(f"{args.encoder_checkpoint} is not a multi-agent STARS --phase ssl checkpoint")
        args.encoder_recipe = dict(payload["encoder_recipe"])
        if payload["graph"] != {"knn": args.knn, "radius": args.radius}:
            raise ValueError(f"Graph topology differs from the encoder checkpoint ({payload['graph']})")
    args.decoder_recipe = {
        **DECODER_RECIPE,
        "latent_dim": args.cvae_latent_dim if args.decoder == "cvae" else 0,
        "diffusion": args.decoder in ("diffusion", "joint_diffusion"),
        "diffusion_social_layers": args.diffusion_social_layers if args.decoder == "joint_diffusion" else 0,
        "diffusion_steps": args.diffusion_steps,
        "sample_steps": args.sample_steps,
        "social_layer": not args.no_social_layer,
        "representation": args.representation,
        "goal": args.goal,
        "refine": args.refine,
        **({"hidden_dim": args.decoder_hidden_dim} if args.decoder_hidden_dim else {}),
        **({"dropout": args.decoder_dropout} if args.decoder_dropout is not None else {}),
    }
    args.init_payload = None
    if args.phase == "forecast" and args.init_checkpoint:
        # Fine-tuning: the architecture is the checkpoint's, whatever the decoder flags say.
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if init.get("model") != "stars_ma" or init.get("phase") != "forecast":
            raise ValueError(f"{args.init_checkpoint} is not a multi-agent STARS forecast checkpoint")
        if init["graph"] != {"knn": args.knn, "radius": args.radius}:
            raise ValueError(f"Graph topology differs from the init checkpoint ({init['graph']})")
        if (init["window"]["observation_steps"], init["window"]["prediction_steps"]) != (args.obs, args.pred):
            raise ValueError(f"Window differs from the init checkpoint ({init['window']})")
        args.encoder_recipe, args.decoder_recipe = dict(init["encoder_recipe"]), dict(init["decoder_recipe"])
        args.goal = bool(args.decoder_recipe.get("goal", False))
        args.init_payload = init
    if args.rft_steps > args.pred or args.rft_eval_steps > args.pred:
        raise ValueError(f"--rft-steps/--rft-eval-steps must be <= the prediction steps ({args.pred})")
    args.goals = args.goal
    base_recipe = SSL_RECIPE if args.phase == "ssl" else FORECAST_RECIPE
    args.recipe = {**base_recipe, **({"max_epochs": args.epochs} if args.epochs else {})}
    args.recipe["learning_rate"] = _scaled_lr(base_recipe["learning_rate"], args)
    if args.weight_decay is not None:
        args.recipe["weight_decay"] = args.weight_decay
    if args.phase == "forecast":
        args.recipe["kl_weight"] = args.kl_weight
    elif args.beta_warmup_epochs is not None:
        args.recipe["beta_warmup_epochs"] = args.beta_warmup_epochs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    skip = {"device_realized", "run_config", "run", "main", "init_payload"}
    args.run_config = {
        **{key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items() if key not in skip},
        "model": "stars_ma",
        "global_max_nodes": args.max_nodes * args.world_size,
        "device_realized": str(args.device_realized),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": str(REFERENCE.relative_to(ROOT)),
        **forecasting.flat_provenance(),
        "forecasting_config": config_as_dict(forecasting),
    }
    args.run = RunStatus(args.run_dir, args.run_config, resume=args.resume) if args.main else _NullRun()
    run = args.run
    try:
        run.update(status="running", stage="loading_datasets")
        device = args.device_realized
        train_store = load_stores(args.train_entries, "train", args, forecasting).to(device)
        val_store = load_stores(args.val_entries, "val", args, forecasting).to(device)
        if args.target_slot:  # STARS's single-target graphs: one seeded target per window, robot slot
            train_store = per_target_store(train_store, targets="one", max_graph_nodes=args.max_graph_nodes,
                                           seed=args.data_seed)[0].to(device)
            val_store = per_target_store(val_store, targets="one", max_graph_nodes=args.max_graph_nodes,
                                         seed=args.data_seed)[0].to(device)
        weights = window_weights(train_store, {entry["name"]: entry["weight"] for entry in args.train_entries})
        counts = {name: int((train_store.dataset == i).sum()) for i, name in enumerate(train_store.names)}
        run.update(stage="datasets_ready", n_train_windows=len(train_store), n_val_windows=len(val_store),
                   n_train_nodes=int(train_store.ptr[-1]), train_windows_per_dataset=counts,
                   val_datasets=val_store.names)
        if train_store.goal is not None:
            run.update(train_goal_stats=goal_stats(train_store))

        model, groups = _build_model(args, payload)
        model = model.to(device)
        model.encoder.checkpoint_temporal = args.activation_checkpointing
        _maybe_compile(model, args)
        optimizer = torch.optim.AdamW(groups(args.recipe["learning_rate"]), weight_decay=args.recipe["weight_decay"])
        steps_per_epoch = len(epoch_batches(train_store, args, 0, train=True, weights=weights))
        if not steps_per_epoch:
            raise RuntimeError("The training split yields no batches")
        scheduler = _scheduler(optimizer, args, steps_per_epoch)
        step_module = _Step(model, args)
        if args.world_size > 1:
            step_module = nn.parallel.DistributedDataParallel(
                step_module, device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True, gradient_as_bucket_view=True)
        stopper = EarlyStopping(args.recipe["patience"], args.recipe["min_delta"])
        start_epoch, best_epoch, global_step = 1, None, 0
        latest = args.run_dir / "checkpoints" / "latest.pt"
        if args.resume and latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            stopper = EarlyStopping(**state["stopper"])
            start_epoch, best_epoch, global_step = state["epoch"] + 1, state["best_epoch"], state["global_step"]
            run.log("resumed", epoch=state["epoch"])
        trainable = [p for p in model.parameters() if p.requires_grad]
        run.update(stage=args.phase, parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in trainable), steps_per_epoch=steps_per_epoch,
                   peak_learning_rate=args.recipe["learning_rate"], world_size=args.world_size)
        samples = args.samples if (args.phase == "forecast" and model.stochastic) else 1
        checkpoint = args.run_dir / "checkpoints" / "best.pt"

        def save_latest(epoch: int) -> None:
            latest.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                        "global_step": global_step,
                        "stopper": {"patience": stopper.patience, "min_delta": stopper.min_delta,
                                    "best": stopper.best, "bad_epochs": stopper.bad_epochs}},
                       latest.with_suffix(".tmp"))
            os.replace(latest.with_suffix(".tmp"), latest)

        if args.init_payload is not None and start_epoch == 1 and args.phase == "forecast" and not args.benchmark:
            # Epoch 0: the initial checkpoint on the same validation, so best.pt is never worse than it.
            val_batches = epoch_batches(val_store, args, 0, train=False)
            metrics = evaluate_forecaster(model, val_store, val_batches, args, samples=samples,
                                          rollout_exec_steps=args.rollout_exec_steps)
            if args.rft_eval_steps:
                metrics.update(closed_loop_metrics(model, val_store, val_batches, args, args.rft_eval_steps))
            selection = selection_metric(metrics, val_store.names, args.selection)
            run.metric({"stage": args.phase, "epoch": 0, "selection": selection, **{f"val_{k}": v for k, v in metrics.items()}})
            stopper.update(selection)
            best_epoch = 0
            if args.main:
                _save(checkpoint, model, args, selection={"metric": selection_name(args, samples), "value": selection,
                                                          "best_epoch": 0, "metrics": metrics})
        for epoch in range(start_epoch, args.recipe["max_epochs"] + 1):
            model.train()
            if getattr(model, "freeze_encoder", False):
                model.encoder.eval()  # frozen: no dropout, as features of a fixed encoder
            if args.phase == "ssl":
                r = args.recipe
                weight = min(r["beta_max"], r["beta_max"] * epoch / max(1, r["beta_warmup_epochs"]))
            else:
                warm = args.kl_warmup_epochs if args.kl_warmup_epochs is not None else args.recipe["kl_warmup_epochs"]
                weight = args.recipe["kl_weight"] * min(1.0, epoch / max(1, warm))
                if args.rft_steps:  # curriculum: rollout length from --rft-start-steps to --rft-steps
                    start = args.rft_start_steps or args.rft_steps
                    ramp = min(1.0, (epoch - 1) / args.rft_ramp_epochs) if args.rft_ramp_epochs else 1.0
                    args.rft_current_steps = int(round(start + (args.rft_steps - start) * ramp))
            batches = epoch_batches(train_store, args, epoch, train=True, weights=weights)
            if args.benchmark:
                batches = (batches * math.ceil(args.benchmark / max(1, len(batches))))[: args.benchmark]
            sums = {}
            nodes = edges = skipped = bad_streak = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for index, windows in enumerate(run.progress(batches, desc=f"{args.phase} epoch {epoch}")):
                if args.benchmark and index == min(5, len(batches) // 4):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started, nodes, edges = time.perf_counter(), 0, 0  # exclude warm-up steps
                batch = _batch(train_store, windows, args)
                with _autocast(args):
                    losses = step_module(batch, weight)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, args.recipe["grad_clip"])
                # The norm is taken after DDP's all-reduce, so every rank skips the same steps.
                if not torch.isfinite(norm):
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    bad_streak += 1
                    if bad_streak >= MAX_NONFINITE_STREAK:
                        raise FloatingPointError(f"{bad_streak} consecutive non-finite gradients (epoch {epoch})")
                    continue
                bad_streak = 0
                optimizer.step()
                if args.schedule != "plateau":
                    scheduler.step()
                global_step += 1
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + value.detach().float()
                nodes += batch.num_nodes
                edges += int(batch.edge_index.shape[1])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            train_metrics = {f"train_{k}": float(_all_mean(v / max(1, len(batches)), args)) for k, v in sums.items()}
            train_metrics["train_skipped_steps"] = skipped
            throughput = {"epoch_seconds": elapsed, "nodes_per_second": nodes * args.world_size / elapsed,
                          "edges_per_second": edges * args.world_size / elapsed,
                          "steps_per_second": len(batches) / elapsed,
                          "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                          "learning_rate": optimizer.param_groups[0]["lr"]}
            if args.benchmark:
                run.update(status="complete", stage="benchmark", **throughput, **train_metrics)
                if args.main:
                    print({**throughput, **train_metrics}, flush=True)
                return

            if epoch % args.eval_every and epoch != args.recipe["max_epochs"]:
                run.metric({"stage": args.phase, "epoch": epoch, **train_metrics, **throughput})
                if args.main:  # resumable after every epoch, not only after validated ones
                    save_latest(epoch)
                continue
            val_batches = epoch_batches(val_store, args, 0, train=False)
            if args.phase == "ssl":
                metrics = evaluate_ssl(model, val_store, val_batches, args,
                                       weight if args.ssl_val_beta is None else args.ssl_val_beta)
                selection, name = metrics["loss"], "val_ssl_loss"
            else:
                metrics = evaluate_forecaster(model, val_store, val_batches, args, samples=samples,
                                              rollout_exec_steps=args.rollout_exec_steps)
                if args.rft_eval_steps:
                    metrics.update(closed_loop_metrics(model, val_store, val_batches, args, args.rft_eval_steps))
                selection, name = selection_metric(metrics, val_store.names, args.selection), selection_name(args, samples)
            if args.schedule == "plateau":
                scheduler.step(selection)
            if getattr(args, "rft_current_steps", 0):
                metrics["rft_steps"] = args.rft_current_steps
            run.metric({"stage": args.phase, "epoch": epoch, "beta" if args.phase == "ssl" else "kl_weight": weight,
                        "selection": selection, **train_metrics, **throughput,
                        **{f"val_{k}": v for k, v in metrics.items()}})
            improved, stopped = stopper.update(selection)
            if improved:
                best_epoch = epoch
                if args.main:
                    _save(checkpoint, model, args, selection={"metric": name, "value": selection,
                                                              "best_epoch": epoch, "metrics": metrics})
            if args.main:
                save_latest(epoch)
            run.update(stage=args.phase, epoch=epoch, best_selection=stopper.best, best_epoch=best_epoch,
                       **{key: throughput[key] for key in ("epoch_seconds", "nodes_per_second")})
            if stopped:
                run.log("early stopping", epoch=epoch, best=stopper.best)
                break
        if best_epoch is None:
            raise RuntimeError("No finite validation metric was observed")
        run.update(status="complete", stage="training", best_validation_metric=stopper.best, best_epoch=best_epoch)
    except Exception as error:
        run.log("run failed", error=repr(error), traceback=traceback.format_exc())
        run.update(status="failed", error=repr(error))
        raise


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_distributed(args)
    try:
        train(args)
    finally:
        if args.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    if args.world_size > 1:
        # Multi-GPU runs on all datasets hit a bus error in interpreter teardown after training
        # finished and saved, which marked successful jobs FAILED; everything is on disk, so skip it.
        import sys

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sys.stdout.flush(), sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
