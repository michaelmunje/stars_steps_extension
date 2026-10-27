"""Train multi-agent STARS (``stars_ma_model``): self-supervised encoder, then forecaster.

Phase 1, ``--phase ssl``: the extended STARS encoder (node motion + SE(2) edge features at the
native rate, neighbourhood graphs) is trained with STARS's self-supervised graph-VAE objective
plus node-motion reconstruction, selected on validation loss.

Phases 2-3, ``--phase forecast``: a multi-agent decoder forecasts every pedestrian of each scene
window. ``--encoder-mode`` chooses a frozen or jointly fine-tuned phase-1 encoder
(``--encoder-checkpoint``) or one trained from scratch; ``--decoder cvae`` adds the probabilistic
decoder (phase 3). Forecast checkpoints are selected on the mean over validation datasets of
best-of-K all-agent ADE.

Datasets come from a YAML config (``--config``, see ``configs/stars_ma/``) or ``--datasets``.
Every option below can also be set in the config (same names, underscores); explicit flags win.

Scale: every split is extracted once into a cached ``WindowStore`` (``stars_ma_batch``) that
lives on the GPU; graphs are built batch-wise on the device; batches hold up to ``--max-nodes``
pedestrians of similar-sized windows; ``torchrun --nproc-per-node 4`` trains data-parallel
(DDP) across GPUs; ``--amp`` (default) runs the networks in bf16.
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
from baselines.stars_ma_batch import WindowStore, build_batch, cached_store, concat_stores, node_budget_batches, window_weights
from baselines.stars_ma_graph import to_local, to_scene
from baselines.stars_ma_metrics import MetricAccumulator, batch_metrics
from baselines.stars_ma_model import DECODER_RECIPE, ENCODER_RECIPE, StarsMotionVAE, StarsMultiAgentForecaster
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
    # Two-day converted subset: 2013-09-29 is the test split, 2013-05-22 is train/val (decision 93).
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
    "kl_weight": 0.1,  # 1.0 collapses the latent on STEPS (docs/decisions.md 32-35)
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, help="Early-stopping patience in validations (0 disables)")
    parser.add_argument("--max-nodes", type=int, default=16384, help="Pedestrians per batch on each GPU")
    parser.add_argument("--learning-rate", type=float,
                        help=f"Peak LR (default: the recipe LR x sqrt(global nodes per batch / {REFERENCE_NODES}))")
    parser.add_argument("--max-learning-rate", type=float, default=3e-3, help="Cap for the scaled LR")
    parser.add_argument("--warmup-epochs", type=float, default=2.0, help="Linear warm-up before the cosine decay")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast on CUDA")
    graph = parser.add_argument_group("graph")
    graph.add_argument("--knn", type=int, default=16, help="Sources kept per destination (default 16)")
    graph.add_argument("--radius", type=float, default=10.0, help="Neighbourhood radius in metres (default 10)")
    graph.add_argument("--fully-connected", action="store_true", help="STARS's topology: ignore --knn/--radius")
    encoder = parser.add_argument_group("encoder")
    encoder.add_argument("--hidden-dim", type=int)
    encoder.add_argument("--latent-dim", type=int)
    encoder.add_argument("--edge-only", action="store_true", help="Ablation: STARS's edge-only encoder (no node motion)")
    forecast = parser.add_argument_group("forecast")
    forecast.add_argument("--encoder-checkpoint", type=Path, help="A --phase ssl checkpoint (frozen/finetune)")
    forecast.add_argument("--encoder-mode", choices=("frozen", "finetune", "scratch"), default="frozen")
    forecast.add_argument("--init-checkpoint", type=Path,
                          help="A --phase forecast checkpoint: start from all its weights and recipes (with "
                               "--encoder-mode scratch, i.e. everything trainable at one LR)")
    forecast.add_argument("--decoder", choices=("deterministic", "cvae"), default="deterministic")
    forecast.add_argument("--cvae-latent-dim", type=int, default=32)
    forecast.add_argument("--representation", choices=("mu", "hidden"), default=DECODER_RECIPE["representation"])
    forecast.add_argument("--kl-weight", type=float, default=FORECAST_RECIPE["kl_weight"])
    forecast.add_argument("--samples", type=int, default=20, help="K for best-of-K validation (CVAE)")
    forecast.add_argument("--decoder-hidden-dim", type=int)
    forecast.add_argument("--decoder-dropout", type=float)
    forecast.add_argument("--encoder-lr-ratio", type=float,
                          help="Fine-tuning: encoder LR / decoder LR (default: the recipe's 0.1)")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float, help="Encoder dropout (scratch encoders / phase 1)")
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
    for key in ("run_dir", "encoder_checkpoint", "init_checkpoint", "cache_dir"):
        if isinstance(getattr(args, key), str):
            setattr(args, key, Path(getattr(args, key)))
    if not args.phase or not args.run_dir:
        parser.error("--phase and --run-dir are required (flags or config)")
    if not args.train_entries:
        parser.error("No training datasets: give --datasets or a config with `datasets`")
    if args.fully_connected:
        args.knn = args.radius = None
    if args.phase == "forecast":
        if args.encoder_mode != "scratch" and not args.encoder_checkpoint:
            parser.error(f"--encoder-mode {args.encoder_mode} needs --encoder-checkpoint")
        if args.encoder_mode == "scratch" and args.encoder_checkpoint:
            parser.error("--encoder-mode scratch does not take --encoder-checkpoint")
        if args.init_checkpoint and (args.encoder_checkpoint or args.encoder_mode != "scratch"):
            parser.error("--init-checkpoint takes --encoder-mode scratch and no --encoder-checkpoint")
    elif args.encoder_checkpoint or args.init_checkpoint:
        parser.error("--encoder-checkpoint/--init-checkpoint apply to --phase forecast")
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
            # overlapping windows; decision 93). The cache key includes the row count.
            keep = np.sort(np.random.default_rng(args.data_seed).choice(len(source.rows), int(cap), replace=False))
            source.rows = [source.rows[i] for i in keep]
            args.run.log(f"{entry['name']}/{split}: capped to {int(cap)} windows")
        options = dict(windows=windows, forecasting=config_as_dict(forecasting), cache_dir=args.cache_dir,
                       workers=args.cache_workers, log=lambda message: args.run.log(message))
        if args.main:
            store = cached_store(source, entry["name"], split, args.obs, args.pred, **options)
        barrier(args)
        if not args.main:
            store = cached_store(source, entry["name"], split, args.obs, args.pred, **{**options, "workers": 0})
        if args.max_scenes:
            store = store.select(torch.arange(min(args.max_scenes, len(store))))
        stores.append(store)
    if not stores:
        raise RuntimeError(f"No {split} windows in {[entry['name'] for entry in entries]}")
    return concat_stores(stores)


def epoch_batches(store: WindowStore, args, epoch: int, *, train: bool, weights=None) -> list[torch.Tensor]:
    """This rank's batches: the same global list on every rank (seeded), split round-robin."""
    generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
    batches = node_budget_batches(store.sizes, args.max_nodes, generator=generator, shuffle=train,
                                  weights=weights if train else None)
    if train and args.world_size > 1:
        # Every rank takes the same number of steps (DDP); tiny splits repeat batches.
        count = max(args.world_size, len(batches) // args.world_size * args.world_size)
        batches = (batches * math.ceil(count / len(batches)))[:count]
    batches = batches[args.rank::args.world_size]
    return batches[: args.max_batches] if args.max_batches else batches


def _batch(store, windows, args):
    return build_batch(store, windows.to(store.device), dt=CANONICAL_DT, knn=args.knn, radius=args.radius)


# ----------------------------------------------------------------------------- losses


def forecast_loss(model, batch, kl_weight: float) -> dict[str, torch.Tensor]:
    """Teacher-forced mean L2 error over valid (agent, step) pairs plus the CVAE KL per supervised agent."""
    future_local = to_local(batch.future, batch.origin, batch.heading)
    out = model.training_forward(batch, future_local, batch.future_valid)
    error = torch.linalg.norm(out["prediction"].float() - torch.nan_to_num(future_local), dim=-1)
    valid = batch.future_valid.to(error.dtype)
    losses = {"forecast": (error * valid).sum() / valid.sum().clamp(min=1)}
    losses["kl"] = out["kl"].float().sum() / batch.future_valid.any(dim=-1).sum().clamp(min=1) \
        if out["kl"] is not None else losses["forecast"].new_zeros(())
    losses["loss"] = losses["forecast"] + kl_weight * losses["kl"]
    return losses


class _Step(nn.Module):
    """One training loss as a module forward, so DistributedDataParallel can synchronise it."""

    def __init__(self, model: nn.Module, args) -> None:
        super().__init__()
        self.model, self.args = model, args

    def forward(self, batch, weight: float) -> dict[str, torch.Tensor]:
        if self.args.phase == "ssl":
            return self.model.loss(batch, weight)
        return forecast_loss(self.model, batch, weight)


def _autocast(args):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=args.amp and args.device_realized.type == "cuda")


# ----------------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate_forecaster(model, store: WindowStore, batches, args, *, samples: int) -> dict[str, float]:
    """Teacher-forced best-of-K metrics, pooled and per dataset."""
    model.eval()
    teacher = MetricAccumulator()
    for windows in batches:
        batch = _batch(store, windows, args)
        with _autocast(args):
            local = model.predict(batch, samples=samples)
        batch_metrics(to_scene(local.float(), batch.origin, batch.heading), batch, teacher, dataset_names=store.names)
    teacher.all_reduce()
    return teacher.result()


@torch.no_grad()
def evaluate_ssl(model, store, batches, args, beta) -> dict[str, float]:
    model.eval()
    sums = {key: torch.zeros((), device=store.device, dtype=torch.float64) for key in ("loss", "edge", "node", "kl")}
    nodes = torch.zeros((), device=store.device, dtype=torch.float64)
    for windows in batches:
        batch = _batch(store, windows, args)
        with _autocast(args):
            losses = model.loss(batch, beta)
        for key in sums:
            sums[key] += losses[key].double() * batch.num_nodes
        nodes += batch.num_nodes
    stacked = torch.stack([*sums.values(), nodes])
    if args.world_size > 1:
        dist.all_reduce(stacked)
    return {key: float(stacked[i] / stacked[-1].clamp(min=1)) for i, key in enumerate(sums)}


def selection_metric(metrics: dict[str, float], names: list[str]) -> float:
    """Mean of the per-dataset agent ADEs, so a large dataset does not drown small ones (decision 6)."""
    values = [metrics[f"{name}/agent_ade"] for name in names if f"{name}/agent_ade" in metrics]
    return float(np.mean(values)) if values else metrics["agent_ade"]


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
        "graph": {"knn": args.knn, "radius": args.radius},
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


def _scaled_lr(base: float, args) -> float:
    if args.learning_rate is not None:
        return args.learning_rate
    factor = math.sqrt(args.world_size * args.max_nodes / REFERENCE_NODES)
    return min(base * factor, max(base, args.max_learning_rate))


def _scheduler(optimizer, args, steps_per_epoch: int):
    """Linear warm-up, then cosine decay to 2% of the peak LR."""
    warmup = max(1, int(args.warmup_epochs * steps_per_epoch))
    total = max(warmup + 1, args.recipe["max_epochs"] * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / (total - warmup))
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


MAX_NONFINITE_STREAK = 100   # consecutive non-finite gradient steps before a run is stopped


def _build_model(args, payload):
    if args.phase == "ssl":
        model = StarsMotionVAE(args.obs, args.encoder_recipe)
        groups = lambda lr: [{"params": model.parameters(), "lr": lr}]
        return model, groups
    model = StarsMultiAgentForecaster(args.obs, args.pred, dt=CANONICAL_DT,
                                      encoder_recipe=args.encoder_recipe, decoder_recipe=args.decoder_recipe)
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
        ratio = args.encoder_lr_ratio or FORECAST_RECIPE["encoder_learning_rate"] / FORECAST_RECIPE["learning_rate"]
        groups = lambda lr: [{"params": model.decoder.parameters(), "lr": lr},
                             {"params": model.encoder.parameters(), "lr": lr * ratio}]
    else:
        groups = lambda lr: [{"params": model.parameters(), "lr": lr}]
    return model, groups


def train(args) -> None:
    forecasting = resolve_forecasting_config(preset=DEFAULT_PRESET)
    args.obs, args.pred = forecasting.window.observation_steps, forecasting.window.prediction_steps
    payload = None
    encoder_overrides = {"hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
                         "use_node_features": False if args.edge_only else None, "dropout": args.dropout}
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
        "representation": args.representation,
        **({"hidden_dim": args.decoder_hidden_dim} if args.decoder_hidden_dim else {}),
        **({"dropout": args.decoder_dropout} if args.decoder_dropout is not None else {}),
    }
    init = None
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if init.get("model") != "stars_ma" or init.get("phase") != "forecast":
            raise ValueError(f"{args.init_checkpoint} is not a multi-agent STARS forecaster checkpoint")
        if init["graph"] != {"knn": args.knn, "radius": args.radius}:
            raise ValueError(f"Graph topology differs from the init checkpoint ({init['graph']})")
        args.encoder_recipe, args.decoder_recipe = dict(init["encoder_recipe"]), dict(init["decoder_recipe"])
    base_recipe = SSL_RECIPE if args.phase == "ssl" else FORECAST_RECIPE
    args.recipe = {**base_recipe, **({"max_epochs": args.epochs} if args.epochs else {}),
                   **({"patience": args.patience} if args.patience is not None else {})}
    args.recipe["learning_rate"] = _scaled_lr(base_recipe["learning_rate"], args)
    if args.weight_decay is not None:
        args.recipe["weight_decay"] = args.weight_decay
    if args.phase == "forecast":
        args.recipe["kl_weight"] = args.kl_weight
        if init is not None:
            args.recipe["kl_warmup_epochs"] = 0     # the latent is already trained: no KL annealing

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    skip = {"device_realized", "run_config", "run", "main", "epoch_callback"}
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
        weights = window_weights(train_store, {entry["name"]: entry["weight"] for entry in args.train_entries})
        counts = {name: int((train_store.dataset == i).sum()) for i, name in enumerate(train_store.names)}
        run.update(stage="datasets_ready", n_train_windows=len(train_store), n_val_windows=len(val_store),
                   n_train_nodes=int(train_store.ptr[-1]), train_windows_per_dataset=counts,
                   val_datasets=val_store.names)

        model, groups = _build_model(args, payload)
        if init is not None:
            model.load_state_dict(init["state_dict"])
            run.log("forecaster weights loaded", checkpoint=str(args.init_checkpoint))
        model = model.to(device)
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
        samples = args.samples if (args.phase == "forecast" and model.latent_dim) else 1
        checkpoint = args.run_dir / "checkpoints" / "best.pt"

        for epoch in range(start_epoch, args.recipe["max_epochs"] + 1):
            model.train()
            if getattr(model, "freeze_encoder", False):
                model.encoder.eval()  # frozen: no dropout, as features of a fixed encoder
            if args.phase == "ssl":
                r = args.recipe
                weight = min(r["beta_max"], r["beta_max"] * epoch / max(1, r["beta_warmup_epochs"]))
            else:
                weight = args.recipe["kl_weight"] * min(1.0, epoch / max(1, args.recipe["kl_warmup_epochs"]))
            batches = epoch_batches(train_store, args, epoch, train=True, weights=weights)
            sums = {}
            nodes = skipped = bad_streak = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for windows in run.progress(batches, desc=f"{args.phase} epoch {epoch}"):
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
                scheduler.step()
                global_step += 1
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + value.detach().float()
                nodes += batch.num_nodes
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            train_metrics = {f"train_{k}": float(_all_mean(v / max(1, len(batches)), args)) for k, v in sums.items()}
            train_metrics["train_skipped_steps"] = skipped
            throughput = {"epoch_seconds": elapsed, "nodes_per_second": nodes * args.world_size / elapsed,
                          "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                          "learning_rate": optimizer.param_groups[0]["lr"]}
            val_batches = epoch_batches(val_store, args, 0, train=False)
            if args.phase == "ssl":
                metrics = evaluate_ssl(model, val_store, val_batches, args, weight)
                selection, name = metrics["loss"], "val_ssl_loss"
            else:
                metrics = evaluate_forecaster(model, val_store, val_batches, args, samples=samples)
                selection, name = selection_metric(metrics, val_store.names), f"val_top{samples}_agent_ADE_macro"
            run.metric({"stage": args.phase, "epoch": epoch, "beta" if args.phase == "ssl" else "kl_weight": weight,
                        "selection": selection, **train_metrics, **throughput,
                        **{f"val_{k}": v for k, v in metrics.items()}})
            if getattr(args, "epoch_callback", None) is not None:
                args.epoch_callback(epoch, selection, metrics)  # e.g. Optuna reporting / pruning
            improved, stopped = stopper.update(selection)
            if improved:
                best_epoch = epoch
                if args.main:
                    _save(checkpoint, model, args, selection={"metric": name, "value": selection,
                                                              "best_epoch": epoch, "metrics": metrics})
            if args.main:
                latest.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                            "global_step": global_step,
                            "stopper": {"patience": stopper.patience, "min_delta": stopper.min_delta,
                                        "best": stopper.best, "bad_epochs": stopper.bad_epochs}},
                           latest.with_suffix(".tmp"))
                os.replace(latest.with_suffix(".tmp"), latest)
            run.update(stage=args.phase, epoch=epoch, best_selection=stopper.best, best_epoch=best_epoch,
                       **{key: throughput[key] for key in ("epoch_seconds", "nodes_per_second")})
            if stopped:
                run.log("early stopping", epoch=epoch, best=stopper.best)
                break
        if best_epoch is None:
            raise RuntimeError("No finite validation metric was observed")
        run.update(status="complete", stage="training", best_validation_metric=stopper.best, best_epoch=best_epoch)
        return {"best": stopper.best, "best_epoch": best_epoch}
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
        # Multi-GPU runs on all datasets hit a bus error in interpreter teardown *after* training
        # finished and saved (docs/decisions.md 42), which marked successful jobs FAILED. Everything
        # is on disk by now, so skip the teardown.
        import sys

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sys.stdout.flush(), sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
