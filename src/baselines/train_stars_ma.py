"""Train multi-agent STARS (``stars_ma_model``) on STEPS: self-supervised encoder, then forecaster.

Phase 1, ``--phase ssl``: the extended STARS encoder (node motion + SE(2) edge features at the
native STEPS rate, neighbourhood graphs) is trained with STARS's self-supervised graph-VAE
objective plus node-motion reconstruction, selected on validation loss.

Phases 2-4, ``--phase forecast``: a multi-agent decoder forecasts every pedestrian of each scene
window. ``--encoder-mode`` chooses a frozen or jointly fine-tuned phase-1 encoder
(``--encoder-checkpoint``) or one trained from scratch; ``--decoder cvae`` adds the probabilistic
decoder (phase 3); ``--rollout-exec-steps`` adds a closed-loop multi-step loss (phase 4): the
model is rolled out over the prediction horizon on its own predictions, rebuilding the graph
every round, and the rollout is supervised against ground truth. Checkpoints are selected on
validation best-of-K all-agent ADE.

Ablations map to flags: ``--edge-only`` (STARS's edge-only encoder), ``--fully-connected``
(STARS's topology), ``--encoder-mode`` (frozen / fine-tuned / scratch), ``--decoder``
(deterministic / CVAE) and ``--rollout-exec-steps 0`` vs ``> 0`` (one-shot vs multi-step).
The original single-target STARS probe stays available as ``baselines.train_stars``.
"""
from __future__ import annotations

import argparse
import math
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from baselines.device import training_device
from baselines.early_stopping import EarlyStopping
from baselines.provenance import snapshot_dataset, snapshot_split_inputs
from baselines.stars import REFERENCE, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY, VAE_RECIPE
from baselines.stars_ma_data import StarsMultiAgentDataset
from baselines.stars_ma_graph import to_local, to_scene
from baselines.stars_ma_metrics import MetricAccumulator, forecasting_metrics
from baselines.stars_ma_model import (
    DECODER_RECIPE,
    ENCODER_RECIPE,
    StarsMotionVAE,
    StarsMultiAgentForecaster,
    rollout,
)
from baselines.status import RunStatus, checkpoint_metadata
from baselines.transfer_data import parse_source_datasets, pretraining_datasets
from config import Config, SeedConfig
from data import CANONICAL_DT, seed_worker
from forecasting_config import config_as_dict, resolve_forecasting_config

DEFAULT_PRESET = "steps_default"
# STARS's phase-1 optimisation recipe (stars.VAE_RECIPE) for the encoder.
SSL_RECIPE = {key: VAE_RECIPE[key] for key in
              ("learning_rate", "weight_decay", "grad_clip", "batch_size", "max_epochs", "beta_max",
               "beta_warmup_epochs", "patience", "min_delta")}
FORECAST_RECIPE = {
    "learning_rate": 1e-3,
    "encoder_learning_rate": 1e-4,  # fine-tuning only
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "batch_size": 32,
    "max_epochs": 200,
    "patience": 20,
    "min_delta": 1e-4,
    "kl_weight": 1.0,
    "kl_warmup_epochs": 10,
}


def _loader(dataset, args, batch_size: int, *, shuffle: bool):
    from torch_geometric.loader import DataLoader

    common = {"num_workers": args.num_workers, "worker_init_fn": seed_worker,
              "pin_memory": args.device_realized.type == "cuda"}
    if args.num_workers > 0:
        common["prefetch_factor"] = 4
        common["persistent_workers"] = True
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(args.seed), **common)


def _limited(loader, max_batches):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def _ssl_epoch(model, loader, device, beta, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums, batches = {"loss": 0.0, "edge": 0.0, "node": 0.0, "kl": 0.0}, 0
    with torch.set_grad_enabled(training):
        for batch in _limited(loader, args.max_batches):
            batch = batch.to(device, non_blocking=True)
            losses = model.loss(batch, beta, args.node_loss_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.recipe["grad_clip"])
                optimizer.step()
            for key, value in losses.items():
                sums[key] += float(value.detach())
            batches += 1
    if not batches:
        raise RuntimeError("The loader yielded no batches")
    return {key: value / batches for key, value in sums.items()}


def _masked_distance(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean L2 error over valid (agent, step) pairs."""
    error = torch.linalg.norm(prediction - torch.nan_to_num(target), dim=-1)
    return (error * valid).sum() / valid.sum().clamp(min=1)


def forecast_loss(model, batch, args, kl_weight: float) -> dict[str, torch.Tensor]:
    """Teacher-forced loss, CVAE KL, and (optionally) the closed-loop multi-step rollout loss."""
    future_local = to_local(batch.future, batch.origin, batch.heading)
    valid = batch.future_valid.to(future_local.dtype)
    out = model.training_forward(batch, future_local, batch.future_valid)
    losses = {"forecast": _masked_distance(out["prediction"], future_local, valid)}
    losses["kl"] = out["kl"].sum() / batch.future_valid.any(dim=-1).sum().clamp(min=1) if out["kl"] is not None \
        else losses["forecast"].new_zeros(())
    losses["rollout"] = losses["forecast"].new_zeros(())
    if args.rollout_exec_steps:
        # Closed loop: each agent keeps the intent z inferred from its future for the whole rollout.
        rounds = math.ceil(model.prediction_steps / args.rollout_exec_steps)
        simulated = rollout(model, batch, rounds=rounds, exec_steps=args.rollout_exec_steps, knn=args.knn,
                            radius=args.radius, z=out["z"], keep_graph=True)[:, :model.prediction_steps]
        losses["rollout"] = _masked_distance(simulated, batch.future, valid)
    losses["loss"] = losses["forecast"] + kl_weight * losses["kl"] + args.rollout_weight * losses["rollout"]
    return losses


@torch.no_grad()
def evaluate_forecaster(model, loader, device, *, samples: int, max_batches=None, rollout_exec_steps: int = 0,
                        knn=None, radius=None) -> dict[str, float]:
    """Teacher-forced best-of-K metrics and, with ``rollout_exec_steps``, closed-loop ones (``rollout_*``)."""
    model.eval()
    teacher, closed = MetricAccumulator(), MetricAccumulator()
    for batch in _limited(loader, max_batches):
        batch = batch.to(device, non_blocking=True)
        local = model.predict(batch, samples=samples)
        forecasting_metrics(to_scene(local, batch.origin, batch.heading), batch, teacher)
        if rollout_exec_steps:
            rounds = math.ceil(model.prediction_steps / rollout_exec_steps)
            # Each CVAE rollout draws one intent per agent from the prior; K rollouts give best-of-K.
            runs = samples if model.latent_dim else 1
            simulated = torch.stack([
                rollout(model, batch, rounds=rounds, exec_steps=rollout_exec_steps, knn=knn, radius=radius)
                [:, :model.prediction_steps] for _ in range(runs)
            ])
            forecasting_metrics(simulated, batch, closed)
    result = teacher.result()
    result.update({f"rollout_{key}": value for key, value in closed.result().items()})
    return result


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=("ssl", "forecast"))
    parser.add_argument("--dataset", choices=("steps", "atc", "ptdpes"))
    parser.add_argument("--datasets", nargs="+", help="One or more source datasets; comma-separated values are also accepted")
    parser.add_argument("--pretrain-fraction", type=float, default=1.0)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, help="Integration diagnostic: cap every split's window count")
    parser.add_argument("--max-batches", type=int, help="Integration diagnostic: cap batches per epoch and evaluation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    graph = parser.add_argument_group("graph")
    graph.add_argument("--knn", type=int, default=16, help="Sources kept per destination (default 16)")
    graph.add_argument("--radius", type=float, default=10.0, help="Neighbourhood radius in metres (default 10)")
    graph.add_argument("--fully-connected", action="store_true", help="STARS's topology: ignore --knn/--radius")
    encoder = parser.add_argument_group("encoder")
    encoder.add_argument("--hidden-dim", type=int)
    encoder.add_argument("--latent-dim", type=int)
    encoder.add_argument("--edge-only", action="store_true", help="Ablation: STARS's edge-only encoder (no node motion)")
    encoder.add_argument("--node-loss-weight", type=float, default=1.0, help="ssl: weight of node-motion reconstruction")
    forecast = parser.add_argument_group("forecast")
    forecast.add_argument("--encoder-checkpoint", type=Path, help="A --phase ssl checkpoint (frozen/finetune)")
    forecast.add_argument("--encoder-mode", choices=("frozen", "finetune", "scratch"), default="frozen")
    forecast.add_argument("--decoder", choices=("deterministic", "cvae"), default="deterministic")
    forecast.add_argument("--cvae-latent-dim", type=int, default=32)
    forecast.add_argument("--no-social-layer", action="store_true", help="Decoder without its social message-passing round")
    forecast.add_argument("--representation", choices=("mu", "hidden"), default=DECODER_RECIPE["representation"])
    forecast.add_argument("--kl-weight", type=float, default=FORECAST_RECIPE["kl_weight"])
    forecast.add_argument("--rollout-exec-steps", type=int, default=0,
                          help="> 0 adds the closed-loop multi-step loss, executing this many steps per round")
    forecast.add_argument("--rollout-weight", type=float, default=1.0)
    forecast.add_argument("--samples", type=int, default=20, help="K for best-of-K validation (CVAE)")
    forecast.add_argument("--learning-rate", type=float)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    args.source_datasets = parse_source_datasets(args.dataset, args.datasets)
    if not args.source_datasets:
        parser.error("Give --dataset or --datasets")
    if args.fully_connected:
        args.knn = args.radius = None
    if args.phase == "forecast":
        if args.encoder_mode != "scratch" and not args.encoder_checkpoint:
            parser.error(f"--encoder-mode {args.encoder_mode} needs --encoder-checkpoint")
        if args.encoder_mode == "scratch" and args.encoder_checkpoint:
            parser.error("--encoder-mode scratch does not take --encoder-checkpoint")
        if args.rollout_exec_steps < 0:
            parser.error("--rollout-exec-steps must be >= 0")
    elif args.encoder_checkpoint or args.rollout_exec_steps:
        parser.error("--encoder-checkpoint/--rollout-exec-steps apply to --phase forecast")

    args.device_realized = training_device(args.device)
    forecasting = resolve_forecasting_config(preset=DEFAULT_PRESET)
    args.obs, args.pred = forecasting.window.observation_steps, forecasting.window.prediction_steps

    payload = None
    encoder_overrides = {"hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
                         "use_node_features": False if args.edge_only else None}
    args.encoder_recipe = {**ENCODER_RECIPE, **{k: v for k, v in encoder_overrides.items() if v is not None}}
    if args.encoder_checkpoint:
        payload = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("model") != "stars_ma" or payload.get("phase") != "ssl":
            parser.error(f"{args.encoder_checkpoint} is not a multi-agent STARS --phase ssl checkpoint")
        args.encoder_recipe = dict(payload["encoder_recipe"])
        if payload["graph"] != {"knn": args.knn, "radius": args.radius}:
            parser.error(f"Graph topology differs from the encoder checkpoint ({payload['graph']})")
    args.decoder_recipe = {
        **DECODER_RECIPE,
        "latent_dim": args.cvae_latent_dim if args.decoder == "cvae" else 0,
        "social_layer": not args.no_social_layer,
        "representation": args.representation,
    }
    base_recipe = SSL_RECIPE if args.phase == "ssl" else FORECAST_RECIPE
    overrides = {"max_epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate}
    args.recipe = {**base_recipe, **{k: v for k, v in overrides.items() if v is not None}}
    if args.phase == "forecast":
        args.recipe["kl_weight"] = args.kl_weight

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.run_config = {
        **{key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()
           if key not in {"device_realized", "run_config"}},
        "model": "stars_ma",
        "source_datasets": list(args.source_datasets),
        "device_realized": str(args.device_realized),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": str(REFERENCE.relative_to(Path(__file__).resolve().parents[2])),
        **forecasting.flat_provenance(),
        "forecasting_config": config_as_dict(forecasting),
    }
    run = RunStatus(args.run_dir, args.run_config)
    try:
        run.update(status="running", stage="loading_datasets")
        cfg = Config(seeds=SeedConfig(args.data_seed, args.seed), forecasting=forecasting)
        train_source, val_source = pretraining_datasets(args.source_datasets, cfg, args.pretrain_fraction)
        dataset = dict(dt=CANONICAL_DT, knn=args.knn, radius=args.radius, windows="unique", limit=args.max_scenes)
        train = StarsMultiAgentDataset(train_source, args.obs, args.pred, **dataset)
        val = StarsMultiAgentDataset(val_source, args.obs, args.pred, **dataset)
        if not len(train) or not len(val):
            raise RuntimeError("Training or validation population is empty")
        run.update(stage="saving_dataset_manifests", n_train_windows=len(train), n_val_windows=len(val))
        snapshot_split_inputs(args.run_dir)
        snapshot_dataset(args.run_dir, "train", train_source)
        snapshot_dataset(args.run_dir, "val", val_source)
        device, recipe = args.device_realized, args.recipe
        train_loader = _loader(train, args, recipe["batch_size"], shuffle=True)
        val_loader = _loader(val, args, recipe["batch_size"], shuffle=False)
        checkpoint = args.run_dir / "checkpoints" / "best.pt"

        if args.phase == "ssl":
            model = StarsMotionVAE(args.obs, args.encoder_recipe).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
            stopper = EarlyStopping(recipe["patience"], recipe["min_delta"])
            best_epoch = None
            run.update(stage="ssl", parameters=sum(p.numel() for p in model.parameters()))
            for epoch in range(1, recipe["max_epochs"] + 1):
                beta = min(recipe["beta_max"], recipe["beta_max"] * epoch / max(1, recipe["beta_warmup_epochs"]))
                train_losses = _ssl_epoch(model, run.progress(train_loader, desc=f"SSL epoch {epoch}"), device, beta, args, optimizer)
                val_losses = _ssl_epoch(model, val_loader, device, beta, args)
                run.metric({"stage": "ssl", "epoch": epoch, "beta": beta,
                            **{f"train_{k}": v for k, v in train_losses.items()},
                            **{f"val_{k}": v for k, v in val_losses.items()}})
                improved, stopped = stopper.update(val_losses["loss"])
                if improved:
                    best_epoch = epoch
                    _save(checkpoint, model, args, selection={"metric": "val_ssl_loss", "value": stopper.best,
                                                              "best_epoch": epoch})
                run.update(stage="ssl", epoch=epoch, best_val_loss=stopper.best, best_epoch=best_epoch)
                if stopped:
                    run.log("SSL early stopping", epoch=epoch, best_val_loss=stopper.best)
                    break
            if best_epoch is None:
                raise RuntimeError("No finite validation loss was observed")
            run.update(status="complete", stage="training", best_validation_metric=stopper.best,
                       checkpoint_metric="val_ssl_loss")
            return

        model = StarsMultiAgentForecaster(args.obs, args.pred, dt=CANONICAL_DT,
                                          encoder_recipe=args.encoder_recipe, decoder_recipe=args.decoder_recipe)
        if payload is not None:
            encoder_state = {key.removeprefix("encoder."): value for key, value in payload["state_dict"].items()
                             if key.startswith("encoder.")}
            model.encoder.load_state_dict(encoder_state)
            run.log("phase-1 encoder loaded", checkpoint=str(args.encoder_checkpoint), mode=args.encoder_mode)
        model.freeze_encoder = args.encoder_mode == "frozen"
        model = model.to(device)
        if model.freeze_encoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)
            groups = [{"params": model.decoder.parameters(), "lr": recipe["learning_rate"]}]
        elif args.encoder_mode == "finetune":
            groups = [{"params": model.decoder.parameters(), "lr": recipe["learning_rate"]},
                      {"params": model.encoder.parameters(), "lr": recipe["encoder_learning_rate"]}]
        else:
            groups = [{"params": model.parameters(), "lr": recipe["learning_rate"]}]
        optimizer = torch.optim.AdamW(groups, weight_decay=recipe["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=max(1, recipe["patience"] // 3))
        stopper = EarlyStopping(recipe["patience"], recipe["min_delta"])
        samples = args.samples if model.latent_dim else 1
        best_epoch = None
        run.update(stage="forecast", parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        for epoch in range(1, recipe["max_epochs"] + 1):
            model.train()
            if model.freeze_encoder:
                model.encoder.eval()  # frozen: no dropout, as features of a fixed encoder
            kl_weight = recipe["kl_weight"] * min(1.0, epoch / max(1, recipe["kl_warmup_epochs"]))
            sums, batches = {}, 0
            for batch in _limited(run.progress(train_loader, desc=f"forecast epoch {epoch}"), args.max_batches):
                batch = batch.to(device, non_blocking=True)
                losses = forecast_loss(model, batch, args, kl_weight)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], recipe["grad_clip"])
                optimizer.step()
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                batches += 1
            metrics = evaluate_forecaster(model, val_loader, device, samples=samples, max_batches=args.max_batches,
                                          rollout_exec_steps=args.rollout_exec_steps, knn=args.knn, radius=args.radius)
            selection = metrics["agent_ade"]
            scheduler.step(selection)
            run.metric({"stage": "forecast", "epoch": epoch, "kl_weight": kl_weight,
                        **{f"train_{k}": v / max(1, batches) for k, v in sums.items()},
                        **{f"val_{k}": v for k, v in metrics.items()}})
            improved, stopped = stopper.update(selection)
            if improved:
                best_epoch = epoch
                _save(checkpoint, model, args, selection={"metric": f"val_top{samples}_agent_ADE", "value": selection,
                                                          "best_epoch": epoch, "metrics": metrics})
            run.update(stage="forecast", epoch=epoch, best_val_agent_ade=stopper.best, best_epoch=best_epoch)
            if stopped:
                run.log("forecast early stopping", epoch=epoch, best=stopper.best)
                break
        if best_epoch is None:
            raise RuntimeError("No finite validation ADE was observed")
        run.update(status="complete", stage="training", best_validation_metric=stopper.best,
                   checkpoint_metric=f"val_top{samples}_agent_ADE")
    except Exception as error:
        run.log("run failed", error=repr(error), traceback=traceback.format_exc())
        run.update(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
