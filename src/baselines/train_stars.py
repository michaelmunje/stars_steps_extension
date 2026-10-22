"""Pretrain, train from scratch, or fine-tune STARS as a forecaster on the shared benchmark samples.

Two phases, as STARS's own forecasting probe:

1. VAE: the pinned ``HGNNCombinedVAE`` is trained self-supervised on the observation-window
   graphs of the training split (STARS ``homo`` recipe). Skipped by ``--stage finetune``,
   which keeps the pretrained encoder.
2. Head: with the encoder frozen, latent features are computed once per split and a linear
   head is fitted to all prediction steps (STARS probe recipe), selected on validation ADE.

``pretrain`` uses the source datasets; ``scratch`` and ``finetune`` use the target split.
Predictions are deterministic (K=1); ``--latent-samples`` adds a best-of-K validation report
from posterior samples without changing training or selection.
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from baselines.device import training_device
from baselines.early_stopping import EarlyStopping
from baselines.preprocessing_options import (
    add_preprocessing_arguments,
    config_preprocessing,
    handle_preprocessing_introspection,
)
from baselines.provenance import snapshot_dataset, snapshot_split_inputs
from baselines.registry import resolve
from baselines.stars import (
    HEAD_RECIPE,
    REFERENCE,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    VAE_RECIPE,
    ForecastHead,
    DEFAULT_MAX_EDGES,
    StarsDataset,
    best_of_k,
    build_vae,
    edge_chunks,
    pooled_latent,
    vae_loss,
    vae_loss_normalizers,
)
from baselines.status import RunStatus, checkpoint_metadata
from baselines.transfer_data import (
    downstream_datasets,
    parse_source_datasets,
    pretraining_datasets,
    resolve_target_name,
)
from config import Config, SeedConfig
from data import seed_worker
from forecasting_config import config_as_dict

DEFAULT_PRESET = "steps_default"


def _sources(args, cfg: Config):
    if args.stage == "pretrain":
        train, val = pretraining_datasets(args.source_datasets, cfg, args.pretrain_fraction)
        return train, val, None
    train, val, _test, manifest = downstream_datasets(
        args.target_dataset, args.fold, cfg,
        validation_mode=args.validation_mode, target_fraction=args.target_fraction,
    )
    return train, val, manifest


def _loader(dataset, args, batch_size: int, *, shuffle: bool):
    from torch_geometric.loader import DataLoader

    common = {"num_workers": args.num_workers, "worker_init_fn": seed_worker,
              "pin_memory": args.device_realized.type == "cuda"}
    if args.num_workers > 0:
        common["prefetch_factor"] = 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(args.seed), **common)


def _vae_epoch(model, loader, device, beta, latent_dim, *, max_edges, optimizer=None, grad_clip=None,
               max_batches=None, run=None, desc=""):
    """One STARS VAE epoch. Each batch is split by edge count and its gradients accumulated with
    whole-batch denominators, so the optimised loss is exactly STARS's full-batch loss."""
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "edge": 0.0, "kl": 0.0}
    batches = 0
    iterable = run.progress(loader, desc=desc, total=min(len(loader), max_batches or len(loader))) if run else loader
    with torch.set_grad_enabled(training):
        for index, batch in enumerate(iterable):
            if max_batches is not None and index >= max_batches:
                break
            chunks = [chunk.to(device, non_blocking=True) for chunk in edge_chunks(batch, max_edges)]
            edge_entries, latent_entries = vae_loss_normalizers(chunks, latent_dim)
            if training:
                optimizer.zero_grad(set_to_none=True)
            totals = {"loss": 0.0, "edge": 0.0, "kl": 0.0}
            for chunk in chunks:
                losses = vae_loss(model, chunk, beta, edge_entries, latent_entries)
                if training:
                    losses["loss"].backward()
                for key, value in losses.items():
                    totals[key] += float(value.detach())
            if training:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key, value in totals.items():
                sums[key] += value
            batches += 1
    if not batches:
        raise RuntimeError("STARS VAE loader yielded no batches")
    return {key: value / batches for key, value in sums.items()}


@torch.no_grad()
def _features(model, head, loader, device, *, max_edges, max_batches=None, run=None, desc=""):
    """Frozen-encoder head inputs and model-frame targets for a whole split."""
    model.eval()
    features, targets = [], []
    iterable = run.progress(loader, desc=desc, total=min(len(loader), max_batches or len(loader))) if run else loader
    for index, batch in enumerate(iterable):
        if max_batches is not None and index >= max_batches:
            break
        for chunk in edge_chunks(batch, max_edges):
            chunk = chunk.to(device, non_blocking=True)
            features.append(head.features(pooled_latent(model, chunk), chunk.history).cpu())
            targets.append(chunk.future.cpu())
    return torch.cat(features), torch.cat(targets)


def _ade_fde(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    # Rigid transforms preserve distance, so model-frame ADE/FDE equal the world-frame values.
    distance = torch.linalg.norm(prediction - target, dim=-1)
    return float(distance.mean()), float(distance[:, -1].mean())


def _fit_head(head, train_xy, val_xy, args, device, run):
    recipe = args.head_recipe
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=recipe["lr_patience"])
    stopper = EarlyStopping(recipe["patience"], recipe["min_delta"])
    generator = torch.Generator().manual_seed(args.seed)
    train_x, train_y = (tensor.to(device) for tensor in train_xy)
    val_x, val_y = (tensor.to(device) for tensor in val_xy)
    best_state, best_epoch, best = None, None, None
    for epoch in range(1, recipe["max_epochs"] + 1):
        head.train()
        order = torch.randperm(len(train_x), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), recipe["batch_size"]):
            index = order[start:start + recipe["batch_size"]]
            loss = torch.nn.functional.mse_loss(head(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        head.eval()
        with torch.no_grad():
            prediction = head(val_x)
            val_loss = float(torch.nn.functional.mse_loss(prediction, val_y))
            ade, fde = _ade_fde(prediction, val_y)
        scheduler.step(val_loss)
        run.metric({"stage": "head_val", "epoch": epoch, "train_mse": total / len(order), "val_mse": val_loss,
                    "top1_ade": ade, "top1_fde": fde, "lr": optimizer.param_groups[0]["lr"]})
        improved, stopped = stopper.update(ade)
        if improved:
            best_state, best_epoch, best = copy.deepcopy(head.state_dict()), epoch, {"top1_ade": ade, "top1_fde": fde}
        run.update(stage="head", head_epoch=epoch, head_best_top1_ade=stopper.best, head_best_epoch=best_epoch)
        if stopped:
            run.log("head early stopping", epoch=epoch, best_top1_ade=stopper.best)
            break
    if best_state is None:
        raise RuntimeError("No finite validation ADE was observed for the STARS head")
    head.load_state_dict(best_state)
    return head, best_epoch, best


def _save(path: Path, model, head, args, *, vae_selection, head_selection, latent_report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": "stars",
        "vae": model.state_dict(),
        "head": head.state_dict(),
        "stage": args.stage,
        "vae_recipe": args.vae_recipe,
        "head_recipe": args.head_recipe,
        "architecture": args.architecture,
        "vae_selection": vae_selection,
        "checkpoint_selection": {"metric": "val_top1_ADE", "value": head_selection["top1_ade"],
                                 "best_epoch": head_selection["best_epoch"]},
        "latent_sampling_report": latent_report,
        "run_config": args.run_config,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        **checkpoint_metadata(path.parents[1]),
    }, temporary)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("pretrain", "scratch", "finetune"))
    parser.add_argument("--fold", choices=("eth", "hotel", "univ", "zara1", "zara2"))
    parser.add_argument("--target-dataset", choices=("eth_ucy", "grand_central", "madras"))
    parser.add_argument("--target-fraction", type=float, default=1.0)
    parser.add_argument("--validation-mode", choices=("none", "grouped"))
    parser.add_argument("--pretrain-dataset", choices=("atc", "ptdpes", "steps"))
    parser.add_argument("--pretrain-datasets", nargs="+", help="One or more source datasets; comma-separated values are also accepted")
    parser.add_argument("--pretrain-fraction", type=float, default=1.0)
    parser.add_argument("--checkpoint", help="finetune: a STARS pretrain checkpoint or registered model ID")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, help="Integration diagnostic: cap every split's sample count")
    parser.add_argument("--max-batches", type=int, help="Integration diagnostic: cap batches per VAE epoch and per feature pass")
    parser.add_argument("--vae-epochs", type=int, help=f"default {VAE_RECIPE['max_epochs']} (early stopping, patience {VAE_RECIPE['patience']})")
    parser.add_argument("--vae-batch-size", type=int, help=f"default {VAE_RECIPE['batch_size']}")
    parser.add_argument("--head-epochs", type=int, help=f"default {HEAD_RECIPE['max_epochs']}")
    parser.add_argument("--hidden-dim", type=int, help=f"default {VAE_RECIPE['hidden_dim']}")
    parser.add_argument("--latent-dim", type=int, help=f"default {VAE_RECIPE['latent_dim']}")
    parser.add_argument("--no-history", action="store_true", help="Head reads the STARS latent only, not the target's observed history")
    parser.add_argument("--latent-samples", type=int, default=1, help="K>1 also reports best-of-K validation ADE/FDE from posterior latent samples")
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES, help="Edge budget per forward pass; bounds memory, does not change the loss")
    add_preprocessing_arguments(parser)
    return parser


def main() -> None:
    handle_preprocessing_introspection(default_preset=DEFAULT_PRESET)
    parser = _parser()
    args = parser.parse_args()
    args.source_datasets = parse_source_datasets(args.pretrain_dataset, args.pretrain_datasets)
    if args.stage == "pretrain":
        if args.target_dataset or args.fold:
            parser.error("--target-dataset/--fold are only valid for scratch or fine-tuning")
        if not args.source_datasets:
            parser.error("--stage pretrain needs --pretrain-dataset or --pretrain-datasets")
        args.target_dataset = None
    else:
        try:
            args.target_dataset = resolve_target_name(args.target_dataset, args.fold)
        except ValueError as error:
            parser.error(str(error))
    if args.stage == "finetune" and not args.checkpoint:
        parser.error("--checkpoint is required for finetune")
    if args.stage != "finetune" and args.checkpoint:
        parser.error("--checkpoint applies only to --stage finetune")
    if args.target_dataset == "eth_ucy":
        args.validation_mode = args.validation_mode or "grouped"
    elif args.validation_mode is not None:
        parser.error("--validation-mode applies only to --target-dataset eth_ucy")
    if not 0.0 < args.target_fraction <= 1.0 or not 0.0 < args.pretrain_fraction <= 1.0:
        parser.error("--target-fraction and --pretrain-fraction must be in (0, 1]")
    if args.latent_samples < 1 or args.max_edges < 1:
        parser.error("--latent-samples and --max-edges must be positive")

    args.device_realized = training_device(args.device)
    args.forecasting = config_preprocessing(args, default_preset=DEFAULT_PRESET)["forecasting"]
    obs, pred = args.forecasting.window.observation_steps, args.forecasting.window.prediction_steps

    payload = None
    if args.stage == "finetune":
        checkpoint = resolve(args.checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("model") != "stars" or payload.get("stage") != "pretrain":
            parser.error(f"{checkpoint} is not a STARS pretrain checkpoint")
        trained_hash = payload["run_config"]["forecast_preprocessing_hash"]
        if trained_hash != args.forecasting.forecast_preprocessing_hash:
            parser.error(f"Preprocessing differs from the pretrained checkpoint ({trained_hash})")
        args.vae_recipe = dict(payload["vae_recipe"])
        args.architecture = dict(payload["architecture"])
    else:
        checkpoint = None
        overrides = {"max_epochs": args.vae_epochs, "batch_size": args.vae_batch_size,
                     "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim}
        args.vae_recipe = {**VAE_RECIPE, **{key: value for key, value in overrides.items() if value is not None}}
        args.architecture = {"observation_steps": obs, "prediction_steps": pred,
                             "latent_dim": args.vae_recipe["latent_dim"], "use_history": not args.no_history}
    args.head_recipe = {**HEAD_RECIPE, **({"max_epochs": args.head_epochs} if args.head_epochs else {})}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.run_config = {
        **{key: value for key, value in vars(args).items() if key not in {"forecasting", "device_realized", "run_config"}},
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model": "stars",
        "k": args.latent_samples,
        "source_datasets": list(args.source_datasets) if args.stage == "pretrain"
        else list((payload or {}).get("run_config", {}).get("source_datasets", [])),
        "pretraining_checkpoint": str(checkpoint) if checkpoint else None,
        "pretraining_seed": args.seed if args.stage == "pretrain" else (payload or {}).get("run_config", {}).get("seed"),
        "finetuning_seed": args.seed if args.stage != "pretrain" else None,
        "device_realized": str(args.device_realized),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": str(REFERENCE.relative_to(Path(__file__).resolve().parents[2])),
        **args.forecasting.flat_provenance(),
        "forecasting_config": config_as_dict(args.forecasting),
    }
    run = RunStatus(args.run_dir, args.run_config)
    try:
        run.update(status="running", stage="loading_datasets")
        cfg = Config(seeds=SeedConfig(args.data_seed, args.seed), forecasting=args.forecasting)
        train_source, val_source, fraction_manifest = _sources(args, cfg)
        if val_source is None:
            raise RuntimeError("STARS selects the head on validation ADE; a validation split is required")
        train = StarsDataset(train_source, obs, limit=args.max_scenes)
        val = StarsDataset(val_source, obs, limit=args.max_scenes)
        if not len(train) or not len(val):
            raise RuntimeError("STARS training or validation population is empty")
        run.update(stage="saving_dataset_manifests", n_train=len(train), n_val=len(val))
        snapshot_split_inputs(args.run_dir)
        snapshot_dataset(args.run_dir, "train", train)
        snapshot_dataset(args.run_dir, "val", val)
        if fraction_manifest is not None:
            destination = args.run_dir / "datasets" / "target_fraction_manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fraction_manifest, destination)

        device = args.device_realized
        model = build_vae(device, args.vae_recipe)
        head = ForecastHead(args.architecture["latent_dim"], obs, pred, use_history=args.architecture["use_history"])
        if payload is not None:
            model.load_state_dict(payload["vae"])
            head.load_state_dict(payload["head"])
            vae_selection = {**payload["vae_selection"], "source": "pretrained checkpoint"}
            run.log("pretrained STARS encoder and head loaded; encoder frozen", checkpoint=str(checkpoint))
        else:
            recipe = args.vae_recipe
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
            stopper = EarlyStopping(recipe["patience"], recipe["min_delta"])
            train_loader = _loader(train, args, recipe["batch_size"], shuffle=True)
            val_loader = _loader(val, args, recipe["batch_size"], shuffle=False)
            best_state, best_epoch = None, None
            run.update(stage="vae", parameters=sum(p.numel() for p in model.parameters()))
            for epoch in range(1, recipe["max_epochs"] + 1):
                beta = min(recipe["beta_max"], recipe["beta_max"] * epoch / max(1, recipe["beta_warmup_epochs"]))
                train_losses = _vae_epoch(model, train_loader, device, beta, recipe["latent_dim"], max_edges=args.max_edges,
                                          optimizer=optimizer, grad_clip=recipe["grad_clip"], max_batches=args.max_batches,
                                          run=run, desc=f"STARS VAE epoch {epoch}")
                val_losses = _vae_epoch(model, val_loader, device, beta, recipe["latent_dim"], max_edges=args.max_edges,
                                        max_batches=args.max_batches)
                run.metric({"stage": "vae", "epoch": epoch, "beta": beta,
                            **{f"train_{k}": v for k, v in train_losses.items()},
                            **{f"val_{k}": v for k, v in val_losses.items()}})
                improved, stopped = stopper.update(val_losses["loss"])
                if improved:
                    best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
                run.update(stage="vae", vae_epoch=epoch, vae_best_val_loss=stopper.best, vae_best_epoch=best_epoch)
                if stopped:
                    run.log("VAE early stopping", epoch=epoch, best_val_loss=stopper.best)
                    break
            if best_state is None:
                raise RuntimeError("No finite STARS VAE validation loss was observed")
            model.load_state_dict(best_state)
            vae_selection = {"metric": "val_vae_loss", "value": stopper.best, "best_epoch": best_epoch}
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        run.update(stage="head_features")
        feature_batch = 4 * args.vae_recipe["batch_size"]
        train_xy = _features(model, head, _loader(train, args, feature_batch, shuffle=False), device,
                             max_edges=args.max_edges, max_batches=args.max_batches, run=run, desc="STARS train features")
        val_loader = _loader(val, args, feature_batch, shuffle=False)
        val_xy = _features(model, head, val_loader, device, max_edges=args.max_edges, max_batches=args.max_batches,
                           run=run, desc="STARS val features")
        head, head_epoch, head_best = _fit_head(head, train_xy, val_xy, args, device, run)
        head_selection = {**head_best, "best_epoch": head_epoch}

        latent_report = None
        if args.latent_samples > 1:
            model.eval()
            ades, fdes = [], []
            for index, batch in enumerate(val_loader):
                if args.max_batches is not None and index >= args.max_batches:
                    break
                ade, fde, _world = best_of_k(model, head, batch.to(device), latent_samples=args.latent_samples,
                                             max_edges=args.max_edges)
                ades.extend(ade.tolist())
                fdes.extend(fde.tolist())
            latent_report = {"k": args.latent_samples, f"top{args.latent_samples}_ade": float(np.mean(ades)),
                             f"top{args.latent_samples}_fde": float(np.mean(fdes)), "n_predictions": len(ades)}
            run.metric({"stage": "val_latent_sampling", **latent_report})
        _save(args.run_dir / "checkpoints/best.pt", model, head, args,
              vae_selection=vae_selection, head_selection=head_selection, latent_report=latent_report)
        run.update(status="complete", stage="training", best_validation_metric=head_best["top1_ade"],
                   checkpoint_metric="val_top1_ADE", vae_selection=vae_selection, head_selection=head_selection,
                   latent_sampling_report=latent_report)
    except Exception as error:
        run.log("run failed", error=repr(error), traceback=traceback.format_exc())
        run.update(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
