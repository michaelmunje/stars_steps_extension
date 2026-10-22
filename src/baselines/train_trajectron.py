from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path

import dill
import numpy as np
import torch
import zstandard as zstd
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader

from baselines.status import RunStatus,checkpoint_metadata
from baselines.early_stopping import EarlyStopping
from baselines.device import training_device
from baselines.provenance import snapshot_environment,snapshot_split_inputs,snapshot_trajectron_shards
from baselines.registry import resolve
from baselines.resume import (
    add_resume_argument,
    capture_rng_state,
    checkpoint_resume_fields,
    load_resume_state,
    resolve_resume_checkpoint,
    restore_optimizer_scheduler_state,
    restore_rng_state,
    resume_guard_config,
)
from baselines.dataset_specs import pretrain_spec
from baselines.trajectron_data import REFERENCE, ROOT, environment, fold_environment, reference_hyperparams, validate_trajectron_preprocessing
from baselines.trajectron_cache import ShardedNodeDataset,_normalize_types,load_shard,prepare_shards,recording_names
from baselines.preprocessing_options import add_preprocessing_arguments, handle_preprocessing_introspection, resolved_forecasting_from_args
from forecasting_config import config_as_dict

from model.dataset import EnvironmentDataset, collate
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron

UPSTREAM_COMMIT = "1031c7bd1a444273af378c1ec1dcca907ba59830"
ENVIRONMENT_CACHE_VERSION = 1
VALIDATION_PIPELINE_VERSION = 2


def validate_trajectron_finetune_checkpoint(
    payload: dict,
    checkpoint_path: Path,
    *,
    expected_hash: str | None,
    expected_commit: str,
) -> None:
    """Refuse an incompatible checkpoint before ``--stage finetune`` loads its weights.

    Guards against silently resuming from a checkpoint produced by a different
    preprocessing recipe or a different pinned upstream commit, and requires a
    selected (not merely latest) pretraining checkpoint with a recorded
    validation metric.
    """
    if checkpoint_path.name != "best.pt":
        raise ValueError(
            f"fine-tuning requires the pretrained checkpoints/best.pt, got {checkpoint_path.name!r}"
        )
    if payload.get("stage") != "pretrain":
        raise ValueError(
            f"fine-tuning source must be a pretraining checkpoint, got stage={payload.get('stage')!r}"
        )
    actual_hash = payload.get("run_config", {}).get("forecast_preprocessing_hash")
    if actual_hash != expected_hash:
        raise ValueError(
            f"checkpoint forecast_preprocessing_hash {actual_hash!r} does not match "
            f"this run's forecast_preprocessing_hash {expected_hash!r}"
        )
    actual_commit = payload.get("upstream_commit")
    if actual_commit != expected_commit:
        raise ValueError(
            f"checkpoint upstream_commit {actual_commit!r} does not match the pinned "
            f"upstream_commit {expected_commit!r}"
        )
    selection = payload.get("checkpoint_selection") or {}
    if selection.get("value") is None:
        raise ValueError(
            "pretrained best checkpoint has no source validation metric "
            "(checkpoint_selection.value is None); pass --allow-legacy-checkpoint "
            "only for an explicit compatibility run"
        )


def resolve_trajectron_seed(explicit_seed: int | None, stage: str, checkpoint_payload: dict | None) -> int:
    """Resolve the training seed per the finetune-inherits-pretrain-seed policy.

    Explicit ``--seed`` always wins. For ``--stage finetune`` with a checkpoint
    and no explicit seed, inherit the pretraining run's seed so fine-tuning
    reproduces the same stochastic continuation as its source run. Otherwise
    (pretrain/scratch, or finetune without a checkpoint) default to 0.
    """
    if explicit_seed is not None:
        return explicit_seed
    if stage == "finetune" and checkpoint_payload is not None:
        return checkpoint_payload.get("run_config", {})["seed"]
    return 0


def _environment_cache_path(args, hp) -> Path:
    if args.stage == "pretrain":
        roots=[pretrain_spec(args.pretrain_dataset).path]
    else:
        roots=[ROOT/"datasets/eth",ROOT/"datasets/ucy"]
    files=[]
    for root in roots:
        index=root/"indices"/f"scenes-seed{args.data_seed}.parquet"
        if index.exists(): files.append((str(index.relative_to(ROOT)),index.stat().st_size,index.stat().st_mtime_ns))
        for path in sorted((root/"converted").glob("*.traj.parquet")):
            stat=path.stat();files.append((str(path.relative_to(ROOT)),stat.st_size,stat.st_mtime_ns))
    protocol=ROOT/"configs/protocols/eth_ucy_five_fold.json"
    payload={"version":ENVIRONMENT_CACHE_VERSION,"upstream_commit":UPSTREAM_COMMIT,"stage":args.stage,
             "pretrain_dataset":args.pretrain_dataset,"fold":args.fold,"data_seed":args.data_seed,
             "validation_mode":args.validation_mode,
             "max_recordings":args.max_recordings,"edge_addition_filter":hp["edge_addition_filter"],
             "edge_removal_filter":hp["edge_removal_filter"],"files":files,
             "forecast_preprocessing_hash":hp.get("steps_forecast_preprocessing_hash"),
             "protocol":hashlib.sha256(protocol.read_bytes()).hexdigest() if protocol.exists() else None}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:20]
    label=args.pretrain_dataset if args.stage=="pretrain" else f"eth_ucy-{args.fold}-{args.stage}"
    return ROOT/"datasets/.cache/trajectron"/f"{label}-seed{args.data_seed}-{digest}.dill.zst"


def _build_environments(args, hp, run):
    if args.stage == "pretrain":
        source = pretrain_spec(args.pretrain_dataset).path
        names = {path.name.removesuffix(".traj.parquet") for path in (source / "converted").glob("*.traj.parquet")}
        train_env = environment({source: names},"train",max_recordings=args.max_recordings,data_seed=args.data_seed)
        val_env = environment({source: names},"val",max_recordings=args.max_recordings,data_seed=args.data_seed)
    else:
        train_env = fold_environment(args.fold,"train",max_recordings=args.max_recordings,data_seed=args.data_seed,validation_mode=args.validation_mode)
        val_env = fold_environment(args.fold,"val",max_recordings=args.max_recordings,data_seed=args.data_seed,validation_mode=args.validation_mode)
    if train_env is None or not train_env.scenes: raise RuntimeError("No scenes selected")
    if args.stage!="pretrain" and (val_env is None or not val_env.scenes):
        raise RuntimeError(f"No validation population for --stage {args.stage} (fold {args.fold!r}, validation_mode {args.validation_mode!r}); checkpoint selection requires a non-empty target validation split")
    for role,env in (("train",train_env),("val",val_env)):
        if env is None:continue
        for scene in run.progress(env.scenes,desc=f"building {role} scene graphs",unit="scene"):
            scene.calculate_scene_graph(env.attention_radius,hp["edge_addition_filter"],hp["edge_removal_filter"])
    return train_env,val_env


def prepared_environments(args,hp,run):
    if args.no_environment_cache:return _build_environments(args,hp,run)
    path=_environment_cache_path(args,hp);path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+".lock"):
        if path.exists():
            run.update(status="running",stage="loading_environment_cache",environment_cache=str(path),cache_hit=True)
            with path.open("rb") as compressed:
                with zstd.ZstdDecompressor().stream_reader(compressed) as stream:train_env,val_env=dill.load(stream)
            run.log("environment cache loaded",path=str(path),size_bytes=path.stat().st_size)
            return train_env,val_env
        run.update(status="running",stage="building_environment_cache",environment_cache=str(path),cache_hit=False)
        train_env,val_env=_build_environments(args,hp,run)
        temporary=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
        run.update(status="running",stage="writing_environment_cache",environment_cache=str(path))
        with temporary.open("wb") as compressed:
            with zstd.ZstdCompressor(level=1).stream_writer(compressed) as stream:
                dill.dump((train_env,val_env),stream,protocol=dill.HIGHEST_PROTOCOL)
        os.replace(temporary,path)
        run.log("environment cache written",path=str(path),size_bytes=path.stat().st_size)
        return train_env,val_env


def loader(env, hp, *, shuffle: bool, batch_size: int):
    dataset = EnvironmentDataset(
        env, hp["state"], hp["pred_state"], node_freq_mult=False,
        scene_freq_mult=False, hyperparams=hp,
        min_history_timesteps=hp["minimum_history_length"],
        min_future_timesteps=hp["prediction_horizon"], return_robot=True,
    )
    node_dataset = dataset.node_type_datasets[0]
    return DataLoader(node_dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate), node_dataset.node_type


def save(
    registrar, optimizer, scheduler, run_dir: Path, iteration: int, *, stage: str,
    optimization: dict, epoch: int, stopper, best_epoch: int | None,
    batch_in_epoch: int, epoch_complete: bool, epoch_rng_state: dict,
    best: bool = False,
):
    target = run_dir / "checkpoints" / ("best.pt" if best else "latest.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    torch.save({"upstream_commit": UPSTREAM_COMMIT, "stage": stage,
                "epoch": epoch, "iteration": iteration,
                "models": registrar.model_dict.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "optimization": optimization,
                "validation_pipeline_version": VALIDATION_PIPELINE_VERSION,
                "checkpoint_selection":{"metric":"val_minADE","value":None if stopper.best==float("inf") else stopper.best,"best_epoch":best_epoch},
                **checkpoint_resume_fields(
                    stopper=stopper,batch_in_epoch=batch_in_epoch,
                    epoch_complete=epoch_complete,
                    loader_generator_epoch_state=None,loader_generator_state=None,
                ),
                "epoch_rng_state":epoch_rng_state,
                **checkpoint_metadata(run_dir)}, temp)
    os.replace(temp, target)


def validation_minade(
    model: Trajectron,
    train_env,
    val_env,
    *,
    val_shards=None,
    k: int = 20,
    stride: int = 20,
    max_scenes: int = 10,
    seed: int = 42,
) -> dict[str, float | int]:
    """Evaluate a fixed validation population with best-of-K displacement error.

    The sharded PTDPES path loads only a few validation scene chunks at a time,
    preserving the bounded-memory contract used during training.  Validation RNG
    is isolated and reset to the same seed every epoch, so checkpoint selection
    changes with model weights rather than with a different Monte Carlo draw.
    """
    from baselines.evaluate_trajectron import evaluate

    if max_scenes <= 0:
        raise ValueError("--validation-max-scenes must be positive")
    rng_state = capture_rng_state()
    total_predictions = 0
    total_recordings = 0
    ade_sum = 0.0
    fde_sum = 0.0
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if val_shards is None:
            if val_env is None:
                raise RuntimeError("Validation minADE requested without a validation environment")
            normalized_val_env = _normalize_types(val_env, train_env)
            result = evaluate(model, normalized_val_env, k=k, stride=stride, max_scenes=max_scenes)
            return {
                "minade": float(result["topk_ade"]),
                "minfde": float(result["topk_fde"]),
                "n_predictions": int(result["n_predictions"]),
                "n_recordings": int(result["n_recordings"]),
            }

        remaining = max_scenes
        for shard in val_shards:
            if remaining <= 0:
                break
            shard_env = _normalize_types(load_shard(shard.path), train_env)
            selected = min(remaining, len(shard_env.scenes))
            if selected <= 0:
                continue
            try:
                result = evaluate(model, shard_env, k=k, stride=stride, max_scenes=selected)
            except RuntimeError as error:
                if "evaluation population is empty" not in str(error):
                    raise
            else:
                count = int(result["n_predictions"])
                total_predictions += count
                total_recordings += int(result["n_recordings"])
                ade_sum += count * float(result["topk_ade"])
                fde_sum += count * float(result["topk_fde"])
            remaining -= selected
            del shard_env
            gc.collect()
        if total_predictions == 0:
            raise RuntimeError("Trajectron++ validation minADE population is empty")
        return {
            "minade": ade_sum / total_predictions,
            "minfde": fde_sum / total_predictions,
            "n_predictions": total_predictions,
            "n_recordings": total_recordings,
        }
    finally:
        # ``Trajectron.set_environment`` reconstructs the upstream node-model
        # wrapper and with it the annealing schedules.  Validation scenes share
        # the same standardization and node type as ``train_env``, so predictions
        # can use the existing training wrapper directly.  Keeping it in place
        # preserves the exact annealer state across validation boundaries.
        restore_rng_state(rng_state)


def main():
    parser = argparse.ArgumentParser(description="Thin trainer around the pinned official Trajectron++ model")
    parser.add_argument("--stage", choices=["pretrain", "scratch", "finetune"], required=True)
    parser.add_argument("--fold", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    add_resume_argument(parser)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="Explicit seed; finetune with a checkpoint and no --seed inherits the pretraining run's seed, otherwise defaults to 0")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true", help="Bypass the finetune checkpoint-compatibility guard for an explicit compatibility run")
    parser.add_argument("--data-seed",type=int,default=42)
    parser.add_argument("--pretrain-dataset",choices=["atc","ptdpes","steps"],default="ptdpes")
    parser.add_argument("--status-every", type=int, default=100)
    parser.add_argument("--patience",type=int,default=10,help="Stop after this many validation epochs without improvement; 0 disables")
    parser.add_argument("--min-delta",type=float,default=1e-4,help="Minimum validation improvement for early stopping")
    parser.add_argument("--validation-k",type=int,default=20,help="Samples used for validation minADE/minFDE")
    parser.add_argument("--validation-stride",type=int,default=20,help="Timestep stride inside validation scene chunks")
    parser.add_argument("--validation-max-scenes",type=int,default=10,help="Fixed scene-chunk budget used for checkpoint selection")
    parser.add_argument("--validation-mode",choices=["none","grouped"],default=None,help="ETH/UCY target validation policy for --stage scratch/finetune; defaults to grouped there, mirroring train_moflow.py/train_social_vae.py. --stage pretrain always uses the source dataset's own validation split and ignores this")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--no-environment-cache",action="store_true",help="Rebuild without reading or writing the persistent prepared-environment cache")
    add_preprocessing_arguments(parser)
    handle_preprocessing_introspection(default_preset="trajectron_pp")
    args = parser.parse_args()
    if args.stage != "pretrain" and args.fold is None:
        parser.error("--fold is required for ETH/UCY scratch and finetune")
    # Mirrors train_moflow.py/train_social_vae.py: default grouped validation
    # for target stages, keep pretrain on its own source-dataset val split (N-016).
    if args.stage == "pretrain":
        if args.validation_mode is not None:
            parser.error("--validation-mode does not apply to --stage pretrain; pretrain always uses the source dataset's own validation split")
        args.validation_mode = "none"
    else:
        args.validation_mode = args.validation_mode or "grouped"
    forecasting = resolved_forecasting_from_args(args, default_preset="trajectron_pp")
    validate_trajectron_preprocessing(forecasting)
    checkpoint=resolve(args.checkpoint) if args.checkpoint else None

    device = training_device(args.device)
    # Loaded once, early, on the target device: reused below both to resolve
    # the finetune-inherits-pretrain-seed policy and to validate/load the
    # checkpoint's weights, avoiding reading a potentially large file twice.
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False) if checkpoint else None
    explicit_seed = args.seed
    args.seed = resolve_trajectron_seed(explicit_seed, args.stage, checkpoint_payload)
    if args.stage == "finetune" and checkpoint_payload is not None and explicit_seed is None:
        print(f"seed not given; inheriting pretraining seed {args.seed} from checkpoint {checkpoint}")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    hp = reference_hyperparams()
    hp["batch_size"] = args.batch_size
    hp["augment"] = forecasting.augmentation.random_rotation.enabled
    hp["steps_forecast_preprocessing_hash"] = forecasting.forecast_preprocessing_hash
    optimization={"optimizer":"Adam","learning_rate":hp["learning_rate"],
                  "learning_decay_rate":hp["learning_decay_rate"],
                  "batch_size":args.batch_size,"epochs":args.epochs,
                  "patience":args.patience,"min_delta":args.min_delta,
                  "checkpoint_metric":"val_minADE","validation_k":args.validation_k,
                  "validation_stride":args.validation_stride,
                  "validation_max_scenes":args.validation_max_scenes}
    run_config={**vars(args), "run_dir": str(args.run_dir),
                    "checkpoint":str(checkpoint) if checkpoint else None,"checkpoint_reference":args.checkpoint,
                    "device_realized": str(device), "upstream_commit": UPSTREAM_COMMIT,
                    "optimization":optimization,
                    "minimum_history_length":hp["minimum_history_length"],
                    "maximum_history_length":hp["maximum_history_length"],
                    "pretraining_seed":args.seed if args.stage=="pretrain" else (checkpoint_payload or {}).get("run_config",{}).get("seed"),
                    "finetuning_seed":args.seed if args.stage!="pretrain" else None,
                    **forecasting.flat_provenance(), "forecasting_config": config_as_dict(forecasting),
                    "upstream_config": str(REFERENCE.parent / "experiments/pedestrians/models/eth_vel/config.json")}
    resume_path=resolve_resume_checkpoint(args.run_dir,args.resume)
    resume_state=load_resume_state(
        resume_path,stage=args.stage,optimization=optimization,
        # Inverted guard: every recorded field is compared except the explicit
        # exemptions in resume.py (L-005/L-017), so a new argument is guarded by default.
        expected_run_config=resume_guard_config(run_config,"trajectron"),model_key="models",
    ) if resume_path is not None else None
    run = RunStatus(args.run_dir,run_config,resume=resume_state is not None)
    run.update(status="running", stage="preparing_environment", iteration=resume_state.iteration if resume_state else 0)

    if args.stage=="pretrain":
        source=pretrain_spec(args.pretrain_dataset).path
        train_names=recording_names(source,"train",args.data_seed)
        val_names=recording_names(source,"val",args.data_seed)
        if args.max_recordings is not None:
            train_names=train_names[:args.max_recordings]
            val_names=val_names[:args.max_recordings]
        run.update(
            status="running",stage="preparing_cache_shards",
            n_train_recordings=len(train_names),n_val_recordings=len(val_names),
        )
        train_shards=prepare_shards(source,"train",args.data_seed,hp,run,recordings=train_names)
        val_shards=prepare_shards(source,"val",args.data_seed,hp,run,recordings=val_names)
        if not train_shards or not val_shards:raise RuntimeError("No cached train/validation shards selected")
        train_env=load_shard(train_shards[0].path);val_env=load_shard(val_shards[0].path)
        train_data=ShardedNodeDataset(train_shards,hp,train_env,args.data_seed,True,bool(hp.get("augment",True)))
        val_data=ShardedNodeDataset(val_shards,hp,train_env,args.data_seed,False,False)
        train_loader=DataLoader(train_data,batch_size=args.batch_size,num_workers=0,collate_fn=collate)
        val_loader=DataLoader(val_data,batch_size=args.batch_size,num_workers=0,collate_fn=collate)
        node_type=train_env.NodeType.PEDESTRIAN;n_train_scenes=len(train_shards);n_val_scenes=len(val_shards)
    else:
        train_env,val_env=prepared_environments(args,hp,run)
        train_loader,node_type=loader(train_env,hp,shuffle=True,batch_size=args.batch_size);train_loader.dataset.augment=bool(hp.get("augment",True))
        val_loader,_=loader(val_env,hp,shuffle=False,batch_size=args.batch_size) if val_env is not None else (None,None)
        n_train_scenes=len(train_env.scenes);n_val_scenes=len(val_env.scenes) if val_env is not None else 0
    run.update(status="running",stage="saving_dataset_manifests",n_train_scenes=n_train_scenes,n_val_scenes=n_val_scenes)
    snapshot_split_inputs(args.run_dir);snapshot_environment(args.run_dir,"train",train_env)
    if val_env is not None:snapshot_environment(args.run_dir,"val",val_env)
    if args.stage=="pretrain":
        snapshot_trajectron_shards(args.run_dir,"train",train_shards)
        snapshot_trajectron_shards(args.run_dir,"val",val_shards)
        # The upstream model constructor reads env.scenes[0].dt while creating
        # its dynamics object, so retain exactly one lightweight template
        # scene. Clearing this list caused a post-cache IndexError after hours
        # of successful cache preparation. Validation scenes are not used for
        # model construction and can be released completely.
        template_scene=train_env.scenes[0]
        template_scene.nodes=[]
        template_scene.temporal_scene_graph=None
        train_env.scenes=[template_scene];val_env.scenes=[];gc.collect()

    registrar = ModelRegistrar(str(args.run_dir / "checkpoints"), device)
    model = Trajectron(registrar, hp, None, device); model.set_environment(train_env); model.set_annealing_params()
    if checkpoint and resume_state is None:
        if args.stage == "finetune":
            if args.allow_legacy_checkpoint:
                run.log("trajectron finetune checkpoint compatibility guard bypassed",
                        checkpoint=str(checkpoint))
            else:
                try:
                    validate_trajectron_finetune_checkpoint(
                        checkpoint_payload, checkpoint,
                        expected_hash=hp["steps_forecast_preprocessing_hash"],
                        expected_commit=UPSTREAM_COMMIT,
                    )
                except ValueError as error:
                    parser.error(str(error))
        registrar.model_dict.load_state_dict(checkpoint_payload["models"])
    optimizer = torch.optim.Adam(registrar.parameters(), lr=hp["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hp["learning_decay_rate"])
    iteration = 0; stopper=EarlyStopping(args.patience,args.min_delta);best_epoch=None
    start_epoch=1;resume_batch=0
    if resume_state is not None:
        registrar.model_dict.load_state_dict(resume_state.payload["models"])
        restore_optimizer_scheduler_state(resume_state,optimizer,scheduler,device)
        iteration=resume_state.iteration;stopper.best=resume_state.best
        stopper.bad_epochs=resume_state.bad_epochs;best_epoch=resume_state.best_epoch
        start_epoch=resume_state.epoch+(1 if resume_state.epoch_complete else 0)
        resume_batch=0 if resume_state.epoch_complete else resume_state.batch_in_epoch
        if start_epoch>args.epochs:raise RuntimeError(f"Resume checkpoint already reached epoch {resume_state.epoch} of {args.epochs}")
        if isinstance(train_loader.dataset,ShardedNodeDataset):
            train_loader.dataset.epoch=start_epoch-1
        # Upstream keeps KL/temperature/logit schedules outside the model and
        # optimizer state dicts. Replaying their cheap scheduler steps restores
        # the exact values reached after ``iteration`` completed updates.
        for _ in range(iteration):
            model.step_annealers(node_type)
        model.set_curr_iter(iteration)
        if resume_state.epoch_complete:
            restore_rng_state(resume_state.payload.get("rng_state"))
        run.log("training state resumed",checkpoint=str(resume_state.checkpoint),epoch=start_epoch,
                iteration=iteration,batch_in_epoch=resume_batch,
                best_val_minade=None if stopper.best==float("inf") else stopper.best,
                unguarded_run_config_fields=list(resume_state.unguarded_fields))
        if resume_state.payload.get("validation_pipeline_version") != VALIDATION_PIPELINE_VERSION:
            # Version 1 rebuilt the upstream node-model wrapper during
            # validation, losing its annealing variables before the next epoch.
            # Re-score the intact completed-epoch weights with the corrected
            # validation path and replace only the stale selection state.
            if not resume_state.epoch_complete:
                raise ValueError(
                                       "Cannot migrate a mid-epoch Trajectron++ checkpoint from the old validation pipeline"
                )
            with torch.no_grad():
                migrated_validation = validation_minade(
                    model,train_env,val_env,
                    val_shards=val_shards if args.stage=="pretrain" else None,
                    k=args.validation_k,stride=args.validation_stride,
                    max_scenes=args.validation_max_scenes,seed=args.data_seed,
                )
            stopper.best=float(migrated_validation["minade"])
            stopper.bad_epochs=0
            best_epoch=resume_state.epoch
            run.metric({"stage":"val","epoch":resume_state.epoch,"iteration":iteration,
                        "minade":stopper.best,"minfde":migrated_validation["minfde"],
                        "n_predictions":migrated_validation["n_predictions"],
                        "n_recordings":migrated_validation["n_recordings"],
                        "reason":"validation_pipeline_migration"})
            save(registrar,optimizer,scheduler,args.run_dir,iteration,stage=args.stage,
                 optimization=optimization,epoch=resume_state.epoch,stopper=stopper,
                 best_epoch=best_epoch,batch_in_epoch=resume_state.batch_in_epoch,
                 epoch_complete=True,epoch_rng_state=resume_state.payload["epoch_rng_state"],best=True)
            run.log("validation checkpoint migrated",epoch=resume_state.epoch,
                    iteration=iteration,best_val_minade=stopper.best,
                    validation_pipeline_version=VALIDATION_PIPELINE_VERSION)
    run.update(status="running", stage="training", n_train_items=len(train_loader.dataset),
               n_val_items=len(val_loader.dataset) if val_loader is not None else 0, n_train_scenes=n_train_scenes, n_val_scenes=n_val_scenes)
    for epoch in range(start_epoch, args.epochs + 1):
        if resume_state is not None and epoch==start_epoch and resume_batch:
            epoch_rng_state=resume_state.payload.get("epoch_rng_state")
            if epoch_rng_state is None:raise ValueError("Mid-epoch Trajectron++ checkpoint lacks epoch RNG state")
            restore_rng_state(epoch_rng_state)
        else:
            epoch_rng_state=capture_rng_state()
        epoch_iterator=iter(train_loader);batch_in_epoch=0
        for _ in range(resume_batch if epoch==start_epoch else 0):
            try:next(epoch_iterator);batch_in_epoch+=1
            except StopIteration:break
        if resume_state is not None and epoch==start_epoch and resume_batch:
            restore_rng_state(resume_state.payload.get("rng_state"))
        batches=run.progress(epoch_iterator,desc=f"train epoch {epoch}/{args.epochs}",
                             total=max(0,len(train_loader)-batch_in_epoch))
        for batch in batches:
            model.set_curr_iter(iteration); model.step_annealers(node_type)
            optimizer.zero_grad(); loss = model.train_loss(batch, node_type)
            loss.backward(); nn.utils.clip_grad_value_(registrar.parameters(), hp["grad_clip"])
            optimizer.step(); scheduler.step(); iteration += 1;batch_in_epoch+=1
            value = float(loss.detach())
            batches.set_postfix(loss=f"{value:.4f}",iteration=iteration)
            if iteration % args.status_every == 0:
                run.metric({"stage": "train", "epoch": epoch, "iteration": iteration, "loss": value})
                run.update(status="running", stage="training", epoch=epoch, iteration=iteration, train_loss=value)
                save(registrar,optimizer,scheduler,args.run_dir,iteration,stage=args.stage,
                     optimization=optimization,epoch=epoch,stopper=stopper,best_epoch=best_epoch,
                     batch_in_epoch=batch_in_epoch,epoch_complete=False,epoch_rng_state=epoch_rng_state)
            if args.iterations is not None and iteration >= args.iterations:
                break
        if val_loader is not None:
            with torch.no_grad():
                validation = validation_minade(
                    model,train_env,val_env,
                    val_shards=val_shards if args.stage=="pretrain" else None,
                    k=args.validation_k,stride=args.validation_stride,
                    max_scenes=args.validation_max_scenes,seed=args.data_seed,
                )
            val = float(validation["minade"])
            run.metric({"stage": "val", "epoch": epoch, "iteration": iteration,
                        "minade":val,"minfde":validation["minfde"],
                        "n_predictions":validation["n_predictions"],
                        "n_recordings":validation["n_recordings"]})
            improved,stopped=stopper.update(val)
            if improved:
                best_epoch=epoch
                save(registrar,optimizer,scheduler,args.run_dir,iteration,stage=args.stage,
                     optimization=optimization,epoch=epoch,stopper=stopper,best_epoch=best_epoch,
                     batch_in_epoch=batch_in_epoch,epoch_complete=True,epoch_rng_state=epoch_rng_state,best=True)
        else:
            stopped=False;best_epoch=epoch
            save(registrar,optimizer,scheduler,args.run_dir,iteration,stage=args.stage,
                 optimization=optimization,epoch=epoch,stopper=stopper,best_epoch=best_epoch,
                 batch_in_epoch=batch_in_epoch,epoch_complete=True,epoch_rng_state=epoch_rng_state,best=True)
        save(registrar,optimizer,scheduler,args.run_dir,iteration,stage=args.stage,
             optimization=optimization,epoch=epoch,stopper=stopper,best_epoch=best_epoch,
             batch_in_epoch=batch_in_epoch,epoch_complete=True,epoch_rng_state=epoch_rng_state)
        run.update(status="running", stage="training", epoch=epoch, iteration=iteration, best_val_minade=stopper.best if val_loader is not None else None,
                   early_stopping_bad_epochs=stopper.bad_epochs if val_loader is not None else 0,early_stopping_patience=args.patience if val_loader is not None else 0)
        if stopped:
            run.log("early stopping",epoch=epoch,iteration=iteration,best_val_minade=stopper.best,patience=args.patience)
            break
        if args.iterations is not None and iteration >= args.iterations:
            break
        resume_batch=0;resume_state=None
    run.update(status="complete", stage="training", epoch=epoch, iteration=iteration, best_val_minade=stopper.best if val_loader is not None else None,early_stopped=stopped)


if __name__ == "__main__":
    main()
