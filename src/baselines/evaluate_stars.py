"""Evaluate a STARS forecasting checkpoint on a held-out target test split.

Deterministic by default (K=1, the latent mean). ``--latent-samples K`` reports best-of-K from
posterior latent samples instead. ``evaluation.json`` carries the same fields as the other
baselines' evaluators; ``k`` records which of the two was run.
"""
from __future__ import annotations

import argparse
import math
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

from baselines.device import training_device
from baselines.preprocessing_options import (
    add_preprocessing_arguments,
    config_preprocessing,
    handle_preprocessing_introspection,
)
from baselines.provenance import snapshot_dataset, snapshot_split_inputs
from baselines.registry import resolve
from baselines.stars import (
    DEFAULT_MAX_EDGES,
    REFERENCE,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    ForecastHead,
    StarsDataset,
    best_of_k,
    build_vae,
)
from baselines.status import RunStatus, atomic_json
from baselines.train_stars import DEFAULT_PRESET
from baselines.transfer_data import downstream_test_dataset, resolve_target_name
from config import Config, SeedConfig
from data import seed_worker
from forecasting_config import config_as_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", choices=("eth", "hotel", "univ", "zara1", "zara2"))
    parser.add_argument("--target-dataset", choices=("eth_ucy", "grand_central", "madras"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-samples", type=int, default=1, help="K>1: best-of-K over posterior latent samples")
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES, help="Edge budget per forward pass (memory only)")
    add_preprocessing_arguments(parser)
    return parser


def main() -> None:
    from torch_geometric.loader import DataLoader

    handle_preprocessing_introspection(default_preset=DEFAULT_PRESET)
    parser = _parser()
    args = parser.parse_args()
    try:
        args.target_dataset = resolve_target_name(args.target_dataset, args.fold)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1 or args.latent_samples < 1:
        parser.error("--batch-size and --latent-samples must be positive")
    device = training_device(args.device)
    checkpoint = resolve(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model") != "stars":
        parser.error(f"{checkpoint} is not a STARS checkpoint")
    forecasting = config_preprocessing(args, default_preset=DEFAULT_PRESET)["forecasting"]
    training_config = payload.get("run_config", {})
    checkpoint_hash = training_config.get("forecast_preprocessing_hash")
    if checkpoint_hash != forecasting.forecast_preprocessing_hash:
        raise ValueError(
            "Evaluation preprocessing differs from the checkpoint. Pass the same preset/config "
            f"overrides used for training ({checkpoint_hash} != {forecasting.forecast_preprocessing_hash})."
        )
    checkpoint_data_seed = training_config.get("data_seed")
    if checkpoint_data_seed is not None and checkpoint_data_seed != args.data_seed:
        warnings.warn(f"Checkpoint data_seed {checkpoint_data_seed} does not match --data-seed {args.data_seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    architecture = payload["architecture"]
    model = build_vae(device, payload["vae_recipe"])
    model.load_state_dict(payload["vae"])
    model.eval()
    head = ForecastHead(architecture["latent_dim"], architecture["observation_steps"], architecture["prediction_steps"],
                        use_history=architecture["use_history"]).to(device)
    head.load_state_dict(payload["head"])
    head.eval()

    cfg = Config(seeds=SeedConfig(args.data_seed, args.seed), forecasting=forecasting)
    dataset = StarsDataset(downstream_test_dataset(args.target_dataset, args.fold, cfg),
                           forecasting.window.observation_steps, limit=args.max_scenes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    run = RunStatus(args.run_dir, {
        **vars(args),
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_reference": args.checkpoint,
        "model": "stars",
        "device_realized": str(device),
        "architecture": architecture,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": str(REFERENCE.relative_to(Path(__file__).resolve().parents[2])),
        **forecasting.flat_provenance(),
        "forecasting_config": config_as_dict(forecasting),
    })
    try:
        snapshot_split_inputs(args.run_dir)
        snapshot_dataset(args.run_dir, "test", dataset)
        run.update(status="running", stage="evaluation", n_predictions=0)
        ades: list[float] = []
        fdes: list[float] = []
        label = args.fold if args.target_dataset == "eth_ucy" else args.target_dataset
        batches = run.progress(loader, desc=f"STARS evaluate {label}")
        for batch in batches:
            ade, fde, _world = best_of_k(model, head, batch.to(device), latent_samples=args.latent_samples,
                                         max_edges=args.max_edges)
            ades.extend(ade.tolist())
            fdes.extend(fde.tolist())
            batches.set_postfix(n=len(ades), ade=f"{float(np.mean(ades)):.3f}")
        if not ades:
            raise RuntimeError("STARS evaluation population is empty")
        n = len(ades)
        selection = payload.get("checkpoint_selection", {})
        stage = payload.get("stage")
        finetuned = stage == "finetune"
        result = {
            "k": args.latent_samples,
            "fpc": False,
            "n_predictions": n,
            "topk_ade": float(np.mean(ades)),
            "topk_fde": float(np.mean(fdes)),
            "topk_ade_standard_error": float(np.std(ades, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "topk_fde_standard_error": float(np.std(fdes, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "visual_scenes": 0,
            "source_datasets": training_config.get("source_datasets", []),
            "target_dataset": args.target_dataset,
            "target_fold": args.fold,
            "target_fraction": training_config.get("target_fraction"),
            "seed": training_config.get("seed", args.seed),
            "pretraining_seed": training_config.get("pretraining_seed"),
            "finetuning_seed": training_config.get("finetuning_seed"),
            "pretraining_checkpoint": training_config.get("pretraining_checkpoint"),
            "pretraining_best_validation_metric": None if finetuned else selection.get("value"),
            "pretraining_best_epoch": None if finetuned else selection.get("best_epoch"),
            "pretraining_validation_metric": None if finetuned else selection.get("value"),
            "fine_tuning_learning_rate": payload.get("head_recipe", {}).get("learning_rate") if finetuned else None,
            "fine_tuning_best_epoch": selection.get("best_epoch") if finetuned else None,
            "fine_tuning_validation_metric": selection.get("value") if finetuned else None,
            "checkpoint_metric": selection.get("metric"),
            "transfer_optimization_hash": None,
            "forecast_preprocessing_hash": checkpoint_hash,
            "data_seed": args.data_seed,
            "checkpoint_data_seed": checkpoint_data_seed,
            "training_stage": stage,
            "checkpoint_reference": args.checkpoint,
        }
        atomic_json(args.run_dir / "evaluation.json", result)
        run.metric({"stage": "test", **result})
        run.update(status="complete", stage="evaluation", **result)
    except Exception as error:
        run.log("run failed", error=repr(error), traceback=traceback.format_exc())
        run.update(status="failed", stage="evaluation", error=repr(error))
        raise


if __name__ == "__main__":
    main()
