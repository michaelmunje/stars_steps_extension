"""Leave-one-scene-out (LOSO) STARS for the standard ETH/UCY protocol.

Starts from ``--init`` (a ``forecast_bench.predict.STARS`` key or a checkpoint path; it must never have seen
ETH/UCY) and trains the whole model on the held-out fold's standard Social-GAN training files
(``<fold>/train/*_train.txt``: the other four scenes), selecting the epoch with the best minADE20 on the fold's
``val`` files (also the other four scenes). The held-out scene is never used. Same CVAE loss as
``train_stars_ma`` (teacher-forced L2 + KL). AdamW with a linear warmup and cosine decay to 0 over
``--epochs``, optionally a lower learning rate for the encoder, early stopping after ``--patience`` epochs
without a validation improvement. ``--scratch`` re-initialises the weights (same architecture): the
"LOSO from scratch" control, with a KL warmup. ``--rft-steps K`` adds closed-loop rollout fine-tuning
(``train_stars_ma``'s rollout loss on the fold's training scenes, curriculum from ``--rft-start-steps``) and
selects on the K-step closed-loop ADE of the fold's val split. Resumable (``latest.pt`` every epoch).

    python -m forecast_bench.finetune_stars_loo --fold eth --init <pretrained best.pt> --lr 1e-4 \
        --out experiments/forecast_bench/loso/<name>/eth
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from forecast_bench.predict import STARS, to_store
from forecast_bench.windows import FOLDS, ROOT, _pack, sgan_windows

SGAN_SPLITS = ROOT / "references/moflow/data/eth_ucy/original"


def fold_windows(fold: str, split: str) -> dict:
    windows = []
    for path in sorted((SGAN_SPLITS / fold / split).glob("*.txt")):
        windows += sgan_windows(path)
    return _pack(windows)


def main() -> None:
    from baselines.stars_ma_batch import build_batch, node_budget_batches
    from baselines.stars_ma_graph import to_scene
    from baselines.stars_ma_model import StarsMultiAgentForecaster
    from baselines.train_stars_ma import closed_loop_metrics, forecast_loss, load_forecaster
    from data import CANONICAL_DT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", required=True, choices=FOLDS)
    parser.add_argument("--init", default="stars_noethucy", help="STARS key or checkpoint path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-mult", type=float, default=1.0, help="encoder LR = lr * this")
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0, help="stop after this many epochs without a val gain (0: off)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=float, default=0.0, help="linear KL ramp (scratch runs)")
    parser.add_argument("--scratch", action="store_true", help="re-initialise all weights (no pretraining)")
    parser.add_argument("--rft-steps", type=int, default=0, help="closed-loop rollout fine-tuning steps (0: off)")
    parser.add_argument("--rft-start-steps", type=int, default=4)
    parser.add_argument("--rft-ramp-epochs", type=int, default=3)
    parser.add_argument("--rft-speed-weight", type=float, default=1.0)
    parser.add_argument("--max-nodes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, help="Smoke test: cap training batches per epoch")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        print(f"{args.out} already complete")
        return

    device = torch.device(args.device)
    model, payload = load_forecaster(STARS.get(args.init, Path(args.init)), device)
    if args.scratch:
        window = payload["window"]
        model = StarsMultiAgentForecaster(window["observation_steps"], window["prediction_steps"], dt=window["dt"],
                                          encoder_recipe=payload["encoder_recipe"],
                                          decoder_recipe=payload["decoder_recipe"]).to(device)
    graph = payload["graph"]
    loss_args = SimpleNamespace(rollout_exec_steps=0, rollout_weight=0.0, knn=graph["knn"], radius=graph["radius"],
                                rft_current_steps=0, rft_latent="prior", rft_rho=0.8, rft_bptt=0,
                                rft_speed_weight=args.rft_speed_weight, seed=args.seed, amp=True,
                                device_realized=device, world_size=1)
    stores = {split: to_store(fold_windows(args.fold, split), args.device) for split in ("train", "val")}
    batch = lambda store, w: build_batch(store, w.to(store.device), dt=CANONICAL_DT, knn=graph["knn"],
                                         radius=graph["radius"])
    encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.lr * args.encoder_lr_mult},
                                   {"params": rest, "lr": args.lr}], weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    steps = len(node_budget_batches(stores["train"].sizes, args.max_nodes, shuffle=False)[:args.max_batches])
    warmup, total = max(1, int(args.warmup_epochs * steps)), args.epochs * steps
    schedule = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    meta = {**payload, "loo_fold": args.fold, "loso_args": {k: str(v) for k, v in vars(args).items()}}

    @torch.no_grad()
    def validate() -> dict:
        if args.rft_steps:   # select on the closed loop (one prior sample, AR(1) latent noise, seeded)
            store = stores["val"]
            cl = closed_loop_metrics(model, store, node_budget_batches(store.sizes, 4096, shuffle=False), loss_args,
                                     args.rft_steps)
            return {"val_min_ade": cl["cl_ade"], "val_cl_fde": cl["cl_fde"], "val_cl_speed": cl["cl_speed"],
                    "val_cl_turn": cl["cl_turn"]}
        model.eval()
        store, best, mean = stores["val"], [], []
        for w in node_budget_batches(store.sizes, 8192, shuffle=False):
            b = batch(store, w)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = to_scene(model.predict(b, samples=20).float(), b.origin, b.heading)
            full = b.future_valid.all(-1) & b.valid_mask.bool().all(-1)
            err = torch.linalg.norm(pred - b.future[None], dim=-1).mean(-1)[:, full]
            best.append(err.min(0).values)
            mean.append(err.mean(0))
        return {"val_min_ade": float(torch.cat(best).mean()), "val_mean_ade": float(torch.cat(mean).mean())}

    latest = args.out / "latest.pt"
    if latest.exists():   # resume
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        start, best, best_epoch = state["epoch"] + 1, state["best"], state["best_epoch"]
    else:
        val = validate()
        best, best_epoch, start = val["val_min_ade"], 0, 1
        with open(args.out / "log.jsonl", "w") as log:
            print(json.dumps({"epoch": 0, **val}), file=log, flush=True)
        torch.save({**meta, "state_dict": model.state_dict(), "epoch": 0}, args.out / "best.pt")
    for epoch in range(start, args.epochs + 1):
        if args.patience and epoch - best_epoch > args.patience:
            break
        model.train()
        t0, losses = time.time(), []
        if args.rft_steps:
            ramp = min(1.0, (epoch - 1) / args.rft_ramp_epochs) if args.rft_ramp_epochs else 1.0
            loss_args.rft_current_steps = int(round(args.rft_start_steps + (args.rft_steps - args.rft_start_steps) * ramp))
        kl = args.kl_weight * (min(1.0, (epoch - 1) / args.kl_warmup_epochs) if args.kl_warmup_epochs else 1.0)
        for w in node_budget_batches(stores["train"].sizes, args.max_nodes, generator=generator,
                                     shuffle=True)[:args.max_batches]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forecast_loss(model, batch(stores["train"], w), loss_args, kl)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
        val = validate()
        with open(args.out / "log.jsonl", "a") as log:
            print(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), **val,
                              "lr": scheduler.get_last_lr()[-1], "seconds": time.time() - t0}), file=log, flush=True)
        if val["val_min_ade"] < best:
            best, best_epoch = val["val_min_ade"], epoch
            torch.save({**meta, "state_dict": model.state_dict(), "epoch": epoch}, args.out / "best.pt")
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                    "torch_rng": torch.get_rng_state(), "epoch": epoch, "best": best, "best_epoch": best_epoch},
                   latest.with_suffix(".tmp"))
        latest.with_suffix(".tmp").replace(latest)
    (args.out / "summary.json").write_text(json.dumps({"fold": args.fold, "best_val_min_ade": best,
                                                       "best_epoch": best_epoch, **vars(args)}, default=str, indent=1))
    latest.unlink(missing_ok=True)
    print(f"{args.fold}: best val minADE20 {best:.3f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
