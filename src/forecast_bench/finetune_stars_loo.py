"""Leave-one-out STARS for the standard ETH/UCY protocol: fine-tune a checkpoint that never saw ETH/UCY.

Starts from ``--init`` (default: the pooled no-ETH/UCY CVAE, ``stars_noethucy``) and fine-tunes the
whole model on the held-out fold's standard Social-GAN training files (``<fold>/train/*_train.txt``:
the other four scenes), selecting the epoch with the best minADE20 on the fold's ``val`` files. The
held-out scene is never used. Same CVAE loss as ``train_stars_ma`` (teacher-forced L2 + 0.1 KL).

    python -m forecast_bench.finetune_stars_loo --fold eth --out experiments/forecast_bench/stars_loo/eth
"""
from __future__ import annotations

import argparse
import json
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
    from baselines.train_stars_ma import forecast_loss, load_forecaster
    from data import CANONICAL_DT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", required=True, choices=FOLDS)
    parser.add_argument("--init", default="stars_noethucy", choices=sorted(STARS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--max-nodes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, help="Smoke test: cap training batches per epoch")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    model, payload = load_forecaster(STARS[args.init], torch.device(args.device))
    graph = payload["graph"]
    loss_args = SimpleNamespace(rollout_exec_steps=0, rollout_weight=0.0, knn=graph["knn"], radius=graph["radius"])
    stores = {split: to_store(fold_windows(args.fold, split), args.device) for split in ("train", "val")}
    batch = lambda store, w: build_batch(store, w.to(store.device), dt=CANONICAL_DT, knn=graph["knn"],
                                         radius=graph["radius"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log = open(args.out / "log.jsonl", "w")

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        store, errors = stores["val"], []
        for w in node_budget_batches(store.sizes, 8192, shuffle=False):
            b = batch(store, w)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = to_scene(model.predict(b, samples=20).float(), b.origin, b.heading)
            full = b.future_valid.all(-1) & b.valid_mask.bool().all(-1)
            err = torch.linalg.norm(pred - b.future[None], dim=-1).mean(-1)[:, full]
            errors.append(err.min(0).values)
        return float(torch.cat(errors).mean())

    best = validate()
    print(json.dumps({"epoch": 0, "val_min_ade": best}), file=log, flush=True)
    torch.save({**payload, "state_dict": model.state_dict(), "loo_fold": args.fold, "epoch": 0}, args.out / "best.pt")
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, losses = time.time(), []
        for w in node_budget_batches(stores["train"].sizes, args.max_nodes, generator=generator,
                                     shuffle=True)[:args.max_batches]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forecast_loss(model, batch(stores["train"], w), loss_args, args.kl_weight)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        val = validate()
        print(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_min_ade": val,
                          "seconds": time.time() - start}), file=log, flush=True)
        if val < best:
            best = val
            torch.save({**payload, "state_dict": model.state_dict(), "loo_fold": args.fold, "epoch": epoch},
                       args.out / "best.pt")
    print(f"{args.fold}: best val minADE20 {best:.3f}")


if __name__ == "__main__":
    main()
