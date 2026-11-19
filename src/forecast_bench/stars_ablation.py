"""STARS ablation ladder, rungs L0-L3: frozen self-supervised STARS encoder + STARS's linear forecasting probe on
the ETH/UCY leave-one-scene-out (LOSO) benchmark.

The encoder is a ``train_stars_ma --phase ssl`` checkpoint; its ``graph`` entry says how graphs are built
(STARS's 40-step grid, heading rule, robot-slot target, node crop, kNN/radius). The probe is STARS's
``train_swm_forecasting_linear_probe_v1.py``: a linear layer on ``[z_i | mean z_others | history_i]`` that
predicts all 12 steps (``baselines.stars.ForecastHead`` / ``HEAD_RECIPE``: AdamW 1e-3, weight decay 1e-4,
batch 8, <= 80 epochs, LR on plateau, early stopping on validation ADE). ``z`` is the encoder's posterior mean;
"others" are the other nodes of the graph; history and future are in the agent's local frame (origin at its
last observed position, x along its last heading).

* ``--mode single`` (L0-L2, STARS's protocol): one graph per (window, target) with the target in the robot
  slot; trained on every agent observed at all 20 steps of a training window; at test time every scored agent
  is a target in turn.
* ``--mode multi`` (L3): one graph per window, every node is encoded once and every node with a supervised
  future step trains the head (masked MSE).

For each held-out scene the head trains on the fold's Social-GAN ``train`` files (the other four scenes) and is
selected on its ``val`` files; it predicts the held-out scene's test windows (``forecast_bench.windows``).
Node features are computed once per file (a node's features depend only on its own window).

    python -m forecast_bench.stars_ablation --encoder <ssl best.pt> --mode single --seed 1 --method abl_L0_s1
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from forecast_bench.windows import FOLDS, ROOT, load

SGAN_SPLITS = ROOT / "references/moflow/data/eth_ucy/original"
BENCH = ROOT / "experiments/forecast_bench"


def load_encoder(path: Path, device):
    from baselines.stars_ma_model import StarsMotionVAE

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != "stars_ma" or payload.get("phase") != "ssl":
        raise ValueError(f"{path} is not a train_stars_ma --phase ssl checkpoint")
    vae = StarsMotionVAE(payload["window"]["observation_steps"], payload["encoder_recipe"])
    vae.load_state_dict(payload["state_dict"])
    return vae.encoder.to(device).eval(), payload


@torch.no_grad()
def node_features(encoder, graph: dict, windows: dict, mode: str, device, max_nodes: int = 8192) -> dict:
    """Probe inputs of one window set. ``single``: one row per scored agent (in node order); ``multi``: one row
    per node, with ``scored`` marking the scored ones. Rows: ``x`` features, ``y [12, 2]`` local future,
    ``y_valid [12]``, ``origin``/``heading`` (local -> scene)."""
    from baselines.stars_ma_batch import _ranges, build_batch, graph_options, node_budget_batches, per_target_store
    from baselines.stars_ma_graph import to_local
    from forecast_bench.predict import to_store
    from torch_geometric.utils import scatter
    from data import CANONICAL_DT

    store = to_store(windows, "cpu")
    if mode == "single":
        if not graph.get("target_slot"):
            raise ValueError("--mode single needs a single-target (--target-slot) encoder")
        store, _ = per_target_store(store, targets="scored", max_graph_nodes=graph.get("max_graph_nodes"))
    elif graph.get("target_slot"):
        raise ValueError("--mode multi needs a per-window encoder (no --target-slot)")
    store = store.to(device)
    out = {key: [] for key in ("x", "y", "y_valid", "origin", "heading", "scored", "order")}
    for w in node_budget_batches(store.sizes, max_nodes, shuffle=False):
        w = w.to(device)
        batch = build_batch(store, w, dt=CANONICAL_DT, **graph_options(graph))
        mu = encoder(batch)["mu"].float()
        total = scatter(mu, batch.batch, dim=0, reduce="sum", dim_size=batch.num_graphs)
        count = torch.bincount(batch.batch, minlength=batch.num_graphs).float()
        others = (total[batch.batch] - mu) / (count[batch.batch] - 1).clamp(min=1)[:, None]
        history = to_local(batch.obs_xy.float(), batch.origin.float(), batch.heading.float()).flatten(1)
        future = to_local(torch.nan_to_num(batch.future.float()), batch.origin.float(), batch.heading.float())
        rows = batch.ptr[:-1] if mode == "single" else torch.arange(batch.num_nodes, device=device)
        x = torch.cat([mu, others, history], dim=-1)
        out["x"].append(x[rows].cpu())
        out["y"].append(future[rows].cpu())
        out["y_valid"].append(batch.future_valid[rows].cpu())
        out["origin"].append(batch.origin[rows].float().cpu())
        out["heading"].append(batch.heading[rows].float().cpu())
        full = (batch.future_valid.all(-1) & batch.valid_mask.bool().all(-1))[rows]
        out["scored"].append(full.cpu())
        # position of each row in the set's node order (single: the per-target windows are in node order)
        out["order"].append(w.cpu() if mode == "single" else _ranges(store.ptr[w], store.sizes[w]).cpu())
    feats = {key: torch.cat(value) for key, value in out.items()}
    order = torch.argsort(feats.pop("order"))
    return {key: value[order] for key, value in feats.items()}


def _concat(parts: list[dict]) -> dict:
    return {key: torch.cat([p[key] for p in parts]) for key in parts[0]}


def fit_head(train: dict, val: dict, *, seed: int, device, recipe: dict, log=None):
    """STARS's linear probe: masked MSE on the local-frame future, selected on validation ADE of scored agents."""
    from baselines.early_stopping import EarlyStopping

    torch.manual_seed(seed)
    dim = train["x"].shape[1]
    head = torch.nn.Linear(dim, 24).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=recipe["lr_patience"])
    stopper = EarlyStopping(recipe["patience"], recipe["min_delta"])
    generator = torch.Generator().manual_seed(seed)
    keep = train["y_valid"].any(-1)
    tx, ty, tv = (train[k][keep].to(device) for k in ("x", "y", "y_valid"))
    tv = tv.float()[..., None]
    scored = val["scored"]
    vx, vy = val["x"][scored].to(device), val["y"][scored].to(device)
    best, best_state, best_epoch = None, None, 0
    for epoch in range(1, recipe["max_epochs"] + 1):
        order = torch.randperm(len(tx), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), recipe["batch_size"]):
            index = order[start:start + recipe["batch_size"]]
            error = (head(tx[index]).view(-1, 12, 2) - ty[index]) ** 2 * tv[index]
            loss = error.sum() / (2 * tv[index].sum()).clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        with torch.no_grad():
            prediction = head(vx).view(-1, 12, 2)
            val_mse = float(torch.nn.functional.mse_loss(prediction, vy))
            ade = float(torch.linalg.norm(prediction - vy, dim=-1).mean())
        scheduler.step(val_mse)
        if log:
            log({"epoch": epoch, "train_mse": total / len(order), "val_mse": val_mse, "val_ade": ade,
                 "lr": optimizer.param_groups[0]["lr"]})
        improved, stopped = stopper.update(ade)
        if improved:
            best, best_state, best_epoch = ade, copy.deepcopy(head.state_dict()), epoch
        if stopped:
            break
    head.load_state_dict(best_state)
    return head, {"val_ade": best, "best_epoch": best_epoch, "epochs": epoch}


def fold_files(fold: str, split: str) -> list[Path]:
    return sorted((SGAN_SPLITS / fold / split).glob("*.txt"))


def main() -> None:
    from baselines.stars import HEAD_RECIPE
    from baselines.stars_ma_graph import to_scene
    from forecast_bench.windows import _pack, sgan_windows

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    parser.add_argument("--method", required=True, help="prediction name: <root>/preds/ethucy_<fold>/<method>.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--root", type=Path, default=BENCH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-nodes", type=int, default=8192)
    parser.add_argument("--max-windows", type=int, help="smoke test: cap windows per file / test set")
    parser.add_argument("--head-epochs", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    recipe = {**HEAD_RECIPE, **({"max_epochs": args.head_epochs} if args.head_epochs else {})}
    out_dir = args.root / "ablation" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, payload = load_encoder(args.encoder, device)
    graph = payload["graph"]

    def cap(windows: dict) -> dict:
        if not args.max_windows or len(windows["ptr"]) - 1 <= args.max_windows:
            return windows
        n = int(windows["ptr"][args.max_windows])
        out = {k: (v[:n] if k not in ("ptr", "window_key") else v) for k, v in windows.items()}
        out["ptr"], out["window_key"] = windows["ptr"][:args.max_windows + 1], windows["window_key"][:args.max_windows]
        return out

    started = time.time()
    cache: dict[tuple[str, int], dict] = {}

    def file_features(path: Path) -> dict:
        key = (path.name, path.stat().st_size)  # the same scene file is shared by four folds
        if key not in cache:
            cache[key] = node_features(encoder, graph, cap(_pack(sgan_windows(path))), args.mode, device, args.max_nodes)
        return cache[key]

    summary = {"encoder": str(args.encoder), "mode": args.mode, "seed": args.seed, "graph": graph,
               "encoder_recipe": payload["encoder_recipe"], "head_recipe": recipe, "folds": {}}
    for fold in args.folds.split(","):
        train = _concat([file_features(p) for p in fold_files(fold, "train")])
        val = _concat([file_features(p) for p in fold_files(fold, "val")])
        windows = cap(load(args.root / "windows" / f"ethucy_{fold}.npz"))
        test = node_features(encoder, graph, windows, args.mode, device, args.max_nodes)
        logs = []
        head, info = fit_head(train, val, seed=args.seed, device=device, recipe=recipe, log=logs.append)
        with torch.no_grad():
            rows = test["scored"] if args.mode == "multi" else torch.ones(len(test["x"]), dtype=torch.bool)
            local = head(test["x"][rows].to(device)).view(-1, 12, 2).cpu()
            scene = to_scene(local, test["origin"][rows], test["heading"][rows])
        if len(scene) != int(windows["eval_mask"].sum()):
            raise RuntimeError(f"{fold}: {len(scene)} predictions for {int(windows['eval_mask'].sum())} scored agents")
        gt = windows["fut"][windows["eval_mask"]]
        test_ade = float(np.linalg.norm(scene.numpy() - gt, axis=-1).mean())
        path = args.root / "preds" / f"ethucy_{fold}" / f"{args.method}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez(tmp, pred=scene.numpy()[:, None].astype(np.float32))
        tmp.replace(path)
        (out_dir / f"{fold}_head_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
        summary["folds"][fold] = {**info, "test_ade": test_ade, "n_train_rows": int(train["y_valid"].any(-1).sum()),
                                  "n_val_scored": int(val["scored"].sum()), "n_test": len(scene)}
        print(f"{args.method} {fold}: val ADE {info['val_ade']:.3f} (epoch {info['best_epoch']}), test ADE {test_ade:.3f}, "
              f"{summary['folds'][fold]['n_train_rows']} train rows, {time.time() - started:.0f} s", flush=True)
    summary["seconds"] = time.time() - started
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
