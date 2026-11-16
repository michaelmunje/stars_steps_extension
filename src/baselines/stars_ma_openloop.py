"""Open-loop (teacher-forced) forecasting realism report for multi-agent STARS, and ground-truth statistics.

For every dataset of a split (``unique`` windows by default: each scene window scored once, every
agent with a fully observed future), it reports

* accuracy: single-sample ADE/FDE (sample 0 of K, i.e. a random draw), ADE/FDE of the K-sample
  mean, of the prior-mean latent (CVAE only, ``mode``), min-of-K agent ADE/FDE and joint
  min-of-K ADE/FDE (one sample index per scene);
* collisions among the predicted agents of a scene: the fraction of agent pairs closer than
  ``d`` at some future step (``pair``) and the fraction of pair-steps closer than ``d``
  (``pairstep``), for d in ``COLLISION_DISTANCES``, for the ground truth, each sample (averaged
  over K), the sample mean, the mode, and a *shuffled* joint sample whose agents take independent
  sample indices (if it collides as often as a real joint sample, the K samples are not mutually
  coordinated); ``new_pair`` counts predicted collisions of pairs that never get that close in
  the ground truth;
* kinematics: mean speed over the 12 future steps (from the last observed position) for the
  ground truth and each prediction, its per-horizon profile, the 1-D Wasserstein distance between
  predicted (sample 0) and ground-truth per-step speeds, and the mean absolute turning per step.

Without ``--checkpoint`` only the ground truth and constant velocity are reported (data diagnosis),
plus window statistics (agents per window, complete-future fraction, partially observed agents,
close pairs that are already close and co-moving at the last observed step, i.e. groups).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from baselines.stars_ma_model import observed_groups, select_scene_samples

COLLISION_DISTANCES = (0.2, 0.3, 0.4)


def _speeds(path: torch.Tensor, dt: float) -> torch.Tensor:
    """``[..., T+1, 2]`` -> per-step speeds ``[..., T]``."""
    return torch.linalg.norm(path[..., 1:, :] - path[..., :-1, :], dim=-1) / dt


def _turning(path: torch.Tensor, min_step: float = 0.04) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute heading change between consecutive steps (rad) and its validity (both steps moving)."""
    step = path[..., 1:, :] - path[..., :-1, :]
    angle = torch.atan2(step[..., 1], step[..., 0])
    change = (angle[..., 1:] - angle[..., :-1] + np.pi) % (2 * np.pi) - np.pi
    norm = torch.linalg.norm(step, dim=-1)
    return change.abs(), (norm[..., 1:] > min_step) & (norm[..., :-1] > min_step)


class Report:
    """Per-dataset running sums (``add``) and sample reservoirs (``keep``)."""

    def __init__(self) -> None:
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(lambda: defaultdict(float))
        self.samples = defaultdict(lambda: defaultdict(list))

    def add(self, dataset: str, name: str, total, count) -> None:
        total = float(total.sum()) if torch.is_tensor(total) else float(total)
        count = float(count.sum()) if torch.is_tensor(count) else float(count)
        for key in (dataset, "all"):
            self.sums[key][name] += total
            self.counts[key][name] += count

    def keep(self, dataset: str, name: str, values: torch.Tensor) -> None:
        self.samples[dataset][name].append(values.flatten().float().cpu().numpy())

    def result(self) -> dict:
        out = {}
        for key in self.sums:
            row = {name: self.sums[key][name] / self.counts[key][name]
                   for name in self.sums[key] if self.counts[key][name] > 0}
            row.update({f"n_{name}": self.counts[key][name] for name in ("agents", "pairs", "windows")
                        if name in self.counts[key]})
            samples = {name: np.concatenate(v) for name, v in self.samples[key].items()}
            if "speed_gt" in samples:
                gt = samples["speed_gt"]
                quantiles = np.linspace(0, 1, 512)
                for name, values in samples.items():
                    if name.startswith("speed_") and name != "speed_gt":
                        row[f"{name}_w1"] = float(np.mean(np.abs(np.quantile(values, quantiles) - np.quantile(gt, quantiles))))
                row["speed_gt_median"] = float(np.median(gt))
                row["speed_gt_p90"] = float(np.quantile(gt, 0.9))
            out[key] = row
        return out


def scene_metrics(report: Report, names: list[str], batch, predictions: dict[str, torch.Tensor], dt: float,
                  *, ground_truth_stats: bool = True) -> None:
    """Add one batch. ``predictions`` maps a name to scene-frame ``[K, Nt, P, 2]`` futures; the first
    entry with K > 1 is the sampled model (sample 0, mean, shuffled and min-of-K metrics)."""
    mask = batch.node_mask                                        # [B, N]
    B, N = mask.shape
    complete = batch.to_padded(batch.future_valid.all(dim=-1), False) & mask
    future = batch.to_padded(torch.nan_to_num(batch.future))      # [B, N, P, 2]
    last = batch.to_padded(batch.obs_xy[:, -1])                    # [B, N, 2]
    upper = complete[:, :, None] & complete[:, None, :] & torch.ones(N, N, dtype=torch.bool, device=mask.device).triu(1)
    dataset = batch.dataset

    def per(name, total_bn, count_bn):
        """Add per-graph totals of ``[B]`` tensors split by dataset."""
        for index in torch.unique(dataset).tolist():
            chosen = dataset == index
            report.add(names[index], name, total_bn[chosen].sum(), count_bn[chosen].sum())

    cf = complete.float()
    agents = cf.sum(-1)
    pairs = upper.float().sum((1, 2))
    per("agents", agents, agents)
    per("pairs", pairs, pairs)
    per("windows", torch.ones(B, device=mask.device), torch.ones(B, device=mask.device))

    def pair_distance(xy):                                        # [B, N, P, 2] -> [B, N, N, P]
        return torch.linalg.norm(xy[:, :, None] - xy[:, None], dim=-1)

    group = observed_groups(batch, dt) & upper
    ungrouped = upper & ~group
    ng_pairs = ungrouped.float().sum((1, 2))

    def collisions(prefix, xy, extra_gt=None):
        distance = pair_distance(xy)
        closest = distance.min(-1).values
        uf = upper.float()
        for d in COLLISION_DISTANCES:
            per(f"{prefix}_pair{d}", ((closest < d).float() * uf).sum((1, 2)), pairs)
            per(f"{prefix}_pairstep{d}", ((distance < d).float().mean(-1) * uf).sum((1, 2)), pairs)
            per(f"{prefix}_nogroup_pair{d}", ((closest < d) & ungrouped).float().sum((1, 2)), ng_pairs)
            if extra_gt is not None:
                per(f"{prefix}_new_pair{d}", ((closest < d) & (extra_gt >= d)).float().mul(uf).sum((1, 2)), pairs)
        per(f"{prefix}_min_distance", (closest.clamp(max=10.0) * uf).sum((1, 2)), pairs)

    gt_distance = pair_distance(future)
    gt_closest = gt_distance.min(-1).values
    if ground_truth_stats:
        collisions("gt", future)
        # Ground-truth close pairs: groups (already within 1 m and co-moving at the last observed
        # step) vs others.
        for d in (0.2, 0.4):
            close = (gt_closest < d) & upper
            per(f"gt_pair{d}_group_share_num", (close & group).float().sum((1, 2)), close.float().sum((1, 2)))
        per("group_pair_rate", group.float().sum((1, 2)), pairs)
        per("gt_duplicate_pair0.05", (((gt_closest < 0.05) & upper).float()).sum((1, 2)), pairs)
        # Window / node bookkeeping (padding, partial observations, partial futures).
        nodes = mask.float().sum(-1)
        per("complete_node_fraction", agents, nodes)
        observed_valid = batch.to_padded(batch.valid_mask) > 0     # [B, N, T]
        partial = (~observed_valid.all(-1)) & mask
        per("partial_obs_node_fraction", partial.float().sum(-1), nodes)
        per("nodes_per_window", nodes, torch.ones(B, device=mask.device))

    # Ground-truth kinematics (complete agents).
    gt_path = torch.cat([last[:, :, None], future], dim=2)
    gt_speed = _speeds(gt_path, dt)                                # [B, N, P]
    for index in torch.unique(dataset).tolist():
        chosen = (dataset == index)[:, None] & complete
        report.keep(names[index], "speed_gt", gt_speed[chosen])
    per("speed_gt", (gt_speed.mean(-1) * cf).sum(-1), agents)
    gt_net = torch.linalg.norm(future[:, :, -1] - last, dim=-1) / (future.shape[2] * dt)
    per("netspeed_gt", (gt_net * cf).sum(-1), agents)
    per("stationary_gt", ((gt_speed.mean(-1) < 0.1).float() * cf).sum(-1), agents)
    turn, turn_valid = _turning(gt_path)
    per("turn_gt", (turn * turn_valid * complete[..., None]).sum((1, 2)), (turn_valid * complete[..., None]).sum((1, 2)))
    for step in range(gt_speed.shape[-1]):
        per(f"speed_gt_t{step + 1}", (gt_speed[..., step] * cf).sum(-1), agents)

    sampled = None
    for name, prediction in predictions.items():
        pred = batch.to_padded(prediction.float(), node_dim=1)     # [K, B, N, P, 2]
        K = pred.shape[0]
        if K > 1 and sampled is None:
            sampled = name
        error = torch.linalg.norm(pred - future[None], dim=-1)    # [K, B, N, P]
        ade, fde = error.mean(-1), error[..., -1]
        per(f"{name}_ade", (ade[0] * cf).sum(-1), agents)
        per(f"{name}_fde", (fde[0] * cf).sum(-1), agents)
        path = torch.cat([last[None, :, :, None].expand(K, -1, -1, -1, -1), pred], dim=3)
        speed = _speeds(path, dt)                                  # [K, B, N, P]
        per(f"{name}_speed", (speed.mean(-1).mean(0) * cf).sum(-1), agents)
        net = torch.linalg.norm(pred[..., -1, :] - last[None], dim=-1) / (pred.shape[3] * dt)   # [K, B, N]
        per(f"{name}_netspeed", (net.mean(0) * cf).sum(-1), agents)
        moving = complete & (gt_speed.mean(-1) > 0.3)
        net_moving = complete & (gt_net > 0.3)
        per(f"{name}_netspeed_ratio_moving", ((net[0] / gt_net.clamp(min=1e-3)) * net_moving).sum(-1),
            net_moving.float().sum(-1))
        per(f"{name}_speed_ratio_moving", ((speed[0].mean(-1) / gt_speed.mean(-1).clamp(min=1e-3)) * moving).sum(-1),
            moving.float().sum(-1))
        for step in range(speed.shape[-1]):
            per(f"{name}_speed_t{step + 1}", (speed[..., step].mean(0) * cf).sum(-1), agents)
        for index in torch.unique(dataset).tolist():
            chosen = (dataset == index)[:, None] & complete
            report.keep(names[index], f"speed_{name}", speed[0][chosen])
        turn, turn_valid = _turning(path[0])
        per(f"{name}_turn", (turn * turn_valid * complete[..., None]).sum((1, 2)),
            (turn_valid * complete[..., None]).sum((1, 2)))
        collisions(name, pred[0], gt_closest)
        if K > 1:
            # Average the collision metrics over all K samples (sample 0 above is one of them).
            for k in range(1, K):
                collisions(f"{name}_k", pred[k], gt_closest)
            mean = pred.mean(0)
            mean_error = torch.linalg.norm(mean - future, dim=-1)
            per(f"{name}_mean_ade", (mean_error.mean(-1) * cf).sum(-1), agents)
            per(f"{name}_mean_fde", (mean_error[..., -1] * cf).sum(-1), agents)
            mean_speed = _speeds(torch.cat([last[:, :, None], mean], dim=2), dt)
            per(f"{name}_mean_speed", (mean_speed.mean(-1) * cf).sum(-1), agents)
            collisions(f"{name}_mean", mean, gt_closest)
            per(f"{name}_minade", (ade.min(0).values * cf).sum(-1), agents)
            per(f"{name}_minfde", (fde.min(0).values * cf).sum(-1), agents)
            scene_ade = (ade * cf).sum(-1) / agents.clamp(min=1)   # [K, B]
            scene_fde = (fde * cf).sum(-1) / agents.clamp(min=1)
            best = scene_ade.argmin(0)
            has = (agents > 0).float()
            per(f"{name}_joint_ade", scene_ade.gather(0, best[None])[0] * has, has)
            per(f"{name}_joint_fde", scene_fde.gather(0, best[None])[0] * has, has)
            # Shuffled joint samples: every agent draws its own sample index.
            generator = torch.Generator(device=pred.device).manual_seed(0)
            choice = torch.randint(0, K, (B, N), device=pred.device, generator=generator)
            shuffled = pred.gather(0, choice[None, :, :, None, None].expand(1, B, N, *pred.shape[3:]))[0]
            collisions(f"{name}_shuffled", shuffled, gt_closest)
            final = pred[..., -1, :]
            spread = sum(torch.linalg.norm(final - final[k:k + 1], dim=-1).sum(0) for k in range(K)) / (K * (K - 1))
            per(f"{name}_diversity_final", (spread * cf).sum(-1), agents)


def main() -> None:
    import yaml

    from baselines.stars_ma_batch import node_budget_batches
    from baselines.stars_ma_graph import to_scene
    from baselines.train_stars_ma import _NullRun, _autocast, _batch, dataset_entries, load_forecaster, load_stores
    from forecasting_config import resolve_forecasting_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="Forecaster checkpoint(s); name=path to label one")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--datasets", default="steps,ptdpes,madras,grand_central,ucy,eth,atc")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--windows", choices=("all", "unique"), default="unique")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=8192)
    parser.add_argument("--max-windows", type=int, help="Seeded cap of windows per dataset (speed)")
    parser.add_argument("--atc-max-windows", type=int, default=10000)
    parser.add_argument("--goals", action="store_true", help="Load goal columns (needed by goal-conditioned checkpoints)")
    parser.add_argument("--oversample", type=int, default=1,
                        help="> 1: draw this many times --samples joint samples per scene and keep the --samples "
                             "with the fewest non-group close pairs (inference-time rejection)")
    parser.add_argument("--temperature", type=float, default=1.0, help="CVAE prior standard-deviation scale")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/.cache/stars_ma"))
    parser.add_argument("--cache-workers", type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    models = {}
    graph = {"knn": 24, "radius": 10.0}
    for item in args.checkpoint:
        name, _, path = str(item).rpartition("=")
        name = name or Path(path).parents[1].name
        model, payload = load_forecaster(Path(path), device)
        model.eval()
        models[name] = model
        graph = payload["graph"]
        args.goals = args.goals or bool(model.decoder_recipe.get("goal"))
        args.goal_beyond_horizon = bool(payload["run_config"].get("goal_beyond_horizon", False))
    if args.config:
        config = yaml.safe_load(args.config.read_text()) or {}
        entries = dataset_entries(config.get("val_datasets") or config["datasets"])
    else:
        entries = dataset_entries(args.datasets)
    for entry in entries:
        if entry["name"] == "atc" and "max_windows" not in entry:
            entry["max_windows"] = {"train": 80000, "val": 10000, "test": 50000}
    args.knn, args.radius = graph["knn"], graph["radius"]
    args.obs, args.pred, args.data_seed = 8, 12, 42
    args.run, args.main, args.world_size, args.rank = _NullRun(), True, 1, 0
    args.max_scenes = None
    forecasting = resolve_forecasting_config(preset="steps_default")
    report = Report()
    started = time.time()
    store = load_stores(entries, args.split, args, forecasting, windows=args.windows)
    if args.max_windows:
        rng = np.random.default_rng(args.seed)
        keep = []
        for index in range(len(store.names)):
            ids = torch.nonzero(store.dataset == index).flatten().numpy()
            keep.append(np.sort(rng.choice(ids, min(len(ids), args.max_windows), replace=False)))
        store = store.select(torch.from_numpy(np.concatenate(keep)))
    store = store.to(device)
    batches = node_budget_batches(store.sizes, args.max_nodes, shuffle=False)
    dt = 0.4
    with torch.no_grad():
        for windows in batches:
            batch = _batch(store, windows, args)
            predictions = {}
            velocity = batch.obs_xy[:, -1] - batch.obs_xy[:, -2]
            horizon = torch.arange(1, args.pred + 1, device=device, dtype=velocity.dtype)
            predictions["cv"] = (batch.obs_xy[:, -1:, :] + horizon[None, :, None] * velocity[:, None])[None]
            for name, model in models.items():
                with _autocast(argparse.Namespace(amp=True, device_realized=device)):
                    if model.latent_dim and args.temperature != 1.0:
                        encoded = model.encode(batch)
                        mu, logvar = model.decoder.prior_params(encoded["context"])
                        z = mu + args.temperature * torch.exp(0.5 * logvar) * torch.randn(
                            (args.samples, *mu.shape), device=mu.device, dtype=mu.dtype)
                        local = model.decode(batch, encoded, z)
                    else:
                        draws = args.samples * args.oversample if model.stochastic else 1
                        local = model.predict(batch, samples=draws)
                    predictions[name] = to_scene(local.float(), batch.origin, batch.heading)
                    if args.oversample > 1 and model.stochastic:
                        predictions[name] = select_scene_samples(predictions[name], batch, args.samples, dt)
                    if model.latent_dim:
                        encoded = model.encode(batch)
                        z = model.sample_latent(encoded["context"], 1, mean=True)
                        predictions[f"{name}_mode"] = to_scene(model.decode(batch, encoded, z).float(),
                                                               batch.origin, batch.heading)
            scene_metrics(report, store.names, batch, predictions, dt)
    result = report.result()
    # Collision metrics averaged over all K samples: sample 0 plus samples 1..K-1.
    for row in result.values():
        for key in list(row):
            for name in models:
                prefix = f"{name}_k_"
                if key.startswith(prefix):
                    base = f"{name}_" + key[len(prefix):]
                    row[f"{name}_allk_" + key[len(prefix):]] = (row[base] + (args.samples - 1) * row[key]) / args.samples
                    del row[key]
    output = {"split": args.split, "windows": args.windows, "samples": args.samples, "temperature": args.temperature,
              "oversample": args.oversample,
              "checkpoints": {name: str(path) for name, path in zip(models, args.checkpoint)},
              "datasets": store.names, "n_windows": len(store), "seconds": time.time() - started, "metrics": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1) + "\n")
    print(json.dumps({k: {m: round(v, 4) for m, v in row.items() if any(
        m.endswith(s) for s in ("_ade", "speed", "_pair0.2", "_pair0.4", "minade"))} for k, row in result.items()
        if k == "all"}, indent=1))


if __name__ == "__main__":
    main()
