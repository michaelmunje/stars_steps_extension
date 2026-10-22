"""Evaluate a multi-agent STARS forecaster: teacher-forced, closed-loop, and long-horizon simulation.

1. Teacher-forced forecasting: best-of-K ADE/FDE (target / all agents / joint), interacting vs
   isolated agents, scene density bins, collision and closest-approach rates, diversity -- pooled
   and per dataset (``<dataset>/<metric>``).
2. Closed-loop rollouts over the same horizon for each ``--rollout-exec-steps`` value (the model
   re-encodes the scene from its own predictions every round), scored with the same metrics.
3. Long-horizon simulation (``--sim-seconds``) past the ground-truth horizon: error is undefined
   there, so it reports stability and realism -- the fraction of agents that ever exceed
   ``--max-plausible-speed``, collision rate and closest approach over the whole rollout, and the
   1-D Wasserstein distance between simulated and real speed / acceleration distributions.

Datasets default to the checkpoint's validation datasets; ``--config`` or ``--datasets`` choose
others. ``--windows all`` (default) scores every benchmark sample, so ``target_*`` is directly
comparable with the single-target baselines on the same split; ``unique`` scores each scene
window once. Data parallel with ``torchrun`` like the trainer.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from baselines.stars_ma_batch import node_budget_batches
from baselines.stars_ma_graph import to_scene
from baselines.stars_ma_metrics import MetricAccumulator, batch_metrics, motion_statistics, wasserstein_1d
from baselines.stars_ma_model import rollout
from baselines.train_stars_ma import (
    _NullRun,
    _autocast,
    _batch,
    dataset_entries,
    load_forecaster,
    load_stores,
    setup_distributed,
)
from forecasting_config import resolve_forecasting_config


def _gather_arrays(values: list[np.ndarray], args) -> np.ndarray:
    values = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    if args.world_size > 1:
        gathered = [None] * args.world_size
        dist.all_gather_object(gathered, values)
        values = np.concatenate(gathered)
    return values


@torch.no_grad()
def simulation_report(model, store, batches, args, *, seconds: float, exec_steps: int, max_speed: float,
                      collision_distance: float) -> dict[str, float]:
    rounds = math.ceil(seconds / model.dt / exec_steps)
    simulated = {"speed": [], "acceleration": []}
    real = {"speed": [], "acceleration": []}
    accumulator = MetricAccumulator()
    for windows in batches:
        batch = _batch(store, windows, args)
        with _autocast(args):
            trajectory = rollout(model, batch, rounds=rounds, exec_steps=exec_steps, knn=args.knn, radius=args.radius)
        path = torch.cat([batch.obs_xy[:, -1:], trajectory.float()], dim=1)
        for key, value in motion_statistics(path, model.dt).items():
            simulated[key].append(value)
        complete = batch.future_valid.all(dim=-1)
        truth = torch.cat([batch.obs_xy[complete, -1:], batch.future[complete]], dim=1)
        for key, value in motion_statistics(truth, model.dt).items():
            real[key].append(value)
        speed = torch.linalg.norm(path[:, 1:] - path[:, :-1], dim=-1) / model.dt
        accumulator.add("implausible_agent_rate", (speed > max_speed).any(dim=-1).sum(), speed.shape[0])
        padded = batch.to_padded(path)                                         # [B, N, T, 2]
        N = padded.shape[1]
        upper = (batch.node_mask[:, :, None] & batch.node_mask[:, None, :]
                 & torch.ones(N, N, dtype=torch.bool, device=padded.device).triu(1))
        closest = torch.full(upper.shape, float("inf"), device=padded.device)
        for t in range(padded.shape[2]):
            closest = torch.minimum(closest, torch.linalg.norm(padded[:, :, None, t] - padded[:, None, :, t], dim=-1))
        accumulator.add("collision_rate", ((closest < collision_distance) & upper).sum(), upper.sum())
        accumulator.add("min_distance", torch.where(upper, closest, 0.0).sum(), upper.sum())
    accumulator.all_reduce()
    report = {f"sim_{key}": value for key, value in accumulator.result().items()}
    for key in simulated:
        sim, gt = _gather_arrays(simulated[key], args), _gather_arrays(real[key], args)
        report[f"sim_{key}_mean"] = float(sim.mean()) if len(sim) else float("nan")
        report[f"real_{key}_mean"] = float(gt.mean()) if len(gt) else float("nan")
        report[f"sim_{key}_wasserstein"] = wasserstein_1d(sim, gt)
    report.update({"sim_seconds": rounds * exec_steps * model.dt, "sim_exec_steps": exec_steps})
    return report


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="YAML whose `val_datasets` (or `datasets`) to evaluate")
    parser.add_argument("--datasets", help="Comma-separated datasets (default: the checkpoint's validation datasets)")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--windows", choices=("all", "unique"), default="all")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/.cache/stars_ma"))
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=16384)
    parser.add_argument("--samples", type=int, default=20, help="K for best-of-K (CVAE)")
    parser.add_argument("--rollout-exec-steps", type=int, nargs="*", default=[1, 4, 12])
    parser.add_argument("--sim-seconds", type=float, default=20.0)
    parser.add_argument("--sim-exec-steps", type=int, default=4)
    parser.add_argument("--max-plausible-speed", type=float, default=3.0, help="m/s")
    parser.add_argument("--collision-distance", type=float, default=0.2, help="m")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_distributed(args)

    torch.manual_seed(args.seed + args.rank)
    model, payload = load_forecaster(args.checkpoint, args.device_realized)
    model.eval()
    graph, window = payload["graph"], payload["window"]
    args.knn, args.radius = graph["knn"], graph["radius"]
    args.obs, args.pred = window["observation_steps"], window["prediction_steps"]
    args.data_seed = payload["run_config"]["data_seed"]
    args.run = _NullRun()
    if args.datasets:
        entries = dataset_entries(args.datasets)
    elif args.config:
        config = yaml.safe_load(args.config.read_text()) or {}
        entries = dataset_entries(config.get("val_datasets") or config["datasets"])
    else:
        entries = payload["datasets"]["val"]
    forecasting = resolve_forecasting_config(preset="steps_default")
    store = load_stores(entries, args.split, args, forecasting, windows=args.windows).to(args.device_realized)
    batches = node_budget_batches(store.sizes, args.max_nodes, shuffle=False)[args.rank::args.world_size]
    if args.max_batches:
        batches = batches[: args.max_batches]
    samples = args.samples if model.latent_dim else 1

    teacher = MetricAccumulator()
    closed = {steps: MetricAccumulator() for steps in args.rollout_exec_steps}
    with torch.no_grad():
        for windows in batches:
            batch = _batch(store, windows, args)
            with _autocast(args):
                local = model.predict(batch, samples=samples)
            batch_metrics(to_scene(local.float(), batch.origin, batch.heading), batch, teacher,
                          collision_distance=args.collision_distance, dataset_names=store.names)
            for steps, accumulator in closed.items():
                rounds = math.ceil(model.prediction_steps / steps)
                with _autocast(args):
                    simulated = torch.stack([
                        rollout(model, batch, rounds=rounds, exec_steps=steps, knn=args.knn, radius=args.radius)
                        [:, :model.prediction_steps] for _ in range(samples)
                    ])
                batch_metrics(simulated.float(), batch, accumulator, collision_distance=args.collision_distance,
                              dataset_names=store.names)
    teacher.all_reduce()
    for accumulator in closed.values():
        accumulator.all_reduce()

    report = {
        "checkpoint": str(args.checkpoint),
        "datasets": store.names, "split": args.split, "windows": args.windows, "k": samples,
        "n_graphs": len(store),
        "teacher_forced": teacher.result(),
        "closed_loop": {f"exec_{steps}": accumulator.result() for steps, accumulator in closed.items()},
    }
    if args.sim_seconds > 0:
        report["simulation"] = simulation_report(
            model, store, batches, args, seconds=args.sim_seconds, exec_steps=args.sim_exec_steps,
            max_speed=args.max_plausible_speed, collision_distance=args.collision_distance)
    if args.main:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        headline = {k: round(v, 4) for k, v in report["teacher_forced"].items()
                    if k.endswith(("agent_ade", "agent_fde", "target_ade", "target_fde"))}
        print(json.dumps({"teacher_forced": headline, **({"simulation": report["simulation"]}
                                                         if "simulation" in report else {})}, indent=1))
    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
