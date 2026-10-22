"""Evaluate a multi-agent STARS forecaster: teacher-forced, closed-loop, and long-horizon simulation.

1. Teacher-forced forecasting: best-of-K ADE/FDE (target / all agents / joint), interacting vs
   isolated agents, scene density bins, collision and closest-approach rates, diversity.
2. Closed-loop rollouts over the same horizon for each ``--rollout-exec-steps`` value (the model
   re-encodes the scene from its own predictions every round), scored with the same metrics.
3. Long-horizon simulation (``--sim-seconds``) past the ground-truth horizon: error is undefined
   there, so it reports stability and realism -- the fraction of agents that ever exceed
   ``--max-plausible-speed``, collision rate and closest approach over the whole rollout, and the
   1-D Wasserstein distance between simulated and real STEPS speed / acceleration distributions.

``--windows all`` (default) scores every benchmark sample, so ``target_*`` is directly comparable
with the single-target baselines on the same split; ``unique`` scores each scene window once.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from baselines.device import training_device
from baselines.dataset_specs import pretrain_specs
from baselines.stars_ma_data import StarsMultiAgentDataset
from baselines.stars_ma_graph import to_scene
from baselines.stars_ma_metrics import (
    MetricAccumulator,
    forecasting_metrics,
    motion_statistics,
    pairwise_stats,
    wasserstein_1d,
)
from baselines.stars_ma_model import rollout
from baselines.train_stars_ma import _limited, load_forecaster
from config import Config, SeedConfig
from data import SceneDataset
from forecasting_config import resolve_forecasting_config


@torch.no_grad()
def simulation_report(model, loader, device, *, seconds: float, exec_steps: int, knn, radius,
                      max_speed: float, collision_distance: float, max_batches=None) -> dict[str, float]:
    rounds = math.ceil(seconds / model.dt / exec_steps)
    simulated = {"speed": [], "acceleration": []}
    real = {"speed": [], "acceleration": []}
    accumulator = MetricAccumulator()
    for batch in _limited(loader, max_batches):
        batch = batch.to(device)
        trajectory = rollout(model, batch, rounds=rounds, exec_steps=exec_steps, knn=knn, radius=radius)
        path = torch.cat([batch.obs_xy[:, -1:], trajectory], dim=1)
        for key, value in motion_statistics(path, model.dt).items():
            simulated[key].append(value)
        complete = batch.future_valid.all(dim=-1)
        truth = torch.cat([batch.obs_xy[complete, -1:], batch.future[complete]], dim=1)
        for key, value in motion_statistics(truth, model.dt).items():
            real[key].append(value)
        speed = torch.linalg.norm(path[:, 1:] - path[:, :-1], dim=-1) / model.dt
        accumulator.add("implausible_agent_rate", (speed > max_speed).any(dim=-1).sum(), speed.shape[0])
        pointer = batch.ptr.tolist()
        for start, stop in zip(pointer[:-1], pointer[1:]):
            collided, closest, pairs = pairwise_stats(path[start:stop], collision_distance)
            accumulator.add("collision_rate", collided, pairs)
            accumulator.add("min_distance", closest, pairs)
    report = {f"sim_{key}": value for key, value in accumulator.result().items()}
    for key in simulated:
        sim, gt = np.concatenate(simulated[key]), np.concatenate(real[key])
        report[f"sim_{key}_mean"] = float(sim.mean())
        report[f"real_{key}_mean"] = float(gt.mean())
        report[f"sim_{key}_wasserstein"] = wasserstein_1d(sim, gt)
    report.update({"sim_seconds": rounds * exec_steps * model.dt, "sim_exec_steps": exec_steps})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["steps"])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--windows", choices=("all", "unique"), default="all")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=20, help="K for best-of-K (CVAE)")
    parser.add_argument("--rollout-exec-steps", type=int, nargs="*", default=[1, 4, 12])
    parser.add_argument("--sim-seconds", type=float, default=20.0)
    parser.add_argument("--sim-exec-steps", type=int, default=4)
    parser.add_argument("--max-plausible-speed", type=float, default=3.0, help="m/s")
    parser.add_argument("--collision-distance", type=float, default=0.2, help="m")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from torch_geometric.loader import DataLoader

    torch.manual_seed(args.seed)
    device = training_device(args.device)
    model, payload = load_forecaster(args.checkpoint, device)
    model.eval()
    graph, window = payload["graph"], payload["window"]
    forecasting = resolve_forecasting_config(preset="steps_default")
    cfg = Config(seeds=SeedConfig(payload["run_config"]["data_seed"], args.seed), forecasting=forecasting)
    sources = [SceneDataset(spec, args.split, cfg) for spec in pretrain_specs(tuple(args.datasets))]
    source = sources[0] if len(sources) == 1 else torch.utils.data.ConcatDataset(sources)
    dataset = StarsMultiAgentDataset(source, window["observation_steps"], window["prediction_steps"], dt=window["dt"],
                                     knn=graph["knn"], radius=graph["radius"], windows=args.windows, limit=args.max_scenes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    samples = args.samples if model.latent_dim else 1

    teacher = MetricAccumulator()
    closed = {steps: MetricAccumulator() for steps in args.rollout_exec_steps}
    with torch.no_grad():
        for batch in _limited(loader, args.max_batches):
            batch = batch.to(device)
            local = model.predict(batch, samples=samples)
            forecasting_metrics(to_scene(local, batch.origin, batch.heading), batch, teacher,
                                collision_distance=args.collision_distance)
            for steps, accumulator in closed.items():
                rounds = math.ceil(model.prediction_steps / steps)
                simulated = torch.stack([
                    rollout(model, batch, rounds=rounds, exec_steps=steps, knn=graph["knn"], radius=graph["radius"])
                    [:, :model.prediction_steps] for _ in range(samples)
                ])
                forecasting_metrics(simulated, batch, accumulator, collision_distance=args.collision_distance)

    report = {
        "checkpoint": str(args.checkpoint),
        "datasets": args.datasets, "split": args.split, "windows": args.windows, "k": samples,
        "n_graphs": len(dataset),
        "teacher_forced": teacher.result(),
        "closed_loop": {f"exec_{steps}": accumulator.result() for steps, accumulator in closed.items()},
    }
    if args.sim_seconds > 0:
        report["simulation"] = simulation_report(
            model, loader, device, seconds=args.sim_seconds, exec_steps=args.sim_exec_steps, knn=graph["knn"],
            radius=graph["radius"], max_speed=args.max_plausible_speed, collision_distance=args.collision_distance,
            max_batches=args.max_batches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"teacher_forced": {k: round(v, 4) for k, v in report["teacher_forced"].items()},
                      **({"simulation": report["simulation"]} if "simulation" in report else {})}, indent=1))


if __name__ == "__main__":
    main()
