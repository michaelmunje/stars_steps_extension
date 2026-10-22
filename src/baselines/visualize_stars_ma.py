"""Plot multi-agent STARS forecasts on test windows: history, ground truth, samples, constant velocity.

    python -m baselines.visualize_stars_ma --checkpoint experiments/stars_ma/runs/<run>/checkpoints/best.pt \\
        --datasets steps --output docs/results/forecasts_<run>.png

Chooses the windows with the most moving pedestrians (ground-truth mean speed > 0.2 m/s), so the
figure shows interactions rather than a crowd of standing people, and draws a grid of them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from baselines.stars_ma_batch import build_batch
from baselines.stars_ma_graph import to_scene
from baselines.train_stars_ma import _NullRun, dataset_entries, load_forecaster, load_stores, setup_distributed
from forecasting_config import resolve_forecasting_config


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", default="steps")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/.cache/stars_ma"))
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_distributed(args)
    torch.manual_seed(args.seed)
    model, payload = load_forecaster(args.checkpoint, args.device_realized)
    model.eval()
    args.knn, args.radius = payload["graph"]["knn"], payload["graph"]["radius"]
    args.obs, args.pred = payload["window"]["observation_steps"], payload["window"]["prediction_steps"]
    args.data_seed, args.max_scenes, args.run = payload["run_config"]["data_seed"], None, _NullRun()
    store = load_stores(dataset_entries(args.datasets), args.split, args,
                        resolve_forecasting_config(preset="steps_default")).to(args.device_realized)
    dt = payload["window"]["dt"]
    start = store.obs_xy[:, -1]
    path = torch.cat([start[:, None], store.future], dim=1)
    speed = torch.linalg.norm(path[:, 1:] - path[:, :-1], dim=-1).mean(-1) / dt
    moving = ((speed > 0.2) & store.future_valid.all(-1)).float()
    per_window = torch.zeros(len(store), device=moving.device).index_add_(
        0, torch.repeat_interleave(torch.arange(len(store), device=moving.device), store.sizes), moving)
    chosen = torch.argsort(per_window, descending=True)[: args.windows]
    batch = build_batch(store, chosen, dt=dt, knn=args.knn, radius=args.radius)
    with torch.no_grad():
        samples = args.samples if model.latent_dim else 1
        local = model.predict(batch, samples=samples)
        scene = to_scene(local.float(), batch.origin, batch.heading).cpu()
    cv = (batch.obs_xy[:, -1:] + torch.arange(1, args.pred + 1, device=batch.obs_xy.device)[None, :, None]
          * (batch.obs_xy[:, -1] - batch.obs_xy[:, -2])[:, None]).cpu()
    obs, future, valid = batch.obs_xy.cpu(), batch.future.cpu(), batch.future_valid.cpu()
    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    colors = plt.cm.tab20(range(20))
    for g, ax in enumerate(axes.flat):
        if g >= len(chosen):
            ax.axis("off")
            continue
        lo, hi = int(batch.ptr[g]), int(batch.ptr[g + 1])
        for n in range(lo, hi):
            c = colors[(n - lo) % 20]
            ax.plot(obs[n, :, 0], obs[n, :, 1], color=c, lw=1.8)
            if valid[n].all():
                ax.plot(*torch.cat([obs[n, -1:], future[n]]).T, color=c, ls="--", lw=1.2)
            for k in range(scene.shape[0]):
                ax.plot(*torch.cat([obs[n, -1:], scene[k, n]]).T, color=c, lw=0.6, alpha=0.35 if samples > 1 else 0.9)
            ax.plot(*torch.cat([obs[n, -1:], cv[n]]).T, color="0.5", lw=0.6, ls=":")
            ax.plot(*obs[n, -1], "o", color=c, ms=3)
        ax.set_aspect("equal"), ax.grid(alpha=0.3)
        ax.set_title(f"window {int(chosen[g])}: {hi - lo} people, {int(per_window[chosen[g]])} moving", fontsize=9)
    handles = [plt.Line2D([], [], color="k", lw=1.8, label="observed (3.2 s)"),
               plt.Line2D([], [], color="k", ls="--", label="ground truth (4.8 s)"),
               plt.Line2D([], [], color="k", lw=0.6, label=f"STARS ({samples} sample{'s' if samples > 1 else ''})"),
               plt.Line2D([], [], color="0.5", ls=":", label="constant velocity")]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle(f"{args.checkpoint.parents[1].name}: {args.datasets} {args.split}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(args.output)


if __name__ == "__main__":
    main()
