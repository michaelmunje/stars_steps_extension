"""Profile multi-agent STARS training steps on one GPU and print where the time goes.

    python scripts/horizon/profile_stars_ma.py --config configs/stars_ma/steps.yaml --phase ssl \
        --run-dir /tmp/profile --max-nodes 16384

Takes the trainer's options; runs a few warm-up steps, then profiles ``--profile-steps`` steps
(CPU + CUDA) and prints the top operators by GPU time, plus a per-stage wall-clock split
(graph build, forward, backward, optimizer). Writes a Chrome trace next to the run directory.
"""
from __future__ import annotations

import sys
import time

import torch

from baselines import train_stars_ma as T


def main() -> None:
    argv = sys.argv[1:]
    steps = 6
    if "--profile-steps" in argv:
        i = argv.index("--profile-steps")
        steps = int(argv[i + 1])
        del argv[i:i + 2]
    args = T.parse_args(argv)
    T.setup_distributed(args)
    args.run = T._NullRun()
    forecasting = T.resolve_forecasting_config(preset=T.DEFAULT_PRESET)
    args.obs, args.pred = forecasting.window.observation_steps, forecasting.window.prediction_steps
    args.encoder_recipe = {**T.ENCODER_RECIPE, **({"use_node_features": False} if args.edge_only else {})}
    args.decoder_recipe = {**T.DECODER_RECIPE, "latent_dim": args.cvae_latent_dim if args.decoder == "cvae" else 0}
    args.recipe = dict(T.SSL_RECIPE if args.phase == "ssl" else T.FORECAST_RECIPE)
    device = args.device_realized
    store = T.load_stores(args.train_entries, "train", args, forecasting).to(device)
    model, groups = T._build_model(args, None)
    model = model.to(device).train()
    T._maybe_compile(model, args)
    optimizer = torch.optim.AdamW(groups(1e-4))
    step = T._Step(model, args)
    batches = T.epoch_batches(store, args, 1, train=True)
    stages = {"build": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}

    def run(windows, timed):
        marks = [time.perf_counter()]
        batch = T._batch(store, windows, args)
        torch.cuda.synchronize(); marks.append(time.perf_counter())
        with T._autocast(args):
            losses = step(batch, 0.5)
        torch.cuda.synchronize(); marks.append(time.perf_counter())
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.cuda.synchronize(); marks.append(time.perf_counter())
        optimizer.step()
        torch.cuda.synchronize(); marks.append(time.perf_counter())
        if timed:
            for key, a, b in zip(stages, marks, marks[1:]):
                stages[key] += b - a
        return batch

    for windows in batches[:5]:
        run(windows, False)
    nodes = edges = 0
    for windows in batches[5:5 + steps]:
        batch = run(windows, True)
        nodes, edges = nodes + batch.num_nodes, edges + batch.edge_index.shape[1]
    total = sum(stages.values())
    print(f"{steps} steps, {nodes / total:.0f} nodes/s, {edges / total:.0f} edges/s, "
          f"{edges / max(nodes, 1):.1f} edges/node")
    print({key: f"{100 * value / total:.1f}%" for key, value in stages.items()})

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        for windows in batches[5:5 + steps]:
            run(windows, False)
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=70))
    trace = args.run_dir.with_suffix(".trace.json")
    trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace))
    print("trace:", trace)


if __name__ == "__main__":
    main()
