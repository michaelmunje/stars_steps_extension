"""Open-loop predictions of constant velocity and STARS checkpoints on the common windows.

    python -m forecast_bench.predict --method cv --sets ethucy_eth,...
    python -m forecast_bench.predict --method stars_pooled --sets ... --device cuda

``<stars method>_mode`` (e.g. ``stars_pooled_mode``) decodes the CVAE prior mean (z = mu, no sampling):
one deterministic prediction (K = 1), the fair comparison to deterministic baselines such as CV.

Writes ``<root>/preds/<set>/<method>.npz`` (``pred [E, K, 12, 2]`` for the scored nodes, see
``forecast_bench.metrics``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from forecast_bench.windows import PRED, ROOT, load

RUNS = ROOT / "experiments/stars_ma/runs"
STARS = {  # method -> checkpoint
    "stars_steps": RUNS / "cvae_kl01_finetune/checkpoints/best.pt",
    "stars_pooled": RUNS / "cvae_all_lr3e-4_seed42/checkpoints/best.pt",
    "stars_noethucy": RUNS / "cvae_noethucy_nogoal/checkpoints/best.pt",
    "stars_noethucy_goal": RUNS / "cvae_noethucy_goal/checkpoints/best.pt",
    "stars_steps_diffusion": RUNS / "diffusion_finetune/checkpoints/best.pt",
    "stars_steps_joint_diffusion": RUNS / "joint_diffusion_finetune/checkpoints/best.pt",
    # branch stars-improve: all-7 pooled recipe (seed 2, best of 3 on validation), 4x joint-sample
    # oversampling with collision-based selection; needs stars-improve's src first on PYTHONPATH
    "stars_improved": RUNS / "si_final_base_s2/checkpoints/best_os4.pt",
    # collision-aware variant: refinement round + decoder collision hinge (weight 10), KL 0.15
    "stars_improved_social": RUNS / "si_final_social_s42/checkpoints/best.pt",
}
# ETH/UCY protocol (branch ethucy-protocol): ep_zs_s<seed> = protocol A (pretrained on STEPS, PTDPES, MADRAS,
# Grand Central; zero-shot), ep_loso_s<seed> = protocol B (A fine-tuned leave-one-scene-out, per-fold checkpoint
# chosen on the fold's val split), ep_scratch_s<seed> = LOSO from scratch (no pretraining). Suffix _os4: 4x
# joint-sample oversampling with collision-based selection; _mode: prior-mean latent (K = 1).
PROTOCOL_PRE = RUNS / "ep_pre4_s{seed}/checkpoints/best.pt"


def protocol_checkpoint(method: str, fold: str, root: Path) -> tuple[Path, bool, int]:
    """``ep_<kind>_s<seed>[_os4|_mode]`` -> (checkpoint, mode, oversample) for one ETH/UCY fold. Kinds: zs, zs_rft
    (pretraining run), loso, loso_rft, scratch (per-fold checkpoints under ``<root>/loso``)."""
    import re
    kind, seed, suffix = re.fullmatch(r"ep_([a-z_]+?)_s(\d+)(_os4|_mode)?", method).groups()
    mode, oversample = suffix == "_mode", 4 if suffix == "_os4" else 1
    if kind == "zs":
        return Path(str(PROTOCOL_PRE).format(seed=seed)), mode, oversample
    if kind == "zs_rft":
        return RUNS / f"ep_rft_zs_s{seed}/checkpoints/best.pt", mode, oversample
    return root / "loso" / f"ep_{kind}_s{seed}" / fold / "best.pt", mode, oversample


MODE = {f"{m}_mode": m for m in ("stars_pooled", "stars_noethucy", "stars_noethucy_goal", "stars_improved", "stars_improved_social")}


def constant_velocity(windows: dict) -> np.ndarray:
    obs = windows["obs"][windows["eval_mask"]]
    velocity = obs[:, -1] - obs[:, -2]
    steps = np.arange(1, PRED + 1, dtype=np.float32)
    return (obs[:, -1, None] + steps[:, None] * velocity[:, None])[:, None]


def to_store(windows: dict, device):
    from baselines.stars_ma_batch import WindowStore

    t = lambda key, dtype=None: torch.as_tensor(windows[key], dtype=dtype)
    W = len(windows["ptr"]) - 1
    return WindowStore(t("obs"), t("obs_valid"), t("fut"), t("fut_valid"), t("ptr", torch.long),
                       torch.zeros(W, dtype=torch.long), torch.arange(W), ["set"], goal=t("goal")).to(device)


@torch.no_grad()
def stars(windows: dict, checkpoint: Path, device: str, samples: int, max_nodes: int, seed: int = 0,
          mode: bool = False, oversample: int = 1) -> np.ndarray:
    from baselines.stars_ma_batch import _ranges, build_batch, node_budget_batches
    from baselines.stars_ma_graph import to_scene
    from baselines.train_stars_ma import load_forecaster
    from data import CANONICAL_DT

    torch.manual_seed(seed)
    model, payload = load_forecaster(checkpoint, torch.device(device))
    model.eval()
    if oversample > 1:
        model.decoder_recipe["oversample"] = oversample
    graph = payload["graph"]
    store = to_store(windows, device)
    k = samples if model.stochastic and not mode else 1
    out = np.zeros((len(windows["obs"]), k, PRED, 2), dtype=np.float32)
    for batch_windows in node_budget_batches(store.sizes, max_nodes, shuffle=False):
        batch_windows = batch_windows.to(store.device)
        batch = build_batch(store, batch_windows, dt=CANONICAL_DT, knn=graph["knn"], radius=graph["radius"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            if mode:    # prior-mean latent: one deterministic prediction
                encoded = model.encode(batch)
                local = model.decode(batch, encoded, model.sample_latent(encoded["context"], 1, mean=True))
            else:
                local = model.predict(batch, samples=k)
        scene = to_scene(local.float(), batch.origin, batch.heading)          # [K, n, P, 2]
        nodes = _ranges(store.ptr[batch_windows], store.sizes[batch_windows])
        out[nodes.cpu().numpy()] = scene.transpose(0, 1).cpu().numpy()
    return out[windows["eval_mask"]]


def stars_predictor(checkpoint: Path, device: str):
    """Scene-level STARS: ``f(obs [N, 8, 2], obs_valid [N, 8], k, seed, goal=None) -> [N, k, 12, 2]``.

    Agents that are not STARS nodes (not visible at the last step or < 2 observed points) are held still.
    ``goal [N, 2]`` is the oracle goal, used only by goal-conditioned checkpoints.
    """
    from baselines.stars_ma_batch import graph_batch_from_padded
    from baselines.stars_ma_graph import to_scene
    from baselines.train_stars_ma import load_forecaster
    from data import CANONICAL_DT

    model, payload = load_forecaster(Path(checkpoint), torch.device(device))
    model.eval()
    graph = payload["graph"]
    use_goal = bool(model.decoder_recipe.get("goal"))

    @torch.no_grad()
    def predict(obs, obs_valid, k: int = 1, seed: int = 0, goal=None) -> np.ndarray:
        torch.manual_seed(seed)
        node = obs_valid[:, -1] & (obs_valid.sum(1) >= 2)
        out = np.repeat(obs[:, None, -1:], PRED, axis=2).repeat(k, axis=1).astype(np.float32)
        if not node.any():
            return out
        xy = torch.as_tensor(np.nan_to_num(obs[node]), dtype=torch.float32, device=device)[None]
        valid = torch.as_tensor(obs_valid[node], device=device)[None]
        goals = None
        if use_goal and goal is not None:
            g = torch.as_tensor(goal[node], dtype=torch.float32, device=device)
            goals = torch.cat([g, torch.ones_like(g[:, :1])], -1)[None]
        batch = graph_batch_from_padded(xy, valid, torch.ones_like(valid[..., 0]), dt=CANONICAL_DT, knn=graph["knn"],
                                        radius=graph["radius"], prediction_steps=PRED, goal=goals)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            local = model.predict(batch, samples=k if model.stochastic else 1)
        scene = to_scene(local.float(), batch.origin, batch.heading).transpose(0, 1).cpu().numpy()
        out[node] = scene if scene.shape[1] == k else scene.repeat(k, axis=1)
        return out

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, help="cv, stars_loo, a STARS / MODE key, or ep_<zs|loso|scratch>_s<seed>[_os4|_mode]")
    parser.add_argument("--sets", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/forecast_bench")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=8192)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    for name in args.sets.split(","):
        path = args.root / "preds" / name / f"{args.method}.npz"
        if args.skip_existing and path.exists():
            continue
        windows = load(args.root / "windows" / f"{name}.npz")
        if args.method == "cv":
            pred = constant_velocity(windows)
        elif args.method == "stars_loo":   # per-fold checkpoint from forecast_bench.finetune_stars_loo
            checkpoint = args.root / "stars_loo" / name.removeprefix("ethucy_") / "best.pt"
            pred = stars(windows, checkpoint, args.device, args.samples, args.max_nodes)
        elif args.method.startswith("ep_"):
            checkpoint, mode, oversample = protocol_checkpoint(args.method, name.removeprefix("ethucy_"), args.root)
            pred = stars(windows, checkpoint, args.device, args.samples, args.max_nodes, mode=mode,
                         oversample=oversample)
        elif args.method in MODE:
            pred = stars(windows, STARS[MODE[args.method]], args.device, args.samples, args.max_nodes, mode=True)
        else:
            pred = stars(windows, STARS[args.method], args.device, args.samples, args.max_nodes)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez(tmp, pred=pred.astype(np.float32))
        tmp.replace(path)   # atomic: an interrupted run never leaves a partial file behind
        print(f"{name}/{args.method}: {pred.shape}", flush=True)


if __name__ == "__main__":
    main()
