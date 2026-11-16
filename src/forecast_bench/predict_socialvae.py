"""SocialVAE (pinned official model) open-loop predictions on the common windows.

    python -m forecast_bench.predict_socialvae --method socialvae_official --sets ethucy_eth,ethucy_hotel,...
    python -m forecast_bench.predict_socialvae --method socialvae_steps --sets pooled_steps,ethucy_eth,...

* ``socialvae_official``: the official ETH/UCY checkpoint of the set's fold (``ethucy_<fold>`` only).
* ``socialvae_steps``: our STEPS-pretrained checkpoint (trained in world frame with random rotation/flip/scale, so
  it is fed the window frame as is; the model is translation invariant).

Inputs follow the official loader: the target's 8 observed steps as (pos, vel, acc) with per-step differences
(first step of a track uses the forward difference), every other node of the window as a neighbour with the same
state and ``1e9`` where it is not observed (only the 8 observed steps are given; the model never reads more),
neighbours seen at only isolated single steps dropped, and a single ``1e9`` dummy when there is none.
Plain sampling (no FPC). Differences to the official loader: velocity/acceleration start inside the window (the
official loader differences over the whole file, so step-0 velocity uses the frame before the window), and
neighbours that left before the last observed step are not window nodes.

``make_predictor(variant, device)`` returns ``f(obs [N, 8, 2], obs_valid [N, 8], k, seed) -> [N, k, 12, 2]``
predicting every agent of one scene (partially observed agents are back-filled for their own history).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.social_vae import build
from forecast_bench.windows import FOLDS, ROOT, load
from forecasting_representation import _derivatives

PAD = 1e9
OFFICIAL = ROOT / "references/social-vae/models"
STEPS = ROOT / "experiments/pretrained/socialvae_transfer_v1_steps_0/checkpoints/best.pt"


def load_model(variant: str, device: str) -> torch.nn.Module:
    """``official:<fold>`` or ``steps``."""
    if variant == "steps":
        payload = torch.load(STEPS, map_location="cpu", weights_only=False)
        arch = payload["architecture"]
        model = build(torch.device(device), prediction_steps=arch["prediction_steps"],
                      neighborhood_radius=arch["neighborhood_radius"], hidden_dim=arch["hidden_dim"])
    else:
        fold = variant.split(":")[1]
        payload = torch.load(OFFICIAL / {"zara1": "zara01", "zara2": "zara02"}.get(fold, fold) / "ckpt-best",
                             map_location="cpu", weights_only=False)
        model = build(torch.device(device))  # official configs: OB_RADIUS 2, RNN_HIDDEN_DIM 256, PRED_HORIZON 12
    model.load_state_dict(payload["model"])
    return model.eval()


def fill_history(obs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate gaps and hold the first/last observed position (agents with >= 1 observed step)."""
    out = np.array(obs, dtype=np.float32, copy=True)
    t = np.arange(obs.shape[1])
    for i in np.flatnonzero(~valid.all(1) & valid.any(1)):
        for d in range(2):
            out[i, :, d] = np.interp(t, t[valid[i]], obs[i, valid[i], d])
    return out


def state(obs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[N, 8, 2] -> official (x, y, vx, vy, ax, ay) state [8, N, 6], 1e9 where not observed."""
    xy = obs.transpose(1, 0, 2).astype(np.float32)
    v = valid.T
    vel, acc = _derivatives(xy, v, "per_step_difference", 0.4)
    out = np.concatenate([xy, vel, acc], -1)
    out[~v] = PAD
    return out


def scene_inputs(obs: np.ndarray, valid: np.ndarray, targets: np.ndarray):
    """Per target: history [8, 6] and neighbours [8, Nn, 6]."""
    neighbour_state = state(obs, valid)
    usable = (valid[:, 1:] & valid[:, :-1]).any(1)      # official loader drops single-frame appearances
    own = state(fill_history(obs, valid), np.ones_like(valid))
    for i in targets:
        others = np.flatnonzero(usable & (np.arange(len(obs)) != i))
        neighbours = neighbour_state[:, others] if len(others) else np.full((obs.shape[1], 1, 6), PAD, np.float32)
        yield own[:, i], neighbours


@torch.no_grad()
def predict_scenes(model, scenes, k: int, seed: int, device: str, batch: int = 512) -> np.ndarray:
    """scenes: iterable of (obs, valid, targets) -> [sum(targets), k, 12, 2]."""
    torch.manual_seed(seed)
    items = [item for obs, valid, targets in scenes for item in scene_inputs(obs, valid, targets)]
    out = []
    for a in range(0, len(items), batch):
        chunk = items[a:a + batch]
        nn = max(n.shape[1] for _, n in chunk)
        x = torch.as_tensor(np.stack([h for h, _ in chunk], 1), device=device)
        neighbour = np.full((chunk[0][1].shape[0], len(chunk), nn, 6), PAD, np.float32)
        for j, (_, n) in enumerate(chunk):
            neighbour[:, j, :n.shape[1]] = n
        pred = model(x, torch.as_tensor(neighbour, device=device), n_predictions=k)  # [k, 12, B, 2]
        out.append(pred.permute(2, 0, 1, 3).cpu().numpy())
    return np.concatenate(out)


def make_predictor(variant: str, device: str = "cpu"):
    """``variant``: ``official:<fold>`` or ``steps``; returns f(obs, obs_valid, k, seed) -> [N, k, 12, 2]."""
    model = load_model(variant, device)

    def predict(obs: np.ndarray, obs_valid: np.ndarray, k: int = 20, seed: int = 0) -> np.ndarray:
        obs_valid = np.asarray(obs_valid, dtype=bool)
        return predict_scenes(model, [(np.asarray(obs), obs_valid, np.arange(len(obs)))], k, seed, device)
    return predict


def predict_windows(model, windows: dict, k: int, seed: int, device: str) -> np.ndarray:
    ptr, mask = windows["ptr"], windows["eval_mask"]
    scenes = ((windows["obs"][a:b], windows["obs_valid"][a:b], np.flatnonzero(mask[a:b])) for a, b in zip(ptr[:-1], ptr[1:]))
    return predict_scenes(model, scenes, k, seed, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=["socialvae_official", "socialvae_steps"])
    parser.add_argument("--sets", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/forecast_bench")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    models = {}
    for name in args.sets.split(","):
        if args.method == "socialvae_official":
            fold = name.removeprefix("ethucy_")
            if fold not in FOLDS:
                raise SystemExit(f"socialvae_official has checkpoints for ethucy_<fold> sets only, not {name}")
            variant = f"official:{fold}"
        else:
            variant = "steps"
        if variant not in models:
            models[variant] = load_model(variant, args.device)
        pred = predict_windows(models[variant], load(args.root / "windows" / f"{name}.npz"), args.samples, args.seed, args.device)
        path = args.root / "preds" / name / f"{args.method}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pred=pred.astype(np.float32))
        print(f"{name}/{args.method}: {pred.shape}", flush=True)


if __name__ == "__main__":
    main()
