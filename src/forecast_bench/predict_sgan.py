"""Official pretrained Social-GAN (SGAN-20VP-20: variety loss, pooling, 20 samples) on the common windows.

Checkpoints: references/social-gan/scripts/download_models.sh (Dropbox), unzipped under C=/scratch/09865/mmunje/.cache/social-gan (CACHE = $C/models):

    mkdir -p $C && cd $C && wget -O models.zip 'https://www.dropbox.com/s/h8q5z4axfgzx9eb/models.zip?dl=1' && unzip models.zip

    python -m forecast_bench.predict_sgan --sets ethucy_eth,...        # --models sgan-models for SGAN-20V-20

Official evaluation (scripts/evaluate_model.py on CPU via shims):

    python scripts/forecast_bench_upstream/sgan_evaluate.py --model_path $C/models/sgan-p-models/eth_12_model.pt

Note its metric is scene-joint: per window, the min over samples of the SUM of the agents' errors.

As in the official loader the scene is the fully observed agents (``eval_mask``), inputs are absolute
positions plus per-step displacements (first one zero). Noise is ``global``: one noise vector per scene
per forward pass, so sample k is a joint scene sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from forecast_bench.windows import ROOT, load

CACHE = Path("/scratch/09865/mmunje/.cache/social-gan/models")
sys.path.insert(0, str(ROOT / "references/social-gan"))
from sgan.models import TrajectoryGenerator  # noqa: E402


def build(fold: str, models: str = "sgan-p-models", device: str = "cpu") -> TrajectoryGenerator:
    checkpoint = torch.load(CACHE / models / f"{fold}_12_model.pt", map_location="cpu", weights_only=True)
    a = checkpoint["args"]
    generator = TrajectoryGenerator(
        obs_len=a["obs_len"], pred_len=a["pred_len"], embedding_dim=a["embedding_dim"],
        encoder_h_dim=a["encoder_h_dim_g"], decoder_h_dim=a["decoder_h_dim_g"], mlp_dim=a["mlp_dim"],
        num_layers=a["num_layers"], noise_dim=a["noise_dim"], noise_type=a["noise_type"],
        noise_mix_type=a["noise_mix_type"], pooling_type=a["pooling_type"],
        pool_every_timestep=a["pool_every_timestep"], dropout=a["dropout"], bottleneck_dim=a["bottleneck_dim"],
        neighborhood_size=a["neighborhood_size"], grid_size=a["grid_size"], batch_norm=a["batch_norm"])
    generator.load_state_dict(checkpoint["g_state"])
    generator.train()  # as official evaluate_model.py (dropout 0, no batch norm: no effect)
    return generator.to(device)


@torch.no_grad()
def sample(generator, obs: np.ndarray, starts: np.ndarray, k: int) -> np.ndarray:
    """obs [N, 8, 2] world metres, scenes = consecutive blocks starting at ``starts`` -> [N, k, 12, 2]."""
    device = next(generator.parameters()).device
    obs_traj = torch.as_tensor(obs, dtype=torch.float32, device=device).transpose(0, 1).contiguous()  # [8, N, 2]
    obs_rel = torch.zeros_like(obs_traj)
    obs_rel[1:] = obs_traj[1:] - obs_traj[:-1]
    seq_start_end = torch.as_tensor(np.stack([starts, np.r_[starts[1:], len(obs)]], 1), device=device)
    cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **kw: self.to(device)  # upstream hard-codes .cuda() for its states/noise
    try:
        out = [generator(obs_traj, obs_rel, seq_start_end).cumsum(0) + obs_traj[-1] for _ in range(k)]  # relative_to_abs
    finally:
        torch.Tensor.cuda = cuda
    return torch.stack(out).permute(2, 0, 1, 3).cpu().numpy()


def predict(windows: dict, fold: str, models: str = "sgan-p-models", k: int = 20, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    mask = windows["eval_mask"]
    window = np.repeat(np.arange(len(windows["ptr"]) - 1), np.diff(windows["ptr"]))[mask]
    starts = np.flatnonzero(np.r_[True, window[1:] != window[:-1]])
    return sample(build(fold, models), windows["obs"][mask], starts, k).astype(np.float32)


def make_predictor(fold: str, device="cpu", models: str = "sgan-p-models"):
    """f(obs [N, 8, 2], obs_valid [N, 8], k, seed) -> [N, k, 12, 2] for every agent of one scene."""
    from forecast_bench.predict_agentformer import backfill

    generator = build(fold, models, str(device))

    def f(obs: np.ndarray, obs_valid: np.ndarray, k: int = 20, seed: int = 0) -> np.ndarray:
        torch.manual_seed(seed)
        return sample(generator, backfill(obs, obs_valid), np.array([0]), k)
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/forecast_bench")
    parser.add_argument("--models", default="sgan-p-models", choices=["sgan-p-models", "sgan-models"])
    args = parser.parse_args()
    for name in args.sets.split(","):
        windows = load(args.root / "windows" / f"{name}.npz")
        pred = predict(windows, name.removeprefix("ethucy_"), args.models)
        path = args.root / "preds" / name / "sgan_official.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pred=pred)
        print(f"{name}/sgan_official: {pred.shape}", flush=True)


if __name__ == "__main__":
    main()
