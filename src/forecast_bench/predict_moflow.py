"""MoFlow (pinned official teacher model) open-loop predictions on the common windows. Single-agent: each scored
agent is predicted from its own 8 observed steps only.

    python -m forecast_bench.predict_moflow --method moflow_official --sets ethucy_eth,...   (GPU recommended)
    python -m forecast_bench.predict_moflow --method moflow_steps --sets pooled_steps,ethucy_eth,...

* ``moflow_official``: the Hugging Face checkpoint of the set's fold (``ethucy_<fold>`` only), with the official
  ETH 'original' preprocessing of ``eval_eth.py`` (README command): absolute / last-relative / velocity past
  features in world metres, rotated so that the step-6 -> step-7 displacement points along +x
  (``--rotate --rotate_time_frame 6``), output min-max unnormalised with the training split's statistics (computed
  by the official ``ETHDataset``), sampled with 100 ``lin_poly`` steps (p 5, long step 1000).
* ``moflow_steps``: our STEPS checkpoint (``baselines.moflow`` config: no min-max, no rotation flag) in the
  per-target SE(2) frame it was trained in (origin at the last observation, heading of the last observed
  displacement along +y, ``normalization.observed_target_transform``), sampled like its own evaluation
  (``baselines.evaluate_moflow``: 10 Euler steps).

``make_predictor(variant, device)`` returns ``f(obs [N, 8, 2], obs_valid [N, 8], k, seed) -> [N, k, 12, 2]``
(partially observed agents are back-filled, ``predict_socialvae.fill_history``). The model always draws its
20 official samples; ``k`` <= 20 keeps the first ``k``.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch

from baselines import moflow as M
from forecast_bench.predict_socialvae import fill_history
from forecast_bench.windows import FOLDS, ROOT, load
from normalization import observed_target_transform

CKPTS = Path(os.environ.get("MOFLOW_CKPTS", "/scratch/09865/mmunje/.cache/moflow_ckpts/eth_ucy/moflow"))
STEPS = ROOT / "experiments/pretrained/moflow_transfer_v1_steps_0/checkpoints/best.pt"
K = 20


def rotation(theta: np.ndarray) -> np.ndarray:
    """[B] -> [B, 2, 2] matrices of ``rotate_traj`` (rotate by -theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


def _official_eth_dataset():
    path = M.REFERENCE / "data/dataloader_eth_ucy.py"
    with M._scoped_reference_namespace(M.REFERENCE, ("utils",)):
        spec = importlib.util.spec_from_file_location("_moflow_dataloader_eth_ucy", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.ETHDataset


class Predictor:
    def __init__(self, variant: str, device: str = "cpu"):
        self.device, self.official = device, variant.startswith("official")
        if self.official:
            fold = variant.split(":")[1]
            cfg = M.MoFlowConfig(str(CKPTS / fold / "cor_fm.yml"), "bench", train_mode=False)
            cfg.device = device
            cfg.sampling_steps, cfg.solver, cfg.lin_poly_p, cfg.lin_poly_long_step = 100, "lin_poly", 5, 1000
            # sets cfg.agents = 1 and the train-split min/max used to unnormalise the output
            _official_eth_dataset()(cfg, training=True, data_dir=str(M.REFERENCE / "data/eth_ucy"), subset=fold,
                                    rotate_time_frame=6, type="original")
            payload = torch.load(CKPTS / fold / "models/checkpoint_best.pt", map_location="cpu", weights_only=True)
            logger = logging.getLogger("moflow-bench")
            self.model = M.FlowMatcher(cfg, M.ETHMotionTransformer(cfg.MODEL, logger, cfg), logger)
        else:
            cfg, self.model = M.build(torch.device(device))
            payload = torch.load(STEPS, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.model.to(device).eval()
        self.cfg = cfg

    def to_model(self, past: np.ndarray):
        """[B, 8, 2] -> (past abs, rotation matrix to model frame [B, 2, 2], origin [B, 2])."""
        if self.official:
            origin = past[:, -1]
            diff = past[:, self.cfg.rotate_time_frame] - origin     # official: past_rel[:, rotate_time_frame]
            theta = np.arctan(diff[:, 1] / (diff[:, 0] + 1e-5))
            theta = np.where(diff[:, 0] < 0, theta + math.pi, theta)
            R = rotation(theta) if self.cfg.rotate else np.broadcast_to(np.eye(2), (len(past), 2, 2))
            return np.einsum("bij,btj->bti", R, past), R, origin   # absolute positions rotated about the world origin
        transforms = [observed_target_transform(p[:, None], 8) for p in past]
        origin = np.stack([t.center for t in transforms])
        R = rotation(-np.array([t.rotation for t in transforms]))   # theta_rotation(xy, r) rotates by +r
        return np.einsum("bij,btj->bti", R, past - origin[:, None]), R, origin

    @torch.no_grad()
    def __call__(self, past: np.ndarray, k: int = K, seed: int = 0, batch: int = 1000) -> np.ndarray:
        """Fully observed tracks [B, 8, 2] -> [B, k, 12, 2] in the input frame."""
        torch.manual_seed(seed)
        out = []
        for a in range(0, len(past), batch):
            p = np.asarray(past[a:a + batch], dtype=np.float64)
            abs_, R, origin = self.to_model(p)
            rel = abs_ - abs_[:, -1:]
            vel = np.concatenate([rel[:, 1:] - rel[:, :-1], np.zeros_like(rel[:, :1])], 1)
            feats = torch.as_tensor(np.concatenate([abs_, rel, vel], -1)[:, None], dtype=torch.float32, device=self.device)
            fut = torch.zeros(len(p), 1, 12, 2, device=self.device)
            data = {"batch_size": torch.tensor(len(p)), "past_traj_original_scale": feats, "past_traj": feats,
                    "fut_traj": fut, "fut_traj_original_scale": fut, "fut_traj_vel": fut}
            if self.cfg.get("data_norm") == "min_max":
                norm = lambda t, lo, hi: (t - lo) / (hi - lo) * 2 - 1
                data["past_traj"] = norm(feats, self.cfg.past_traj_min, self.cfg.past_traj_max)
                data["fut_traj"] = norm(fut, self.cfg.fut_traj_min, self.cfg.fut_traj_max)
            pred = M.sample_predictions(self.model, data, K)[:, :k]          # [B, k, 12, 2] model frame
            if self.cfg.get("data_norm") == "min_max":
                lo, hi = self.cfg.fut_traj_min, self.cfg.fut_traj_max
                pred = (pred + 1) / 2 * (hi - lo) + lo
            pred = pred.double().cpu().numpy()
            out.append(np.einsum("bji,bktj->bkti", R, pred) + p[:, -1, None, None])  # R^T, back to the input frame
        return np.concatenate(out).astype(np.float32)


def make_predictor(variant: str, device: str = "cpu"):
    """``variant``: ``official:<fold>`` or ``steps``; returns f(obs, obs_valid, k, seed) -> [N, k, 12, 2]."""
    predictor = Predictor(variant, device)

    def predict(obs: np.ndarray, obs_valid: np.ndarray, k: int = K, seed: int = 0) -> np.ndarray:
        return predictor(fill_history(np.asarray(obs), np.asarray(obs_valid, dtype=bool)), k, seed)
    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=["moflow_official", "moflow_steps"])
    parser.add_argument("--sets", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/forecast_bench")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    predictors = {}
    for name in args.sets.split(","):
        if args.method == "moflow_official":
            fold = name.removeprefix("ethucy_")
            if fold not in FOLDS:
                raise SystemExit(f"moflow_official has checkpoints for ethucy_<fold> sets only, not {name}")
            variant = f"official:{fold}"
        else:
            variant = "steps"
        if variant not in predictors:
            predictors[variant] = Predictor(variant, args.device)
        windows = load(args.root / "windows" / f"{name}.npz")
        pred = predictors[variant](windows["obs"][windows["eval_mask"]], K, args.seed)
        path = args.root / "preds" / name / f"{args.method}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pred=pred)
        print(f"{name}/{args.method}: {pred.shape}", flush=True)


if __name__ == "__main__":
    main()
