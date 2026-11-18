"""Official pretrained AgentFormer (DLow, K = 20) on the common windows.

Checkpoints: ``agentformer_models.zip`` (README, Google Drive id 1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB)
unzipped under WORK/.. (``results/<fold>_agentformer{,_pre}/models``); WORK links cfg, datasets and
results like the upstream repo root, because upstream Config/DLow resolve those relative to the cwd:

    A=/scratch/09865/mmunje/.cache/agentformer; R=references/agentformer
    gdown 1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB -O $A/agentformer_models.zip && (cd $A && unzip agentformer_models.zip)
    mkdir -p $A/work && for d in cfg datasets eval.py data utils model; do ln -s $PWD/$R/$d $A/work/$d; done
    ln -s $A/results $A/work/results

    python -m forecast_bench.predict_agentformer --sets ethucy_eth,...

As in the official test, the scene is the fully observed agents (``eval_mask``), coordinates are divided
by the config's ``traj_scale`` and the model centres the scene itself. DLow's 20 samples are joint:
sample k is one coherent scene future.
"""
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import torch

from baselines.agentformer import DLow
from forecast_bench.windows import OBS, PRED, ROOT, load

WORK = Path("/scratch/09865/mmunje/.cache/agentformer/work")
Config = DLow.__init__.__globals__["Config"]  # upstream utils.config.Config, as imported by model/dlow.py


@contextlib.contextmanager
def _upstream_cwd():
    old, load_ = os.getcwd(), torch.load
    os.chdir(WORK)
    # DLow loads its (official, 2021-pickled) predictor checkpoint with a bare torch.load
    torch.load = lambda *a, **k: load_(*a, **{"map_location": "cpu", "weights_only": False, **k})
    try:
        yield
    finally:
        os.chdir(old)
        torch.load = load_


def build(fold: str, device: str = "cpu"):
    with _upstream_cwd():
        cfg = Config(f"{fold}_agentformer")
        model = DLow(cfg)
        checkpoint = torch.load(cfg.model_path % cfg.get_last_epoch())
    model.load_state_dict(checkpoint["model_dict"], strict=False)  # as upstream test.py
    model.set_device(torch.device(device))
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_scene(model, cfg, obs: np.ndarray, obs_valid: np.ndarray) -> np.ndarray:
    """obs [N, 8, 2] world metres (missing history already back-filled) -> [N, 20, 12, 2]."""
    pre = torch.as_tensor(obs, dtype=torch.float32) / cfg.traj_scale
    # the future is not used at inference; give the model the last observation, never GT
    fut = pre[:, -1:].expand(-1, PRED, -1)
    n = len(obs)
    data = {"pre_motion_3D": list(pre), "fut_motion_3D": list(fut),
            "pre_motion_mask": list(torch.as_tensor(obs_valid, dtype=torch.float32)),
            "fut_motion_mask": list(torch.ones(n, PRED)), "heading": None,
            "traj_scale": cfg.traj_scale, "pred_mask": None, "scene_map": None}
    model.set_data(data)
    sample, _ = model.inference(mode="infer", sample_num=cfg.sample_k, need_weights=False)  # [N, K, 12, 2]
    return (sample * cfg.traj_scale).cpu().numpy()


def predict(windows: dict, fold: str, device: str = "cpu", seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    model, cfg = build(fold, device)
    ptr, mask = windows["ptr"], windows["eval_mask"]
    out = []
    for w in range(len(ptr) - 1):
        scored = np.flatnonzero(mask[ptr[w]:ptr[w + 1]]) + ptr[w]   # scored nodes come first in a window
        out.append(predict_scene(model, cfg, windows["obs"][scored], windows["obs_valid"][scored]))
    return np.concatenate(out).astype(np.float32)


def make_predictor(fold: str, device="cpu"):
    """f(obs [N, 8, 2], obs_valid [N, 8], k, seed) -> [N, k, 12, 2] for every agent of one scene (k <= 20)."""
    model, cfg = build(fold, str(device))

    def f(obs: np.ndarray, obs_valid: np.ndarray, k: int = 20, seed: int = 0) -> np.ndarray:
        assert k <= cfg.sample_k, "DLow outputs a fixed set of 20 samples"
        obs = backfill(obs, obs_valid)
        return predict_scene(model, cfg, obs, obs_valid)[:, :k]   # DLow is deterministic given the scene
    return f


def backfill(obs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill each agent's missing history with its next observed position (first observed one before it)."""
    obs = obs.copy()
    for t in range(OBS - 2, -1, -1):
        missing = ~valid[:, t]
        obs[missing, t] = obs[missing, t + 1]
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/forecast_bench")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for name in args.sets.split(","):
        windows = load(args.root / "windows" / f"{name}.npz")
        pred = predict(windows, name.removeprefix("ethucy_"), args.device)
        path = args.root / "preds" / name / "agentformer_official.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pred=pred)
        print(f"{name}/agentformer_official: {pred.shape}", flush=True)


if __name__ == "__main__":
    main()
