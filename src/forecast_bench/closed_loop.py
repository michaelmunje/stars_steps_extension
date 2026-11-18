"""Secondary benchmark: closed-loop (autoregressive) long-horizon rollouts on ETH/UCY, up to 20 s.

Reuses the closed-loop windows and metrics of ``simplenavsim.realism`` (``docs/ethucy_benchmark.md``):
every window's cohort (everyone present at the last observed step) is simulated jointly from 8
observed steps, here for up to 50 steps (20 s), and scored at 4.8, 8, 12, 16 and 20 s on agents whose
recorded track covers the horizon. ``MAX_STEPS``/``HORIZONS`` are extended for this script only.

Rollout schemes (one sample per seed, no best-of-K):
* ``<forecaster>@<s>``: receding horizon for any fixed-horizon forecaster (STARS, SocialVAE, MoFlow,
  ...): predict 12 steps from the current 8-step window (one sample), execute the first ``s`` steps for
  every agent, append them to the history, repeat. Default ``s = 4`` (1.6 s); missing history steps
  stay masked until they leave the window.
* ``stars_*_native``: ``simplenavsim.realism.sim.Stars``, STARS' own closed loop (re-forecast every
  0.4 s, execute one step, temporally correlated CVAE intent).
* ``cv``: constant velocity (open loop, same as closed loop).

    python -m forecast_bench.closed_loop --methods cv,stars_pooled@4,socialvae_official@4 --out <dir>
"""
from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd

from simplenavsim.realism import data as D
from simplenavsim.realism import metrics as M
from simplenavsim.realism import sim as S

D.MAX_STEPS = 50
HORIZONS = {4.8: 12, 8.0: 20, 12.0: 30, 16.0: 40, 20.0: 50}
D.HORIZONS = HORIZONS
FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
PRED = 12


def forecaster(name: str, fold: str, device: str):
    """Scene-level ``f(obs, obs_valid, k, seed, goal=None) -> [N, k, 12, 2]`` for a method name."""
    from forecast_bench.predict import STARS, stars_predictor

    if name in STARS:
        return stars_predictor(STARS[name], device)
    if name == "stars_loo":
        from forecast_bench.windows import ROOT
        return stars_predictor(ROOT / f"experiments/forecast_bench/stars_loo/{fold}/best.pt", device)
    module, _, variant = name.partition("_")          # e.g. socialvae_official -> predict_socialvae, 'official'
    import importlib
    mod = importlib.import_module(f"forecast_bench.predict_{module}")
    if module in ("socialvae", "moflow"):
        return mod.make_predictor(f"{variant}:{fold}" if variant == "official" else variant, device)
    # trajectron / agentformer / sgan: official per-fold ckpt (the T++ adapter runs on CPU only)
    return mod.make_predictor(fold, "cpu" if module == "trajectron" else device)


def receding(predict, stride: int):
    """Closed-loop simulator (``sim`` interface) from a scene-level forecaster."""
    takes_goal = "goal" in inspect.signature(predict).parameters

    def run(x, steps, seed: int = 0, **_):
        B, N = x["agents"].shape
        out = np.zeros((B, N, steps, 2))
        rng = np.random.default_rng(seed)
        for b in range(B):
            a = x["agents"][b]
            obs, valid, goal = x["obs"][b, a].copy(), x["obs_valid"][b, a].copy(), x["goal"][b, a]
            done = []
            for _ in range(math.ceil(steps / stride)):
                extra = {"goal": goal} if takes_goal else {}
                pred = predict(obs, valid, k=1, seed=int(rng.integers(1 << 31)), **extra)[:, 0, :stride]
                pred = np.nan_to_num(pred, nan=0.0)
                done.append(pred)
                obs = np.concatenate([obs, pred], 1)[:, -D.OBS:]
                valid = np.concatenate([valid, np.ones(pred.shape[:2], bool)], 1)[:, -D.OBS:]
            out[b, a] = np.concatenate(done, 1)[:, :steps]
        return out
    return run


def simulator(method: str, fold: str, device: str):
    if method == "cv":
        return S.cv
    if method.endswith("_native"):
        name = method.removesuffix("_native")
        path = S.CHECKPOINTS.get(name)
        if name == "stars_loo":
            from forecast_bench.windows import ROOT
            path = ROOT / f"experiments/forecast_bench/stars_loo/{fold}/best.pt"
        return S.Stars(path, device=device)
    name, _, stride = method.partition("@")
    return receding(forecaster(name, fold, device), int(stride or 4))


def score(traj: np.ndarray, w) -> list[dict]:
    rows = []
    for label, h in HORIZONS.items():
        if not w.scored(h).any():
            continue
        row, _, kin = M.window_metrics(traj, w, h)
        s = w.scored(h)
        speed_gt = M.kinematics(w.obs[s, -1], w.future[s, :h])["speed"]
        rows.append({"horizon_s": label, "dataset": w.dataset, "scene_id": w.scene_id, **row,
                     "speed_sum": kin["speed"].sum(), "speed_gt_sum": speed_gt.sum(), "speed_n": kin["speed"].size,
                     "speed_w1_sum": _w1(kin["speed"], speed_gt) * row["agents"]})
    return rows


def _w1(a, b, n: int = 200) -> float:
    q = (np.arange(n) + 0.5) / n
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", required=True, help="Comma-separated: ground_truth, cv, <forecaster>@<stride>, stars_*_native")
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--stride", type=int, default=25, help="Window start stride in steps (25 = 10 s)")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for method in args.methods.split(","):
        try:
            run_method(method, args)
        except Exception as error:   # one failing baseline must not stop the others
            import traceback
            traceback.print_exc()
            print(f"FAILED {method}: {error!r}", flush=True)


def run_method(method: str, args) -> None:
    if True:
        rows = []
        for fold in args.folds.split(","):
            wins = D.benchmark_windows(scenes=[fold], stride=args.stride)
            fn = None if method == "ground_truth" else simulator(method, fold, args.device)
            seeds = [0] if method in ("ground_truth", "cv") else range(args.seeds)
            for seed in seeds:
                for group in _chunks(wins):
                    if fn is None:
                        trajs = [w.future for w in group]
                    else:
                        sim = fn(S.pad(group), D.MAX_STEPS, seed=seed)
                        trajs = [sim[b, :len(w.agent_ids), :w.steps] for b, w in enumerate(group)]
                    for w, t in zip(group, trajs):
                        rows += [{"method": method, "seed": seed, **r} for r in score(t, w)]
            print(f"{method} {fold}: {len(wins)} windows", flush=True)
        tag = "" if args.folds == ",".join(FOLDS) else "__" + args.folds.replace(",", "-")   # per-fold runs: own file
        pd.DataFrame(rows).to_csv(args.out / f"{method.replace('@', '_s')}{tag}.csv", index=False)


def _chunks(wins, size: int = 16):
    order = sorted(wins, key=lambda w: len(w.agent_ids))
    return [order[i:i + size] for i in range(0, len(order), size)]


if __name__ == "__main__":
    main()
