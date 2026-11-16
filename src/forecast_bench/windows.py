"""Common open-loop forecasting windows (8 observed / 12 predicted steps at 0.4 s) for every method.

One ``.npz`` per evaluation set, flat node arrays with ``ptr`` marking windows (like
``stars_ma_batch.WindowStore``):

    obs [M, 8, 2], obs_valid [M, 8], fut [M, 12, 2], fut_valid [M, 12]   float32 / bool, metres
    ptr [W + 1]          nodes of window w are ptr[w]:ptr[w+1]
    eval_mask [M]        node is scored: observed at all 8 and all 12 steps
    goal [M, 4]          track end (x, y, valid, steps after the last observed step), for goal-conditioned STARS
    window_key [W]       "<recording>:<first frame>"

Nodes are every pedestrian visible at the last observed step with >= 2 observed points (STARS'
rule); methods that only consume fully observed agents (Social-GAN style) use the ``eval_mask``
subset as their scene. Predictions are scored only on ``eval_mask`` nodes.

Sets:
* ``ethucy_<fold>``: the standard leave-one-out test scenes (Social-GAN / SocialVAE ``test/*.txt``
  files, world metres, frame stride 1 i.e. every 0.4 s start), windows with >= 1 fully observed
  pedestrian (SocialVAE's rule; Social-GAN additionally drops windows with a single one).
* ``pooled_<dataset>``: each dataset's test split as our STARS configs define it (``windows=unique``,
  target-centric scene frame), a seeded subset of at most ``--max-windows`` windows (fewer for the
  dense datasets, ``MAX_WINDOWS``, to keep ~20k scored agents per set).

    python -m forecast_bench.windows --out experiments/forecast_bench/windows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OBS, PRED = 8, 12
SEQ = OBS + PRED
FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
POOLED = ("steps", "ptdpes", "madras", "grand_central", "atc", "eth", "ucy")
MAX_WINDOWS = {"ptdpes": 500, "grand_central": 300, "atc": 300}
SGAN_DATA = ROOT / "references" / "social-vae" / "data"


def _pack(windows: list[dict]) -> dict[str, np.ndarray]:
    cat = lambda key: np.concatenate([w[key] for w in windows])
    sizes = np.array([len(w["obs"]) for w in windows])
    out = {key: cat(key) for key in ("obs", "obs_valid", "fut", "fut_valid", "goal")}
    out["ptr"] = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    out["eval_mask"] = out["obs_valid"].all(1) & out["fut_valid"].all(1)
    out["window_key"] = np.array([w["key"] for w in windows])
    return out


def sgan_windows(path: Path) -> list[dict]:
    """Every 20-frame window of one Social-GAN-format file (frame, id, x, y) on its regular frame grid."""
    data = np.loadtxt(path)
    frames = np.unique(data[:, 0])
    step = np.diff(frames).min()
    grid = np.arange(frames[0], frames[-1] + step / 2, step)
    ids = np.unique(data[:, 1])
    column = {pid: i for i, pid in enumerate(ids)}
    xy = np.full((len(grid), len(ids), 2), np.nan, dtype=np.float64)
    rows = np.rint((data[:, 0] - frames[0]) / step).astype(int)
    xy[rows, [column[p] for p in data[:, 1]]] = data[:, 2:4]
    valid = ~np.isnan(xy[..., 0])
    last_row = np.array([np.flatnonzero(valid[:, j]).max() for j in range(len(ids))])
    windows = []
    for t0 in range(len(grid) - SEQ + 1):
        v = valid[t0:t0 + SEQ]
        obs_v = v[:OBS]
        nodes = np.flatnonzero(obs_v[-1] & (obs_v.sum(0) >= 2))
        full = v[:, nodes].all(0)
        if not full.any():
            continue
        # fully observed (scored) agents first, then context agents
        nodes = np.r_[nodes[full], nodes[~full]]
        w = np.nan_to_num(xy[t0:t0 + SEQ, nodes]).transpose(1, 0, 2).astype(np.float32)
        wv = v[:, nodes].T
        end = last_row[nodes]
        goal = np.stack([xy[end, nodes, 0], xy[end, nodes, 1], np.ones(len(nodes)),
                         end - (t0 + OBS - 1)], axis=1).astype(np.float32)
        windows.append(dict(obs=w[:, :OBS], obs_valid=wv[:, :OBS], fut=w[:, OBS:], fut_valid=wv[:, OBS:],
                            goal=goal, key=f"{path.stem}:{int(grid[t0])}"))
    return windows


def ethucy_set(fold: str) -> dict[str, np.ndarray]:
    windows = []
    for path in sorted((SGAN_DATA / {"zara1": "zara01", "zara2": "zara02"}.get(fold, fold) / "test").glob("*.txt")):
        windows += sgan_windows(path)
    return _pack(windows)


def pooled_set(name: str, max_windows: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Test windows of one of our datasets via the STARS window store (with oracle track-end goals)."""
    from types import SimpleNamespace

    import torch

    from baselines.train_stars_ma import _NullRun, dataset_entries, load_stores
    from forecasting_config import resolve_forecasting_config

    args = SimpleNamespace(data_seed=42, seed=42, world_size=1, obs=OBS, pred=PRED, main=True, run=_NullRun(), max_scenes=None,
                           cache_dir=ROOT / "datasets/.cache/stars_ma", cache_workers=0, goals=True)
    store = load_stores(dataset_entries([name]), "test", args, resolve_forecasting_config(preset="steps_default"),
                        windows="unique")
    if len(store) > max_windows:
        keep = np.sort(np.random.default_rng(seed).choice(len(store), max_windows, replace=False))
        store = store.select(torch.as_tensor(keep))
    ptr = store.ptr.numpy()
    windows = []
    for w in range(len(store)):
        a, b = ptr[w], ptr[w + 1]
        windows.append(dict(obs=store.obs_xy[a:b].numpy(), obs_valid=store.obs_valid[a:b].numpy(),
                            fut=store.future[a:b].numpy(), fut_valid=store.future_valid[a:b].numpy(),
                            goal=store.goal[a:b].numpy(), key=f"{name}:{int(store.sample_index[w])}"))
    return _pack(windows)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/forecast_bench/windows")
    parser.add_argument("--sets", default=",".join([f"ethucy_{f}" for f in FOLDS] + [f"pooled_{d}" for d in POOLED]))
    parser.add_argument("--max-windows", type=int, default=1500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.sets.split(","):
        kind, dataset = name.split("_", 1)
        data = ethucy_set(dataset) if kind == "ethucy" else pooled_set(dataset, min(args.max_windows, MAX_WINDOWS.get(dataset, args.max_windows)))
        np.savez_compressed(args.out / f"{name}.npz", **data)
        sizes = np.diff(data["ptr"])
        print(f"{name}: {len(sizes)} windows, {len(data['obs'])} nodes, {int(data['eval_mask'].sum())} scored, "
              f"max {sizes.max()} nodes/window", flush=True)


if __name__ == "__main__":
    main()
