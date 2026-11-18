"""Official pretrained Trajectron++ (ETH/UCY pedestrian models) on the common windows.

Checkpoints: deleted upstream in 58248ee, recovered from git history (configs are still in the tree):

    R=references/trajectron-plus-plus; C=/scratch/09865/mmunje/.cache/trajectron
    for f in eth hotel univ zara1 zara2; do for v in vel attention_radius_3; do mkdir -p $C/models/${f}_$v
      git -C $R show 58248ee^:experiments/pedestrians/models/${f}_$v/model_registrar-100.pt > $C/models/${f}_$v/model_registrar-100.pt
      cp $R/experiments/pedestrians/models/${f}_$v/config.json $C/models/${f}_$v/; done; done

Adapter (one T++ scene per window, observed steps only, every window node is a neighbour candidate):

    python -m forecast_bench.predict_trajectron --sets ethucy_eth,...

Official best-of-20 reproduction (the ``BEST OF 20`` block of experiments/pedestrians/evaluate.py on the
``<fold>_test.pkl`` written by experiments/pedestrians/process_data.py):

    python -m forecast_bench.predict_trajectron --official $C/work_gradient/experiments/processed --sets eth,...

These weights predate upstream 03712b7, which replaced np.gradient in derivative_of (central differences
over whole tracks, so the current-step velocity/acceleration inputs use x[t+1], x[t+2]: future leakage)
with backward differences. The published numbers need the old preprocessing, made with
``OLD_GRADIENT=1 python <repo>/scripts/forecast_bench_upstream/trajectron_run.py <R>/experiments/pedestrians/process_data.py`` run from
$C/work_gradient/experiments/pedestrians (raw -> <R>/experiments/pedestrians/raw, ../../trajectron -> <R>/trajectron);
with the current derivative_of the old weights do much worse (out of distribution). The adapter keeps
np.gradient (the training features) but on observed steps only: no leakage. Samples are independent per
agent (each node draws its own latent).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch

from baselines.trajectron_data import DATA_COLUMNS, STANDARDIZATION, Environment, Node, Scene
from forecast_bench.windows import OBS, PRED, ROOT, load
from model.model_registrar import ModelRegistrar  # noqa: E402  (path set up by trajectron_data)
from model.trajectron import Trajectron  # noqa: E402
import evaluation  # noqa: E402

MODELS = Path("/scratch/09865/mmunje/.cache/trajectron/models")
VARIANT = "vel"  # velocity output + single-integrator dynamics: the README's per-scene eval commands
DT, K = 0.4, 20


def derivative_of(x: np.ndarray, dt: float = DT) -> np.ndarray:
    """np.gradient, as upstream's derivative_of before 03712b7: the features these weights were trained on.

    Applied to observed steps only, so the last step is a one-sided (backward) difference; upstream's
    processed data differentiated whole tracks, so its current-step velocity/acceleration saw x[t+1], x[t+2].
    """
    return np.gradient(x, dt)


def load_model(fold: str, env, variant: str = VARIANT) -> tuple[Trajectron, dict]:
    model_dir = MODELS / f"{fold}_{variant}"
    registrar = ModelRegistrar(str(model_dir), "cpu")
    # pickled nn.ModuleDict from the official repo history (torch 1.4), hence weights_only=False
    registrar.model_dict = torch.load(model_dir / "model_registrar-100.pt", map_location="cpu", weights_only=False)
    hyperparams = json.loads((model_dir / "config.json").read_text())
    model = Trajectron(registrar, hyperparams, None, "cpu")
    model.set_environment(env)
    model.set_annealing_params()
    return model, hyperparams


def environment() -> Environment:
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=STANDARDIZATION)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0}
    env.scenes = [Scene(timesteps=OBS, dt=DT)]  # the model reads dt from scenes[0]
    return env


def window_scene(env, obs: np.ndarray, valid: np.ndarray) -> tuple[Scene, list]:
    """One 8-step T++ scene from a window's observed tracks (each node's contiguous run ending at the last step)."""
    scene = Scene(timesteps=OBS, dt=DT, name="window")
    nodes = []
    for j in range(len(obs)):
        gaps = np.flatnonzero(~valid[j])
        first = gaps.max() + 1 if len(gaps) else 0
        if OBS - first < 2:
            nodes.append(None)
            continue
        x, y = obs[j, first:, 0].astype(np.float64), obs[j, first:, 1].astype(np.float64)
        vx, vy = derivative_of(x, DT), derivative_of(y, DT)
        data = pd.DataFrame({("position", "x"): x, ("position", "y"): y,
                             ("velocity", "x"): vx, ("velocity", "y"): vy,
                             ("acceleration", "x"): derivative_of(vx, DT),
                             ("acceleration", "y"): derivative_of(vy, DT)}, columns=DATA_COLUMNS)
        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=str(j), data=data, first_timestep=first)
        scene.nodes.append(node)
        nodes.append(node)
    return scene, nodes


@torch.no_grad()
def predict(windows: dict, fold: str, variant: str = VARIANT, samples: int = K, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = environment()
    model, hp = load_model(fold, env, variant)
    ptr, mask = windows["ptr"], windows["eval_mask"]
    out = np.full((len(mask), samples, PRED, 2), np.nan, dtype=np.float32)
    for w in range(len(ptr) - 1):
        a, b = ptr[w], ptr[w + 1]
        scene, nodes = window_scene(env, windows["obs"][a:b], windows["obs_valid"][a:b])
        scene.calculate_scene_graph(env.attention_radius, hp["edge_addition_filter"], hp["edge_removal_filter"])
        pred = model.predict(scene, np.array([OBS - 1]), PRED, num_samples=samples, min_history_timesteps=OBS - 1,
                             min_future_timesteps=0, z_mode=False, gmm_mode=False, full_dist=False)
        pred = pred.get(OBS - 1, {})
        for j in np.flatnonzero(mask[a:b]):
            out[a + j] = pred[nodes[j]][0]           # (1, K, 12, 2) -> (K, 12, 2), scene = world frame
    out = out[mask]
    assert np.isfinite(out).all(), "a scored node got no prediction"
    return out


def make_predictor(fold: str, device="cpu", variant: str = VARIANT):
    """f(obs [N, 8, 2], obs_valid [N, 8], k, seed) -> [N, k, 12, 2] for every agent of one scene.

    Missing history is back-filled with the first observed position, so every agent has the 8 steps T++
    needs to be predicted (min_history_timesteps=7). CPU only (the registrar is loaded on CPU).
    """
    from forecast_bench.predict_agentformer import backfill

    env = environment()
    model, hp = load_model(fold, env, variant)

    @torch.no_grad()
    def f(obs: np.ndarray, obs_valid: np.ndarray, k: int = K, seed: int = 0) -> np.ndarray:
        torch.manual_seed(seed)
        scene, nodes = window_scene(env, backfill(obs, obs_valid), np.ones(obs_valid.shape, dtype=bool))
        scene.calculate_scene_graph(env.attention_radius, hp["edge_addition_filter"], hp["edge_removal_filter"])
        pred = model.predict(scene, np.array([OBS - 1]), PRED, num_samples=k, min_history_timesteps=OBS - 1,
                             min_future_timesteps=0, z_mode=False, gmm_mode=False, full_dist=False)[OBS - 1]
        return np.stack([pred[node][0] for node in nodes]).astype(np.float32)
    return f


@torch.no_grad()
def official(processed: Path, fold: str, variant: str = VARIANT, seed: int = 0) -> tuple[float, float, int]:
    """The BEST OF 20 block of experiments/pedestrians/evaluate.py (KDE NLL skipped; it does not affect ADE/FDE)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    with open(processed / f"{fold}_test.pkl", "rb") as f:
        env = dill.load(f, encoding="latin1")
    model, hp = load_model(fold, env, variant)
    ph, max_hl = hp["prediction_horizon"], hp["maximum_history_length"]
    ade, fde = [], []
    for scene in env.scenes:
        scene.calculate_scene_graph(env.attention_radius, hp["edge_addition_filter"], hp["edge_removal_filter"])
        for t in range(0, scene.timesteps, 10):
            predictions = model.predict(scene, np.arange(t, t + 10), ph, num_samples=20, min_history_timesteps=7,
                                        min_future_timesteps=12, z_mode=False, gmm_mode=False, full_dist=False)
            if not predictions:
                continue
            errors = evaluation.compute_batch_statistics(predictions, scene.dt, max_hl=max_hl, ph=ph,
                                                         node_type_enum=env.NodeType, map=None, kde=False,
                                                         best_of=True, prune_ph_to_future=True)
            ade += errors[env.NodeType.PEDESTRIAN]["ade"]
            fde += errors[env.NodeType.PEDESTRIAN]["fde"]
    return float(np.mean(ade)), float(np.mean(fde)), len(ade)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/forecast_bench")
    parser.add_argument("--official", type=Path, help="processed-data dir: run the official eval instead")
    parser.add_argument("--variant", default=VARIANT, choices=["vel", "attention_radius_3"])
    args = parser.parse_args()
    for name in args.sets.split(","):
        fold = name.removeprefix("ethucy_")
        if args.official:
            ade, fde, n = official(args.official, fold, args.variant)
            print(f"official {fold}_{args.variant}: best-of-20 ADE {ade:.4f} FDE {fde:.4f} (n={n})", flush=True)
            continue
        windows = load(args.root / "windows" / f"{name}.npz")
        pred = predict(windows, fold, args.variant)
        path = args.root / "preds" / name / "trajectron_official.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pred=pred)
        print(f"{name}/trajectron_official: {pred.shape}", flush=True)


if __name__ == "__main__":
    main()
