"""Multi-agent STARS: feature invariance, encoder fidelity to STARS, forecaster and rollout contracts."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from baselines import stars_data
from baselines.stars import upstream_model_module
from baselines.stars_ma_batch import WindowStore, build_batch
from baselines.stars_ma_graph import build_graph, edge_features, fill_observed, rotate, to_local
from baselines.stars_ma_model import StarsMotionEncoder, StarsMultiAgentForecaster, rollout

DT = 0.4
SMALL = {"hidden_dim": 32, "latent_dim": 16, "edge_transformer_layers": 1, "node_transformer_layers": 1,
         "num_heads": 4, "dropout": 0.0}


def _walkers(agents: int = 6, steps: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = rng.uniform(-6, 6, size=(1, agents, 2))
    velocity = rng.normal(0, 0.5, size=(1, agents, 2))
    noise = rng.normal(0, 0.03, size=(steps, agents, 2)).cumsum(0)
    return (start + velocity * np.arange(steps)[:, None, None] * DT + noise).astype(np.float32)


def _batch(scenes: list[np.ndarray], knn: int | None = 4, radius: float | None = None):
    """A ``GraphBatch`` of fully observed ``[20, N, 2]`` scenes (8 observed + 12 future steps)."""
    paths = [torch.from_numpy(xy).transpose(0, 1) for xy in scenes]
    ptr = torch.tensor([0] + [len(p) for p in paths]).cumsum(0)
    xy = torch.cat(paths)
    valid = torch.ones(xy.shape[:2], dtype=torch.bool)
    store = WindowStore(xy[:, :8], valid[:, :8], xy[:, 8:], valid[:, 8:], ptr, torch.zeros(len(paths), dtype=torch.long),
                        torch.arange(len(paths)), ["synthetic"])
    return build_batch(store, torch.arange(len(paths)), dt=DT, knn=knn, radius=radius)


def _model(latent_dim: int = 0) -> StarsMultiAgentForecaster:
    torch.manual_seed(0)
    return StarsMultiAgentForecaster(8, 12, dt=DT, encoder_recipe=SMALL,
                                     decoder_recipe={"hidden_dim": 32, "latent_dim": latent_dim, "dropout": 0.0}).eval()


def test_graph_features_match_stars_and_are_invariant_to_the_scene_frame():
    xy = torch.from_numpy(_walkers()[:8])
    valid = torch.ones(8, 6, dtype=torch.bool)
    angle, shift = torch.tensor(0.7), torch.tensor([3.0, -2.0])
    reference = build_graph(xy, valid, dt=DT)
    moved = build_graph(rotate(xy, angle) + shift, valid, dt=DT)
    torch.testing.assert_close(moved["node_attr"], reference["node_attr"], atol=1e-4, rtol=0)
    torch.testing.assert_close(moved["edge_attr"], reference["edge_attr"], atol=1e-4, rtol=0)
    assert torch.equal(moved["edge_index"], reference["edge_index"])
    # Each node's own frame: at the origin, facing along x, at the last observed step.
    torch.testing.assert_close(reference["node_attr"][:, -1, :2], torch.zeros(6, 2), atol=1e-5, rtol=0)
    torch.testing.assert_close(reference["node_attr"][:, -1, 5], torch.ones(6), atol=1e-5, rtol=0)
    # Gap filling and SE(2) edge features are STARS's (numpy adapter).
    gappy = np.random.default_rng(1).random((8, 6)) > 0.3
    gappy[-1] = True
    filled = stars_data.fill_observed(xy.numpy(), gappy)
    np.testing.assert_allclose(fill_observed(xy, torch.from_numpy(gappy)).numpy(), filled, atol=1e-6)
    theta, edges = stars_data.headings_from_motion(filled), stars_data.fully_connected_edges(6)
    np.testing.assert_allclose(edge_features(torch.from_numpy(filled), torch.from_numpy(theta), torch.from_numpy(edges)),
                               stars_data.edge_features(filled, theta, edges), atol=1e-5)


def test_edge_only_encoder_reproduces_upstream_stars_at_40_steps():
    torch.manual_seed(0)
    upstream = upstream_model_module()
    xy = torch.from_numpy(_walkers(agents=5, steps=40, seed=3))
    graph = build_graph(xy, torch.ones(40, 5, dtype=torch.bool), dt=DT)
    data = Data(num_nodes=5, x=None, **graph)
    ours = StarsMotionEncoder(40, {**SMALL, "use_node_features": False}).eval()
    theirs = upstream.CombinedHGNNEncoder(hidden_dim=32, latent_dim=16, num_mp_layers=2, edge_transformer_layers=1,
                                          num_heads=4, dropout=0.0).eval()
    theirs.load_state_dict({**ours.base.state_dict(), "robot_feat_proj.weight": theirs.robot_feat_proj.weight})
    with torch.no_grad():
        torch.testing.assert_close(ours(data)["hidden"], theirs._encode_h(data))


def test_predictions_ignore_the_future_and_are_permutation_equivariant():
    xy = _walkers()
    model = _model()
    prediction = model.predict(_batch([xy]))
    simulated = rollout(model, _batch([xy]), rounds=3, exec_steps=4)
    assert simulated.shape == (6, 12, 2)

    altered = xy.copy()
    altered[8:] += 5.0
    torch.testing.assert_close(model.predict(_batch([altered])), prediction)
    torch.testing.assert_close(rollout(model, _batch([altered]), rounds=3, exec_steps=4), simulated)

    order = np.array([3, 0, 5, 1, 4, 2])
    permuted = model.predict(_batch([xy[:, order]]))
    torch.testing.assert_close(permuted, prediction[:, torch.from_numpy(order)], atol=1e-5, rtol=1e-4)


def test_cvae_training_and_sampling_shapes():
    model = _model(latent_dim=8).train()
    batch = _batch([_walkers(seed=s) for s in range(3)])
    future_local = to_local(batch.future, batch.origin, batch.heading)
    out = model.training_forward(batch, future_local, batch.future_valid)
    assert out["prediction"].shape == (18, 12, 2) and out["kl"].shape == (18,) and bool((out["kl"] >= -1e-5).all())
    (out["prediction"].square().mean() + out["kl"].mean()).backward()
    model.eval()
    samples = model.predict(batch, samples=5)
    assert samples.shape == (5, 18, 12, 2) and samples.std(dim=0).mean() > 0
    assert rollout(model, batch, rounds=2, exec_steps=12).shape == (18, 24, 2)
