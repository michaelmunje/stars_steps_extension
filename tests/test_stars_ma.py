"""Multi-agent STARS: graph construction, encoder fidelity to STARS, forecaster and rollout contracts."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")
from torch.utils.data import ConcatDataset
from torch_geometric.data import Batch, Data

from baselines import stars_data
from baselines.stars import upstream_model_module
from baselines.stars_ma_data import unique_window_indices
from baselines.stars_ma_graph import (
    build_graph,
    edge_features,
    fill_observed,
    headings_from_motion,
    neighbourhood_edges,
    rotate,
    select_agents,
    to_local,
    to_scene,
)
from baselines.stars_ma_metrics import MetricAccumulator, forecasting_metrics
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


def _graph(xy: np.ndarray, valid: np.ndarray, obs: int = 8, **kwargs) -> Data:
    xy_t, valid_t = torch.from_numpy(xy), torch.from_numpy(valid)
    agents = select_agents(valid_t, obs)
    graph = build_graph(xy_t[:obs, agents], valid_t[:obs, agents], dt=DT, **kwargs)
    return Data(
        num_nodes=len(agents), **graph,
        obs_xy=fill_observed(torch.nan_to_num(xy_t[:obs, agents]), valid_t[:obs, agents]).transpose(0, 1).contiguous(),
        future=torch.nan_to_num(xy_t[obs:, agents]).transpose(0, 1).contiguous(),
        future_valid=valid_t[obs:, agents].T.contiguous(),
        is_target=torch.arange(len(agents)) == 0,
    )


def test_torch_graph_matches_numpy_stars_adapter():
    xy = _walkers()[:8]
    valid = np.random.default_rng(1).random((8, 6)) > 0.3
    valid[:, 0] = True
    valid[-1] = True
    numpy_filled = stars_data.fill_observed(xy, valid)
    torch_filled = fill_observed(torch.from_numpy(xy), torch.from_numpy(valid))
    np.testing.assert_allclose(torch_filled.numpy(), numpy_filled, atol=1e-6)
    theta = stars_data.headings_from_motion(numpy_filled)
    np.testing.assert_allclose(headings_from_motion(torch.from_numpy(numpy_filled)).numpy(), theta, atol=1e-6)
    edges = stars_data.fully_connected_edges(6)
    expected = stars_data.edge_features(numpy_filled, theta, edges)
    actual = edge_features(torch.from_numpy(numpy_filled), torch.from_numpy(theta), torch.from_numpy(edges))
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-5)


def test_graph_features_are_invariant_to_the_scene_frame():
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


def test_neighbourhood_topology():
    position = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [10.5, 0.0]])
    valid = torch.ones(5, 8, dtype=torch.bool)
    full = neighbourhood_edges(position, valid, knn=None, radius=None)
    assert full.shape[1] == 5 * 4 and not (full[0] == full[1]).any()
    knn = neighbourhood_edges(position, valid, knn=2, radius=None)
    assert torch.bincount(knn[1], minlength=5).max() == 2
    assert {(0, 2), (1, 2)} <= set(map(tuple, knn[:, knn[1] == 2].T.tolist()))
    near = neighbourhood_edges(position, valid, knn=None, radius=1.5)
    assert set(map(tuple, near.T.tolist())) == {(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3)}
    # No edge between nodes that never share a valid step.
    disjoint = valid.clone()
    disjoint[0, 4:] = False
    disjoint[1, :4] = False
    assert not ((neighbourhood_edges(position, disjoint, knn=None, radius=None) == torch.tensor([[0], [1]])).all(0)).any()


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


def _model(latent_dim: int = 0) -> StarsMultiAgentForecaster:
    torch.manual_seed(0)
    return StarsMultiAgentForecaster(8, 12, dt=DT, encoder_recipe=SMALL,
                                     decoder_recipe={"hidden_dim": 32, "latent_dim": latent_dim, "dropout": 0.0}).eval()


def test_predictions_ignore_the_future_and_are_permutation_equivariant():
    xy = _walkers()
    valid = np.ones((20, 6), dtype=bool)
    model = _model()
    data = _graph(xy, valid, knn=4, radius=None)
    prediction = model.predict(data)
    simulated = rollout(model, Batch.from_data_list([data]), rounds=3, exec_steps=4, knn=4, radius=None)

    altered = xy.copy()
    altered[8:] += 5.0
    other = _graph(altered, valid, knn=4, radius=None)
    torch.testing.assert_close(model.predict(other), prediction)
    torch.testing.assert_close(rollout(model, Batch.from_data_list([other]), rounds=3, exec_steps=4, knn=4, radius=None),
                               simulated)

    order = np.array([3, 0, 5, 1, 4, 2])
    permuted = model.predict(_graph(xy[:, order], valid[:, order], knn=4, radius=None))
    torch.testing.assert_close(permuted, prediction[:, torch.from_numpy(order)], atol=1e-5, rtol=1e-4)


def test_cvae_training_and_sampling_shapes():
    model = _model(latent_dim=8).train()
    batch = Batch.from_data_list([_graph(_walkers(seed=s), np.ones((20, 6), dtype=bool), knn=4, radius=None)
                                  for s in range(3)])
    future_local = to_local(batch.future, batch.origin, batch.heading)
    out = model.training_forward(batch, future_local, batch.future_valid)
    assert out["prediction"].shape == (18, 12, 2) and out["kl"].shape == (18,) and bool((out["kl"] >= -1e-5).all())
    (out["prediction"].square().mean() + out["kl"].mean()).backward()
    model.eval()
    samples = model.predict(batch, samples=5)
    assert samples.shape == (5, 18, 12, 2) and samples.std(dim=0).mean() > 0


def test_closed_loop_rollout_is_differentiable_and_supports_long_rounds():
    model = _model(latent_dim=4).train()
    batch = Batch.from_data_list([_graph(_walkers(seed=s), np.ones((20, 6), dtype=bool), knn=3, radius=8.0)
                                  for s in range(2)])
    simulated = rollout(model, batch, rounds=3, exec_steps=10, knn=3, radius=8.0, keep_graph=True)
    assert simulated.shape == (12, 30, 2)
    simulated.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    with pytest.raises(ValueError):
        rollout(model, batch, rounds=1, exec_steps=13, knn=3, radius=8.0)


def test_round_trip_between_local_and_scene_frames():
    xy = torch.randn(4, 12, 2)
    origin, heading = torch.randn(4, 2), torch.rand(4) * 2 * math.pi
    torch.testing.assert_close(to_scene(to_local(xy, origin, heading), origin, heading), xy, atol=1e-5, rtol=0)


def test_metrics_on_known_predictions():
    xy = _walkers(agents=3)
    valid = np.ones((20, 3), dtype=bool)
    valid[15:, 2] = False  # agent 2's future is incomplete: excluded from ADE/FDE
    batch = Batch.from_data_list([_graph(xy, valid)])
    perfect = batch.future[None].clone()
    shifted = perfect + torch.tensor([0.3, 0.4])  # 0.5 m everywhere
    accumulator = MetricAccumulator()
    forecasting_metrics(torch.cat([shifted, perfect]), batch, accumulator)
    result = accumulator.result()
    assert result["target_ade"] == pytest.approx(0.0, abs=1e-6) and result["agent_fde"] == pytest.approx(0.0, abs=1e-6)
    assert result["n_agent_ade"] == 2 and result["n_target_ade"] == 1
    accumulator = MetricAccumulator()
    forecasting_metrics(shifted, batch, accumulator)
    assert accumulator.result()["joint_ade"] == pytest.approx(0.5, abs=1e-5)


class _Rows:
    def __init__(self, name, rows):
        self.spec = type("Spec", (), {"name": name})()
        self.rows = rows

    def __len__(self):
        return len(self.rows)


def test_unique_windows_keep_the_first_sample_of_each_window():
    first = _Rows("a", [{"recording": "r", "start_step": s} for s in (0, 0, 1, 1, 1)])
    second = _Rows("b", [{"recording": "r", "start_step": 0}, {"recording": "r", "start_step": 0}])
    assert unique_window_indices(ConcatDataset([first, second])) == [0, 2, 5]
