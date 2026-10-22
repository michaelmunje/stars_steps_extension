"""Scaled multi-agent STARS path: batched graphs, vectorised metrics, node budgets, configs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from baselines.stars_ma_batch import (
    WindowStore,
    build_batch,
    concat_stores,
    node_budget_batches,
    window_weights,
)
from baselines.stars_ma_graph import build_graph
from baselines.stars_ma_metrics import MetricAccumulator, batch_metrics, forecasting_metrics
from baselines.stars_ma_model import StarsMultiAgentForecaster, rollout
from baselines.train_stars_ma import dataset_entries, parse_args

DT = 0.4


def _store(seed: int = 0, windows: int = 12, name: str = "synthetic") -> WindowStore:
    rng = np.random.default_rng(seed)
    obs_xy, obs_valid, future, future_valid, sizes = [], [], [], [], []
    for _ in range(windows):
        n = int(rng.integers(1, 25))
        start = rng.uniform(-8, 8, (n, 1, 2))
        path = start + np.cumsum(rng.normal(0, 0.3, (n, 20, 2)), axis=1)
        stationary = rng.random(n) < 0.3
        path[stationary] = start[stationary]
        valid = rng.random((n, 20)) > 0.2
        valid[:, 7] = True
        path = np.where(valid[..., None], path, 0.0).astype(np.float32)
        obs_xy.append(path[:, :8]), obs_valid.append(valid[:, :8])
        future.append(path[:, 8:]), future_valid.append(valid[:, 8:])
        sizes.append(n)
    ptr = torch.zeros(windows + 1, dtype=torch.long)
    ptr[1:] = torch.tensor(sizes).cumsum(0)
    cat = lambda parts: torch.from_numpy(np.concatenate(parts))
    return WindowStore(cat(obs_xy), cat(obs_valid), cat(future), cat(future_valid), ptr,
                       torch.zeros(windows, dtype=torch.long), torch.arange(windows), [name])


@pytest.mark.parametrize("knn,radius", [(16, 10.0), (3, None), (None, 4.0), (None, None)])
def test_batched_graphs_equal_per_window_graphs(knn, radius):
    store = _store()
    batch = build_batch(store, torch.arange(len(store)), dt=DT, knn=knn, radius=radius)
    for w in range(len(store)):
        s, e = int(store.ptr[w]), int(store.ptr[w + 1])
        xy = torch.where(store.obs_valid[s:e, :, None], store.obs_xy[s:e], torch.nan)
        reference = build_graph(xy.transpose(0, 1), store.obs_valid[s:e].T, dt=DT, knn=knn, radius=radius)
        for key in ("node_attr", "origin", "heading", "valid_mask"):
            torch.testing.assert_close(getattr(batch, key)[s:e], reference[key])
        mine = batch.edge_index[:, (batch.edge_index[1] >= s) & (batch.edge_index[1] < e)] - s
        torch.testing.assert_close(mine, reference["edge_index"])
        chosen = (batch.edge_index[1] >= s) & (batch.edge_index[1] < e)
        torch.testing.assert_close(batch.edge_attr[chosen], reference["edge_attr"])
    assert bool(batch.is_target[batch.ptr[:-1]].all()) and int(batch.is_target.sum()) == len(store)


def test_vectorised_metrics_equal_per_graph_metrics():
    store = _store(1)
    batch = build_batch(store, torch.arange(len(store)), dt=DT, knn=16, radius=10.0)
    torch.manual_seed(0)
    for k in (1, 4):
        prediction = batch.future[None] + 0.4 * torch.randn(k, *batch.future.shape)
        reference, vectorised = MetricAccumulator(), MetricAccumulator()
        forecasting_metrics(prediction, batch, reference)
        batch_metrics(prediction, batch, vectorised, dataset_names=store.names)
        expected, actual = reference.result(), vectorised.result()
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key
        assert actual["synthetic/agent_ade"] == pytest.approx(expected["agent_ade"], rel=1e-6)


def test_batched_rollout_equals_graph_by_graph_rollout():
    from torch_geometric.data import Batch, Data

    store = _store(2, windows=5)
    batch = build_batch(store, torch.arange(len(store)), dt=DT, knn=4, radius=8.0)
    torch.manual_seed(0)
    model = StarsMultiAgentForecaster(8, 12, dt=DT, encoder_recipe={"hidden_dim": 32, "latent_dim": 16,
                                      "edge_transformer_layers": 1, "node_transformer_layers": 1, "dropout": 0.0},
                                      decoder_recipe={"hidden_dim": 32, "dropout": 0.0}).eval()
    fields = ("node_type", "node_attr", "valid_mask", "edge_index", "edge_attr", "origin", "heading", "obs_xy")
    graphs = []
    for w in range(len(store)):
        s, e = int(batch.ptr[w]), int(batch.ptr[w + 1])
        edges = (batch.edge_index[1] >= s) & (batch.edge_index[1] < e)
        graphs.append(Data(num_nodes=e - s, **{key: getattr(batch, key)[s:e] for key in fields if "edge" not in key},
                           edge_index=batch.edge_index[:, edges] - s, edge_attr=batch.edge_attr[edges]))
    reference = rollout(model, Batch.from_data_list(graphs), rounds=3, exec_steps=4, knn=4, radius=8.0)
    torch.testing.assert_close(rollout(model, batch, rounds=3, exec_steps=4, knn=4, radius=8.0), reference,
                               atol=1e-5, rtol=1e-4)


def test_node_budget_batches_cover_every_window_once_within_budget():
    sizes = torch.tensor([3, 50, 7, 1, 20, 20, 300, 9, 4, 12])
    batches = node_budget_batches(sizes, 40, generator=torch.Generator().manual_seed(0))
    flat = torch.cat(batches).sort().values
    assert torch.equal(flat, torch.arange(len(sizes)))
    assert all(int(sizes[b].sum()) <= 40 or len(b) == 1 for b in batches)
    ordered = node_budget_batches(sizes, 40, shuffle=False)
    assert torch.equal(torch.cat(ordered), torch.arange(len(sizes)))


def test_store_concat_select_and_weights():
    a, b = _store(0, 4, "a"), _store(1, 3, "b")
    both = concat_stores([a, b])
    assert both.names == ["a", "b"] and len(both) == 7 and both.dataset.tolist() == [0] * 4 + [1] * 3
    torch.testing.assert_close(both.select(torch.tensor([5])).obs_xy, b.select(torch.tensor([1])).obs_xy)
    assert window_weights(both, {"a": 1.0, "b": 1.0}) is None
    assert window_weights(both, {"a": 1.0, "b": 0.5}).tolist() == [1.0] * 4 + [0.5] * 3


def test_config_datasets_and_flag_precedence(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("datasets:\n  - steps\n  - {name: ptdpes, weight: 0.5}\nval_datasets: [steps]\n"
                      "max_nodes: 2048\nknn: 8\n")
    args = parse_args(["--config", str(config), "--phase", "ssl", "--run-dir", str(tmp_path / "r"), "--knn", "12"])
    assert [e["name"] for e in args.train_entries] == ["steps", "ptdpes"]
    assert args.train_entries[1]["weight"] == 0.5 and [e["name"] for e in args.val_entries] == ["steps"]
    assert args.max_nodes == 2048 and args.knn == 12
    assert dataset_entries("grand_central")[0]["split_strategy"] == "temporal"
    with pytest.raises(ValueError):
        dataset_entries(["nope"])
    config.write_text("datasets: [steps]\nmax_nodez: 1\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config), "--phase", "ssl", "--run-dir", str(tmp_path / "r")])
