"""Multi-agent STARS training path: batched graphs, vectorised metrics, configs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from baselines.stars_ma_batch import WindowStore, build_batch
from baselines.stars_ma_graph import build_graph
from baselines.stars_ma_metrics import MetricAccumulator, batch_metrics
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
        chosen = (batch.edge_index[1] >= s) & (batch.edge_index[1] < e)
        torch.testing.assert_close(batch.edge_index[:, chosen] - s, reference["edge_index"])
        torch.testing.assert_close(batch.edge_attr[chosen], reference["edge_attr"])
    assert bool(batch.is_target[batch.ptr[:-1]].all()) and int(batch.is_target.sum()) == len(store)


def test_metrics_on_known_predictions():
    store = _store(1)
    batch = build_batch(store, torch.arange(len(store)), dt=DT, knn=16, radius=10.0)
    perfect = torch.nan_to_num(batch.future)[None]
    shifted = perfect + torch.tensor([0.3, 0.4])  # 0.5 m everywhere
    accumulator = MetricAccumulator()
    batch_metrics(torch.cat([shifted, perfect]), batch, accumulator, dataset_names=store.names)
    result = accumulator.result()
    complete = batch.future_valid.all(dim=-1)
    assert result["agent_ade"] == pytest.approx(0.0, abs=1e-6) and result["target_fde"] == pytest.approx(0.0, abs=1e-6)
    assert result["n_agent_ade"] == int(complete.sum()) and result["n_target_ade"] == int(complete[batch.ptr[:-1]].sum())
    accumulator = MetricAccumulator()
    batch_metrics(shifted, batch, accumulator, dataset_names=store.names)
    assert accumulator.result()["joint_ade"] == pytest.approx(0.5, abs=1e-5)
    assert accumulator.result()["synthetic/agent_ade"] == pytest.approx(0.5, abs=1e-5)


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
