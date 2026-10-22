from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from baselines import stars_data
from baselines.stars import (
    REFERENCE,
    UPSTREAM_COMMIT,
    ForecastHead,
    StarsDataset,
    _load_pinned,
    best_of_k,
    build_vae,
    edge_chunks,
    predict,
    vae_loss,
    vae_loss_normalizers,
)
from data import Scene
from forecasting_config import resolve_forecasting_config
from forecasting_representation import prepare_forecast_representation

ROOT = Path(__file__).resolve().parents[1]
TINY = {"hidden_dim": 16, "latent_dim": 8, "edge_transformer_layers": 1, "num_heads": 2, "dropout": 0.0}


def _upstream_swm():
    return _load_pinned("src/dataloaders/swm_segment_forecasting_dataset.py", "_test_pinned_stars_swm")


def _scene(world_xy: np.ndarray, valid: np.ndarray) -> Scene:
    config = resolve_forecasting_config(preset="steps_default")
    representation = prepare_forecast_representation(world_xy, valid, config, training=False, rng=np.random.default_rng(0))
    agents = world_xy.shape[1]
    return Scene(
        xy=representation.xy, primary_track_id="r:1", neighbor_track_ids=[f"r:{i + 2}" for i in range(agents - 1)],
        dataset="synthetic", recording="r", start_step=0, scene_tag=0, sub_tags=[], valid=valid,
        rotation=representation.transform.rotation, center=representation.transform.origin,
        canonical_xy=world_xy, target_xy_world=representation.target_world_xy,
        coordinate_frame=representation.transform.frame,
    )


def _walkers(agents: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = rng.uniform(-5, 5, size=(agents, 2))
    velocity = rng.uniform(-0.5, 0.5, size=(agents, 2))
    time = np.arange(20, dtype=np.float32)[:, None, None]
    return (start[None] + velocity[None] * time + 0.05 * rng.normal(size=(20, agents, 2))).astype(np.float32)


def _batch(sizes: list[int]) -> Batch:
    graphs = []
    for seed, agents in enumerate(sizes):
        world = _walkers(agents, seed)
        graphs.append(StarsDataset([_scene(world, np.ones((20, agents), dtype=bool))], 8)[0])
    return Batch.from_data_list(graphs)


def test_vectorised_edge_features_match_upstream_helpers():
    upstream = _upstream_swm()
    rng = np.random.default_rng(1)
    xy = rng.normal(size=(8, 5, 2)).astype(np.float32)
    xy[3:5, 2] = xy[2, 2]  # a pause keeps the last heading
    theta = stars_data.headings_from_motion(xy)
    np.testing.assert_allclose(theta, np.stack([upstream._xy_to_theta(xy[:, n]) for n in range(5)], axis=1), atol=1e-6)
    edges = stars_data.fully_connected_edges(5)
    pairs = [(i, j) for i in range(5) for j in range(5) if i != j]
    assert edges.T.tolist() == [list(pair) for pair in pairs]
    orientation = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    expected = np.zeros((len(pairs), 8, 8), dtype=np.float32)
    for e, (i, j) in enumerate(pairs):
        previous = None
        for t in range(8):
            relative = upstream._se2_relative_pose(xy[t, i], orientation[t, i], xy[t, j], orientation[t, j])
            expected[e, t] = np.concatenate([relative, np.zeros(4) if previous is None else relative - previous])
            previous = relative
    features = stars_data.edge_features(xy, theta, edges)
    np.testing.assert_allclose(features, expected, atol=1e-5)
    np.testing.assert_allclose(stars_data.resample_linear(features), upstream._interpolate_edge_attr(expected, 40), atol=1e-5)


def test_graph_uses_target_as_node_zero_and_only_observed_steps():
    world = _walkers(5)
    valid = np.ones((20, 5), dtype=bool)
    valid[:8, 3] = False          # neighbour seen only in the future
    valid[:7, 4] = False          # neighbour seen once in the observation window
    scene = _scene(world, valid)
    graph = stars_data.stars_graph(scene.xy, valid, 8)
    assert graph["node_type"].tolist() == [0, 1, 1]
    assert graph["node_agent_index"].tolist() == [0, 1, 2]
    assert graph["edge_attr"].shape == (6, 40, 8)
    np.testing.assert_allclose(graph["history"], scene.xy[:8, 0])
    changed = np.array(scene.xy, copy=True)
    changed[8:] += 100.0          # the future must not reach the graph
    again = stars_data.stars_graph(changed, valid, 8)
    np.testing.assert_array_equal(again["edge_attr"], graph["edge_attr"])
    np.testing.assert_array_equal(again["valid_mask"], graph["valid_mask"])


def test_edges_without_a_shared_valid_step_are_dropped():
    world = _walkers(3)
    valid = np.ones((20, 3), dtype=bool)
    valid[4:8, 1] = False         # neighbour 1 seen at steps 0-3
    valid[0:4, 2] = False         # neighbour 2 seen at steps 4-7
    graph = stars_data.stars_graph(_scene(world, valid).xy, valid, 8)
    pairs = set(map(tuple, graph["edge_index"].T.tolist()))
    assert (1, 2) not in pairs and (2, 1) not in pairs
    assert {(0, 1), (1, 0), (0, 2), (2, 0)} <= pairs
    mask = graph["valid_mask"]
    assert (mask[graph["edge_index"][0]] * mask[graph["edge_index"][1]]).any(axis=1).all()


def test_predictions_map_back_to_world_targets():
    scene = _scene(_walkers(4), np.ones((20, 4), dtype=bool))
    data = StarsDataset([scene], 8)[0]
    from baselines.stars import predictions_to_world

    world = predictions_to_world(data.future.numpy()[None], data.rotation.numpy(), data.center.numpy())
    np.testing.assert_allclose(world[0, 0], scene.target_xy_world[:, 0], atol=1e-4)


def test_chunked_loss_and_gradients_equal_the_full_batch_stars_loss(monkeypatch):
    torch.manual_seed(0)
    model = build_vae(torch.device("cpu"), TINY).eval()  # eval: no dropout noise
    # The upstream encoder samples z on every forward pass, and the same seed yields different
    # noise for differently shaped tensors; zero it so both paths decode the same latents.
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    batch = _batch([3, 5, 1, 4])
    beta = 0.3

    # STARS train_hgnn_unified_v2.run_epoch, written out on the whole batch.
    edge_recon, _node, mu, logvar = model(batch)
    source, destination = batch.edge_index
    edge_valid = batch.valid_mask[source] * batch.valid_mask[destination]
    expected_edge = ((edge_recon - batch.edge_attr) ** 2 * edge_valid.unsqueeze(-1)).sum() / edge_valid.unsqueeze(-1).sum()
    expected_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    (expected_edge + beta * expected_kl).backward()
    expected_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad()
    chunks = edge_chunks(batch, max_edges=20)
    assert len(chunks) > 1 and all(int(c.edge_index.shape[1]) <= 20 or c.num_graphs == 1 for c in chunks)
    normalizers = vae_loss_normalizers(chunks, TINY["latent_dim"])
    total = {"edge": 0.0, "kl": 0.0}
    for chunk in chunks:
        losses = vae_loss(model, chunk, beta, *normalizers)
        losses["loss"].backward()
        total["edge"] += float(losses["edge"].detach())
        total["kl"] += float(losses["kl"].detach())
    assert abs(total["edge"] - float(expected_edge)) < 1e-5 * max(1.0, float(expected_edge))
    assert abs(total["kl"] - float(expected_kl)) < 1e-6
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    for got, want in zip(grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)


def test_deterministic_forecast_is_identical_with_and_without_chunking():
    torch.manual_seed(0)
    model = build_vae(torch.device("cpu"), TINY).eval()
    head = ForecastHead(TINY["latent_dim"], 8, 12).eval()
    batch = _batch([1, 1, 4, 6, 2])  # includes graphs without neighbours
    ade, fde, world = best_of_k(model, head, batch, max_edges=10)
    full_ade, full_fde, full_world = best_of_k(model, head, batch, max_edges=10_000)
    np.testing.assert_allclose(world, full_world, atol=1e-5)
    np.testing.assert_allclose(ade, full_ade, atol=1e-5)
    assert ade.shape == fde.shape == (5,)
    assert tuple(predict(model, head, batch, latent_samples=3).shape) == (3, 5, 12, 2)
    isolated = _batch([1, 1])  # no edges at all
    assert tuple(predict(model, head, isolated).shape) == (1, 2, 12, 2)


def test_uses_the_pinned_upstream_model():
    model = build_vae(torch.device("cpu"), TINY)
    assert Path(inspect.getsourcefile(type(model))).resolve() == (REFERENCE / "src/models/hgnn_combined_v2.py").resolve()
    upstream = json.loads((ROOT / "src/baselines/upstream.json").read_text())
    assert upstream["stars"]["commit"] == UPSTREAM_COMMIT
