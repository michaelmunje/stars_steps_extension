"""STARS-fidelity graph options (ablation ladder L0/L1): batched per-target graphs equal the original
single-target STARS adapter (``stars_data.stars_graph``) and the original upstream VAE encoder."""
import numpy as np
import torch

from baselines.stars_data import stars_graph
from baselines.stars_ma_batch import WindowStore, build_batch, per_target_store


def _store(seed=0, windows=3):
    rng = np.random.default_rng(seed)
    parts, sizes = [], []
    for _ in range(windows):
        n = int(rng.integers(3, 25))
        start = rng.normal(0, 4, (n, 1, 2))
        step = rng.normal(0, 0.5, (n, 1, 2))
        xy = start + step * np.arange(20)[None, :, None] + rng.normal(0, 0.05, (n, 20, 2))
        xy[1, :3] = xy[1, 3]                       # a node standing still at first (heading rule)
        valid = np.ones((n, 20), dtype=bool)
        valid[2, :3] = False                       # partially observed node
        valid[-1, 15:] = False                     # incomplete future
        parts.append((xy.astype(np.float32), valid))
        sizes.append(n)
    cat = lambda i, sl: torch.from_numpy(np.concatenate([p[i][:, sl] for p in parts]))
    ptr = torch.zeros(windows + 1, dtype=torch.long)
    ptr[1:] = torch.tensor(sizes).cumsum(0)
    xy0 = cat(0, slice(None))
    xy0 = torch.where(cat(1, slice(None))[..., None], xy0, torch.zeros_like(xy0))
    v = cat(1, slice(None))
    return WindowStore(xy0[:, :8], v[:, :8], xy0[:, 8:], v[:, 8:], ptr, torch.zeros(windows, dtype=torch.long),
                       torch.arange(windows), ["t"])


def test_per_target_graph_equals_stars_adapter():
    store = _store()
    per, targets = per_target_store(store, targets="scored", max_graph_nodes=18)
    assert len(per) == len(targets)
    batch = build_batch(per, torch.arange(len(per)), dt=0.4, knn=None, radius=None, resample=40,
                        heading_backfill=False, target_slot=True)
    for w in range(len(per)):
        a, b = int(per.ptr[w]), int(per.ptr[w + 1])
        xy = torch.cat([per.obs_xy[a:b], per.future[a:b]], 1).numpy().transpose(1, 0, 2)
        valid = torch.cat([per.obs_valid[a:b], per.future_valid[a:b]], 1).numpy().T
        ref = stars_graph(xy, valid, 8)
        assert b - a <= 18 and len(ref["node_type"]) == b - a
        assert np.array_equal(batch.node_type[a:b].numpy(), ref["node_type"])
        np.testing.assert_allclose(batch.valid_mask[a:b].numpy(), ref["valid_mask"])
        edges = (batch.edge_index[0] >= a) & (batch.edge_index[0] < b)
        assert np.array_equal(batch.edge_index[:, edges].numpy() - a, ref["edge_index"])
        np.testing.assert_allclose(batch.edge_attr[edges].numpy(), ref["edge_attr"], atol=2e-5)


def test_crop_keeps_nearest():
    store = _store(1, windows=2)
    per, targets = per_target_store(store, targets="observed", max_graph_nodes=4)
    for w in range(len(per)):
        a, b = int(per.ptr[w]), int(per.ptr[w + 1])
        assert b - a == min(4, int(store.sizes[int(torch.searchsorted(store.ptr, targets[w], right=True)) - 1]))
        d = torch.linalg.norm(per.obs_xy[a + 1:b, -1] - per.obs_xy[a, -1], dim=-1)
        assert torch.all(d[1:] >= d[:-1])


def test_one_target_per_window():
    store = _store(2, windows=4)
    per, targets = per_target_store(store, targets="one", seed=3)
    assert len(per) == len(store)
    assert bool(per.obs_valid[per.ptr[:-1]].all())


def test_l0_encoder_equals_upstream_vae_on_per_target_graphs():
    """L0 = the unmodified upstream HGNNCombinedVAE: same weights give the same node posterior means."""
    from baselines.stars import build_vae, node_posterior
    from baselines.stars_ma_model import StarsMotionEncoder

    torch.manual_seed(0)
    store = _store(4, windows=2)
    per, _ = per_target_store(store, targets="scored", max_graph_nodes=18)
    batch = build_batch(per, torch.arange(len(per)), dt=0.4, knn=None, radius=None, resample=40,
                        heading_backfill=False, target_slot=True)
    recipe = {"hidden_dim": 32, "latent_dim": 16, "num_mp_layers": 2, "edge_transformer_layers": 1, "num_heads": 4,
              "dropout": 0.0}
    ours = StarsMotionEncoder(8, {**recipe, "use_node_features": False, "timesteps": 40}).eval()
    theirs = build_vae(torch.device("cpu"), recipe).eval()
    theirs.encoder.load_state_dict({**ours.base.state_dict(), "robot_feat_proj.weight": theirs.encoder.robot_feat_proj.weight})
    with torch.no_grad():
        from torch_geometric.data import Data

        data = Data(num_nodes=batch.num_nodes, x=None, node_type=batch.node_type, edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr, valid_mask=batch.valid_mask)
        mu, _ = node_posterior(theirs, data)
        torch.testing.assert_close(ours(batch)["mu"], mu)
