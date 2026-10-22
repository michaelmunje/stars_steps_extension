"""Thin integration around the pinned STARS homogeneous GNN-VAE (``HGNNCombinedVAE``).

The encoder and VAE stay in ``references/stars/src/models/hgnn_combined_v2.py``. This module
wraps shared forecasting scenes as STARS graphs (``stars_data.py``), adds the forecasting head,
and converts predictions back to metric world coordinates.

Forecasting follows STARS's own probe protocol (``train_swm_forecasting_linear_probe_v1.py``):
a frozen pretrained encoder and a linear head trained with MSE. The head reads
``[z_target ‖ mean(z_neighbours)]`` -- STARS's ``encode`` pooling with the target in the robot
slot -- plus the target's observed history in its own frame, and predicts all prediction
steps at once. STARS edges are relative poses only, so without the history a pedestrian with
no neighbours would carry no information about its own motion.

Deterministic by default (the latent mean, K=1). ``latent_samples > 1`` instead draws node
latents from the encoder's posterior, giving K predictions for best-of-K evaluation.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from baselines.stars_data import NODE_TYPE_HUMAN, NODE_TYPE_TARGET, stars_graph
from normalization import inverse_scene


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "references" / "stars"
UPSTREAM_REPOSITORY = "https://github.com/ut-amrl/stars"
UPSTREAM_COMMIT = "13d840c5c0677ff52e928eea1679ba5ce1f19c14"

# STARS paper "homo" pipeline (src/hetero_scripts/run_train_vae.sh --pipeline homo) on top of
# src/params/config.yaml homognn_params. beta_warmup_epochs is 50, not the script's
# --warmup_epochs 25: train_hgnn_unified_v2.py reads beta_warmup_epochs first and the config
# sets it, so 50 is what that run used.
VAE_RECIPE = {
    "hidden_dim": 256,
    "latent_dim": 128,
    "num_mp_layers": 2,
    "edge_transformer_layers": 3,
    "num_heads": 4,
    "dropout": 0.2,
    "learning_rate": 0.00011564,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "batch_size": 32,
    "max_epochs": 1000,
    "beta_max": 0.9,
    "beta_warmup_epochs": 50,
    "patience": 20,
    "min_delta": 1e-4,
}
# STARS forecasting linear probe (train_swm_forecasting_linear_probe_v1.py defaults).
HEAD_RECIPE = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 8,
    "max_epochs": 80,
    "lr_patience": 8,
    "patience": 14,
    "min_delta": 1e-4,
}


def _load_pinned(relative: str, name: str):
    path = REFERENCE / relative
    if not path.is_file():
        raise FileNotFoundError(f"Pinned STARS source is missing: {path}. Run `git submodule update --init references/stars`.")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned STARS module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def upstream_model_module():
    return _load_pinned("src/models/hgnn_combined_v2.py", "_steps_pinned_stars_hgnn_combined_v2")


def build_vae(device: torch.device, recipe: dict | None = None) -> nn.Module:
    """Instantiate the unmodified upstream ``HGNNCombinedVAE``."""
    recipe = {**VAE_RECIPE, **(recipe or {})}
    model = upstream_model_module().HGNNCombinedVAE(
        hidden_dim=recipe["hidden_dim"],
        latent_dim=recipe["latent_dim"],
        num_mp_layers=recipe["num_mp_layers"],
        edge_transformer_layers=recipe["edge_transformer_layers"],
        num_heads=recipe["num_heads"],
        dropout=recipe["dropout"],
    )
    return model.to(device)


class StarsDataset(Dataset):
    """Adapt shared ``Scene`` objects to one STARS graph per target sample."""

    def __init__(self, source, observation_steps: int, *, limit: int | None = None) -> None:
        self.source = source
        self.observation_steps = int(observation_steps)
        self.limit = min(len(source), limit) if limit is not None else len(source)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        from torch_geometric.data import Data

        scene = self.source[index]
        xy = np.asarray(scene.xy, dtype=np.float32)
        valid = np.asarray(scene.valid, dtype=bool)
        if not valid[:, 0].all():
            raise ValueError("STARS target pedestrian must be complete for all observed and future steps")
        graph = stars_graph(xy, valid, self.observation_steps)
        future = xy[self.observation_steps:, 0]
        target_world = np.asarray(scene.target_xy_world, dtype=np.float32)[:, 0]  # [pred, agents, 2] -> target
        return Data(
            node_type=torch.from_numpy(graph["node_type"]),
            edge_index=torch.from_numpy(graph["edge_index"]),
            edge_attr=torch.from_numpy(graph["edge_attr"]),
            valid_mask=torch.from_numpy(graph["valid_mask"]),
            num_nodes=len(graph["node_type"]),
            history=torch.from_numpy(graph["history"])[None],
            future=torch.from_numpy(np.array(future, copy=True))[None],
            target_world_xy=torch.from_numpy(np.array(target_world, copy=True))[None],
            rotation=torch.tensor([float(scene.rotation)], dtype=torch.float64),
            center=torch.from_numpy(np.asarray(scene.center, dtype=np.float64))[None],
            sample_index=torch.tensor([int(index)]),
        )


def node_posterior(model: nn.Module, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node ``(mu, logvar)`` from the upstream encoder.

    A batch in which no target has a neighbour has no edges, and the upstream temporal edge
    transformer cannot reshape an empty tensor. Its output would be empty anyway, so that case
    runs the same upstream node embedding and message-passing layers with an empty edge set.
    """
    encoder = model.encoder
    if batch.edge_index.shape[1]:
        hidden = encoder._encode_h(batch)
    else:
        import torch.nn.functional as F

        hidden = encoder.node_type_emb(batch.node_type)
        empty = hidden.new_zeros((0, hidden.shape[-1]))
        for conv, norm in zip(encoder.convs, encoder.norms):
            hidden = norm(hidden + encoder.dropout(F.gelu(conv(hidden, batch.edge_index, edge_attr=empty))))
    return encoder.to_mu(hidden), encoder.to_logvar(hidden)


def pooled_latent(model: nn.Module, batch, *, sample: bool = False) -> torch.Tensor:
    """``[z_target ‖ mean(z_neighbours)]`` per graph, as STARS's ``HGNNCombinedVAE.encode``.

    ``sample`` draws node latents from the posterior instead of using the mean.
    """
    from torch_geometric.utils import scatter

    mu, logvar = node_posterior(model, batch)
    node_latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
    graphs = int(batch.num_graphs)
    target = batch.node_type == NODE_TYPE_TARGET
    humans = batch.node_type == NODE_TYPE_HUMAN
    z_target = scatter(node_latent[target], batch.batch[target], dim=0, reduce="mean", dim_size=graphs)
    z_people = scatter(node_latent[humans], batch.batch[humans], dim=0, reduce="mean", dim_size=graphs)
    return torch.cat([z_target, z_people], dim=-1)


class ForecastHead(nn.Module):
    """Linear head: ``[pooled latent ‖ flattened target history]`` -> all prediction steps."""

    def __init__(self, latent_dim: int, observation_steps: int, prediction_steps: int, *, use_history: bool = True) -> None:
        super().__init__()
        self.observation_steps = int(observation_steps)
        self.prediction_steps = int(prediction_steps)
        self.use_history = bool(use_history)
        inputs = 2 * latent_dim + (2 * observation_steps if use_history else 0)
        self.linear = nn.Linear(inputs, 2 * prediction_steps)

    def features(self, latent: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if not self.use_history:
            return latent
        return torch.cat([latent, history.reshape(history.shape[0], -1)], dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).view(-1, self.prediction_steps, 2)


DEFAULT_MAX_EDGES = 8192


def edge_chunks(batch, max_edges: int = DEFAULT_MAX_EDGES) -> list:
    """Split a batch into consecutive sub-batches of at most ``max_edges`` edges.

    Dense scenes give fully connected graphs with thousands of edges, and every edge carries a
    40-step temporal transformer, so memory scales with edges rather than graphs. A single graph
    larger than the budget forms its own chunk.
    """
    from torch_geometric.data import Batch

    if batch.edge_index.shape[1] <= max_edges:
        return [batch]
    chunks, current, edges = [], [], 0
    for graph in batch.to_data_list():
        size = int(graph.edge_index.shape[1])
        if current and edges + size > max_edges:
            chunks.append(Batch.from_data_list(current))
            current, edges = [], 0
        current.append(graph)
        edges += size
    if current:
        chunks.append(Batch.from_data_list(current))
    return chunks


def _edge_valid(batch) -> torch.Tensor:
    source, destination = batch.edge_index
    return (batch.valid_mask[source] * batch.valid_mask[destination]).unsqueeze(-1)


def vae_loss_normalizers(chunks: list, latent_dim: int) -> tuple[float, float]:
    """Whole-batch denominators of STARS's loss: valid (edge, step) pairs and latent entries.

    STARS divides the masked squared error by ``edge_valid.unsqueeze(-1).sum()``, the number of
    valid (edge, step) pairs -- not times the 8 features -- so its edge term sums over features.
    """
    edge_entries = sum(float(_edge_valid(chunk).sum()) for chunk in chunks)
    latent_entries = sum(int(chunk.num_nodes) for chunk in chunks) * latent_dim
    return edge_entries, float(latent_entries)


def vae_loss(model: nn.Module, chunk, beta: float, edge_entries: float, latent_entries: float) -> dict[str, torch.Tensor]:
    """One chunk's share of STARS's ``train_hgnn_unified_v2.run_epoch`` batch loss.

    Valid-masked edge MSE plus ``beta`` times the mean KL, each divided by the whole batch's
    denominators, so summing the chunks reproduces the full-batch loss exactly. The node-feature term is zero
    for every node here (STEPS has no robot features), as STARS treats SSUB robots.
    """
    if chunk.edge_index.shape[1]:
        edge_recon, _node_recon, mu, logvar = model(chunk)
        edge_sum = ((edge_recon - chunk.edge_attr) ** 2 * _edge_valid(chunk)).sum()
    else:
        mu, logvar = node_posterior(model, chunk)
        edge_sum = mu.new_zeros(())
    kl_sum = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    edge = edge_sum / max(edge_entries, 1.0)
    kl = kl_sum / max(latent_entries, 1.0)
    return {"loss": edge + beta * kl, "edge": edge, "kl": kl}


def predictions_to_world(prediction: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    """``[K, B, T, 2]`` model-frame predictions -> world frame, per sample's stored transform."""
    world = np.empty_like(prediction, dtype=np.float64)
    for index in range(prediction.shape[1]):
        world[:, index] = inverse_scene(
            prediction[:, index].transpose(1, 0, 2), float(rotation[index]), center[index]
        ).transpose(1, 0, 2)
    return world


@torch.no_grad()
def predict(model: nn.Module, head: ForecastHead, batch, *, latent_samples: int = 1) -> torch.Tensor:
    """Model-frame predictions ``[K, B, T, 2]``; K=1 is the deterministic latent-mean forecast."""
    if latent_samples == 1:
        latents = [pooled_latent(model, batch)]
    else:
        latents = [pooled_latent(model, batch, sample=True) for _ in range(latent_samples)]
    return torch.stack([head(head.features(latent, batch.history)) for latent in latents])


def best_of_k(model: nn.Module, head: ForecastHead, batch, *, latent_samples: int = 1, max_edges: int = DEFAULT_MAX_EDGES):
    """Best-of-K ADE/FDE in metres against each sample's world-frame target, as the other baselines."""
    prediction = torch.cat(
        [predict(model, head, chunk, latent_samples=latent_samples) for chunk in edge_chunks(batch, max_edges)], dim=1
    ).cpu().numpy()
    world = predictions_to_world(prediction, batch.rotation.cpu().numpy(), batch.center.cpu().numpy())
    target = batch.target_world_xy.cpu().numpy()
    distance = np.linalg.norm(world - target[None], axis=-1)  # K, B, T
    return distance.mean(axis=-1).min(axis=0), distance[..., -1].min(axis=0), world
