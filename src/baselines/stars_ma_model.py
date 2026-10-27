"""Multi-agent STARS: the pinned STARS graph encoder extended with node motion, plus a forecaster.

Architecture (pedestrian history -> STARS graph encoder -> temporal dynamics -> all futures):

* ``StarsMotionEncoder`` keeps the upstream ``CombinedHGNNEncoder`` modules unmodified -- node-type
  embedding, ``TemporalEdgeTransformer`` over SE(2) edge features, ``TransformerConv`` message
  passing with residual LayerNorm, and the ``to_mu``/``to_logvar`` heads -- and adds a second
  ``TemporalEdgeTransformer`` over each node's own motion features (``stars_ma_graph``), injected
  through a bias-free projection exactly as STARS injects SEAN robot features. With
  ``use_node_features=False`` it is STARS's edge-only encoder. Both temporal transformers run at
  the observation length instead of STARS's 40-step grid (their positional embedding is resized;
  upstream code is untouched).
* ``StarsMotionVAE`` is the phase-1 self-supervised model: STARS's edge reconstruction objective
  (``CombinedHGNNDecoder``'s edge head, at the native length) plus reconstruction of the new node
  features, with STARS's KL term.
* ``StarsMultiAgentForecaster`` decodes every node: its representation, current motion and
  optionally a per-agent latent ``z`` pass one social ``TransformerConv`` round over the same graph,
  then a GRU emits per-step displacements in the agent's local frame. ``latent_dim=0`` is the
  deterministic baseline; ``latent_dim>0`` is a CVAE whose prior and posterior are learnt for the
  *future* (not the representation VAE's latent), and whose ``z`` is an agent's intent: sampled
  once per rollout and held across autoregressive rounds.

Node type 0 is reserved for a robot agent; pedestrians are type 1. A robot node (with its own
features and, later, its planned trajectory) can join the same graph without changing the
pedestrian pathway.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from baselines.stars import upstream_model_module
from baselines.stars_data import EDGE_FEATURE_DIM
from baselines.stars_ma_graph import NODE_FEATURE_DIM, to_scene

# STARS "homo" recipe sizes (stars.VAE_RECIPE) plus the new node-motion branch.
ENCODER_RECIPE = {
    "hidden_dim": 256,
    "latent_dim": 128,
    "num_mp_layers": 2,
    "edge_transformer_layers": 3,
    "node_transformer_layers": 2,
    "num_heads": 4,
    "dropout": 0.2,
    "use_node_features": True,
}
DECODER_RECIPE = {
    "hidden_dim": 256,
    "latent_dim": 0,  # 0 = deterministic; > 0 = CVAE intent latent size
    "representation": "mu",  # STARS's node representation (encode_nodes); "hidden" uses the pre-head state
    "dropout": 0.1,
}


def _retime(transformer: nn.Module, steps: int) -> nn.Module:
    """Resize an upstream ``TemporalEdgeTransformer``'s positional embedding from 40 to ``steps``."""
    hidden = transformer.pos_embed.shape[-1]
    transformer.pos_embed = nn.Parameter(torch.zeros(1, steps, hidden))
    return transformer


# Fused attention kernels cap one call at 65,535 sequences (with dropout); a large batch has
# hundreds of thousands of edge sequences, so the temporal transformers run in chunks.
TEMPORAL_CHUNK = 32768


def _temporal(transformer: nn.Module, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """Run a temporal transformer over ``x [S, T, F]`` in chunks of ``TEMPORAL_CHUNK`` sequences."""
    if x.shape[0] <= TEMPORAL_CHUNK:
        return transformer(x, src_key_padding_mask=padding)
    return torch.cat([transformer(x[i:i + TEMPORAL_CHUNK], src_key_padding_mask=padding[i:i + TEMPORAL_CHUNK])
                      for i in range(0, x.shape[0], TEMPORAL_CHUNK)])


class StarsMotionEncoder(nn.Module):
    """STARS's homogeneous graph encoder at the native rate, with optional node motion features."""

    def __init__(self, observation_steps: int, recipe: dict | None = None) -> None:
        super().__init__()
        recipe = {**ENCODER_RECIPE, **(recipe or {})}
        self.recipe = recipe
        upstream = upstream_model_module()
        self.base = upstream.CombinedHGNNEncoder(
            hidden_dim=recipe["hidden_dim"],
            latent_dim=recipe["latent_dim"],
            num_mp_layers=recipe["num_mp_layers"],
            edge_transformer_layers=recipe["edge_transformer_layers"],
            num_heads=recipe["num_heads"],
            dropout=recipe["dropout"],
        )
        _retime(self.base.edge_encoder, observation_steps)
        # The SEAN robot-feature projection has no inputs here; drop it so it does not sit in the
        # optimiser or the checkpoint as dead weights.
        del self.base.robot_feat_proj
        self.use_node_features = bool(recipe["use_node_features"])
        if self.use_node_features:
            self.node_encoder = _retime(
                upstream.TemporalEdgeTransformer(
                    edge_dim=NODE_FEATURE_DIM,
                    hidden_dim=recipe["hidden_dim"],
                    num_layers=recipe["node_transformer_layers"],
                    num_heads=recipe["num_heads"],
                    dropout=recipe["dropout"],
                ),
                observation_steps,
            )
            self.node_proj = nn.Linear(recipe["hidden_dim"], recipe["hidden_dim"], bias=False)

    @property
    def hidden_dim(self) -> int:
        return int(self.recipe["hidden_dim"])

    @property
    def latent_dim(self) -> int:
        return int(self.recipe["latent_dim"])

    def forward(self, data) -> dict[str, torch.Tensor]:
        """``hidden [N, H]``, ``mu``/``logvar [N, L]`` and ``edge_emb [E, H]`` for a (batched) graph."""
        base = self.base
        hidden = base.node_type_emb(data.node_type)
        if self.use_node_features:
            node_pad = ~data.valid_mask.bool()
            hidden = hidden + self.node_proj(_temporal(self.node_encoder, data.node_attr, node_pad))
        source, destination = data.edge_index
        if source.numel():
            edge_valid = (data.valid_mask[source] * data.valid_mask[destination]).bool()
            edge_emb = _temporal(base.edge_encoder, data.edge_attr, ~edge_valid)
        else:
            # The upstream edge transformer cannot reshape an empty batch; no edges, no messages.
            edge_emb = hidden.new_zeros((0, self.hidden_dim))
        for conv, norm in zip(base.convs, base.norms):
            hidden = norm(hidden + base.dropout(F.gelu(conv(hidden, data.edge_index, edge_attr=edge_emb))))
        return {"hidden": hidden, "mu": base.to_mu(hidden), "logvar": base.to_logvar(hidden), "edge_emb": edge_emb}


class MotionReconstructionDecoder(nn.Module):
    """STARS's edge reconstruction head at length ``steps``, plus a node-motion reconstruction head."""

    def __init__(self, latent_dim: int, hidden_dim: int, steps: int, dropout: float, *, node_features: bool) -> None:
        super().__init__()
        self.steps = int(steps)
        self.node_proj = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, steps * EDGE_FEATURE_DIM),
        )
        self.node_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, steps * NODE_FEATURE_DIM),
        ) if node_features else None

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.node_proj(z)
        source, destination = edge_index
        edges = self.edge_head(torch.cat([hidden[source], hidden[destination]], dim=-1)).view(-1, self.steps, EDGE_FEATURE_DIM)
        nodes = self.node_head(z).view(-1, self.steps, NODE_FEATURE_DIM) if self.node_head is not None else None
        return edges, nodes


class StarsMotionVAE(nn.Module):
    """Phase 1: STARS's self-supervised graph VAE with node motion features at the native rate."""

    def __init__(self, observation_steps: int, recipe: dict | None = None) -> None:
        super().__init__()
        self.encoder = StarsMotionEncoder(observation_steps, recipe)
        r = self.encoder.recipe
        self.decoder = MotionReconstructionDecoder(
            r["latent_dim"], r["hidden_dim"], observation_steps, r["dropout"], node_features=r["use_node_features"]
        )

    def loss(self, data, beta: float) -> dict[str, torch.Tensor]:
        """STARS's valid-masked edge MSE (summed over features, per valid edge-step) + node term + beta * KL."""
        out = self.encoder(data)
        mu, logvar = out["mu"], out["logvar"]
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        edges, nodes = self.decoder(z, data.edge_index)
        source, destination = data.edge_index
        zero = mu.new_zeros(())
        if source.numel():
            edge_valid = (data.valid_mask[source] * data.valid_mask[destination]).unsqueeze(-1)
            edge = ((edges - data.edge_attr) ** 2 * edge_valid).sum() / edge_valid.sum().clamp(min=1.0)
        else:
            edge = zero
        if nodes is not None:
            node_valid = data.valid_mask.unsqueeze(-1)
            node = ((nodes - data.node_attr) ** 2 * node_valid).sum() / node_valid.sum().clamp(min=1.0)
        else:
            node = zero
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return {"loss": edge + node + beta * kl, "edge": edge, "node": node, "kl": kl}


def _mlp(inputs: int, hidden: int, outputs: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, outputs))


class TrajectoryDecoder(nn.Module):
    """Per-agent temporal decoder with one social round; deterministic or CVAE.

    Context for agent ``i``: its encoder representation and its last observed motion state. The
    CVAE prior reads that context, the posterior also reads the agent's (local-frame) future; ``z``
    joins the context *before* the social round, so jointly sampled ``z`` produce mutually
    consistent futures rather than independent per-agent ones.
    """

    def __init__(self, representation_dim: int, edge_dim: int, prediction_steps: int, *, num_heads: int,
                 recipe: dict | None = None) -> None:
        super().__init__()
        recipe = {**DECODER_RECIPE, **(recipe or {})}
        self.recipe = recipe
        hidden, self.latent_dim = int(recipe["hidden_dim"]), int(recipe["latent_dim"])
        self.prediction_steps = int(prediction_steps)
        context = representation_dim + NODE_FEATURE_DIM
        if self.latent_dim:
            self.prior = _mlp(context, hidden, 2 * self.latent_dim, recipe["dropout"])
            self.future_encoder = _mlp(3 * prediction_steps, hidden, hidden, recipe["dropout"])
            self.posterior = _mlp(context + hidden, hidden, 2 * self.latent_dim, recipe["dropout"])
        self.input = nn.Linear(context + self.latent_dim, hidden)
        from torch_geometric.nn import TransformerConv

        self.social = TransformerConv(hidden, hidden // num_heads, heads=num_heads, concat=True, beta=True,
                                      dropout=recipe["dropout"], edge_dim=edge_dim)
        self.social_norm = nn.LayerNorm(hidden)
        self.initial = nn.Linear(hidden, hidden)
        self.cell = nn.GRUCell(2 + hidden, hidden)
        self.output = nn.Linear(hidden, 2)

    def context(self, representation: torch.Tensor, node_attr: torch.Tensor) -> torch.Tensor:
        return torch.cat([representation, node_attr[:, -1]], dim=-1)

    def prior_params(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prior(context).chunk(2, dim=-1)

    def posterior_params(self, context: torch.Tensor, future_local: torch.Tensor, future_valid: torch.Tensor):
        mask = future_valid.unsqueeze(-1).to(future_local.dtype)
        future = torch.cat([torch.nan_to_num(future_local) * mask, mask], dim=-1).flatten(1)
        return self.posterior(torch.cat([context, self.future_encoder(future)], dim=-1)).chunk(2, dim=-1)

    def forward(self, context: torch.Tensor, last_velocity: torch.Tensor, edge_index: torch.Tensor,
                edge_emb: torch.Tensor, dt: float, z: torch.Tensor | None = None) -> torch.Tensor:
        """Local-frame future positions ``[N, P, 2]`` (origin at the last observed position)."""
        if self.latent_dim:
            if z is None:
                raise ValueError("The CVAE decoder needs a latent sample z")
            context = torch.cat([context, z], dim=-1)
        hidden = F.gelu(self.input(context))
        if edge_index.numel():
            hidden = self.social_norm(hidden + self.social(hidden, edge_index, edge_attr=edge_emb))
        # Positions accumulate over the horizon: keep the recurrence in float32 under bf16 autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            state = torch.tanh(self.initial(hidden))
            step = last_velocity.float() * dt
            steps = []
            for _ in range(self.prediction_steps):
                state = self.cell(torch.cat([step, hidden], dim=-1), state)
                step = self.output(state)
                steps.append(step)
            return torch.stack(steps, dim=1).cumsum(dim=1)


class StarsMultiAgentForecaster(nn.Module):
    """STARS encoder -> per-agent temporal decoder; predicts every node of the graph."""

    def __init__(self, observation_steps: int, prediction_steps: int, *, dt: float,
                 encoder_recipe: dict | None = None, decoder_recipe: dict | None = None) -> None:
        super().__init__()
        self.observation_steps, self.prediction_steps, self.dt = int(observation_steps), int(prediction_steps), float(dt)
        self.encoder = StarsMotionEncoder(observation_steps, encoder_recipe)
        self.decoder_recipe = {**DECODER_RECIPE, **(decoder_recipe or {})}
        representation_dim = (self.encoder.latent_dim if self.decoder_recipe["representation"] == "mu"
                              else self.encoder.hidden_dim)
        self.decoder = TrajectoryDecoder(representation_dim, self.encoder.hidden_dim, prediction_steps,
                                         num_heads=self.encoder.recipe["num_heads"], recipe=self.decoder_recipe)
        self.freeze_encoder = False

    @property
    def latent_dim(self) -> int:
        return self.decoder.latent_dim

    def encode(self, data) -> dict[str, torch.Tensor]:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            out = self.encoder(data)
        representation = out["mu"] if self.decoder_recipe["representation"] == "mu" else out["hidden"]
        return {"context": self.decoder.context(representation, data.node_attr), "edge_emb": out["edge_emb"]}

    def sample_latent(self, context: torch.Tensor, samples: int, *, mean: bool = False) -> torch.Tensor | None:
        """``[K, N, Z]`` prior samples (or the prior mean), or None for the deterministic decoder."""
        if not self.latent_dim:
            return None
        mu, logvar = self.decoder.prior_params(context)
        if mean:
            return mu.expand(samples, *mu.shape)
        return mu + torch.exp(0.5 * logvar) * torch.randn((samples, *mu.shape), device=mu.device, dtype=mu.dtype)

    def decode(self, data, encoded: dict, z: torch.Tensor | None) -> torch.Tensor:
        """Local-frame predictions ``[K, N, P, 2]`` for latents ``z [K, N, Z]`` (K=1 if deterministic)."""
        velocity = data.node_attr[:, -1, 2:4]
        latents = [None] if z is None else list(z)
        return torch.stack([
            self.decoder(encoded["context"], velocity, data.edge_index, encoded["edge_emb"], self.dt, z=sample)
            for sample in latents
        ])

    def training_forward(self, data, future_local: torch.Tensor, future_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Teacher-forced prediction with posterior latents; returns ``prediction [N, P, 2]`` and ``kl [N]``."""
        encoded = self.encode(data)
        if not self.latent_dim:
            return {"prediction": self.decode(data, encoded, None)[0], "kl": None}
        q_mu, q_logvar = self.decoder.posterior_params(encoded["context"], future_local, future_valid)
        p_mu, p_logvar = self.decoder.prior_params(encoded["context"])
        z = q_mu + torch.exp(0.5 * q_logvar) * torch.randn_like(q_mu)
        # Agents with no supervised future step cannot inform a posterior; use the prior for them.
        has_future = future_valid.any(dim=-1, keepdim=True)
        prior_z = p_mu + torch.exp(0.5 * p_logvar) * torch.randn_like(p_mu)
        z = torch.where(has_future, z, prior_z.detach())
        kl = 0.5 * (p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu) ** 2) / p_logvar.exp() - 1).sum(-1)
        return {"prediction": self.decode(data, encoded, z[None])[0], "kl": kl * has_future.squeeze(-1)}

    @torch.no_grad()
    def predict(self, data, samples: int = 1) -> torch.Tensor:
        """Local-frame predictions ``[K, N, P, 2]``; the deterministic decoder always returns K=1."""
        encoded = self.encode(data)
        return self.decode(data, encoded, self.sample_latent(encoded["context"], samples))


@torch.no_grad()
def rollout(model: StarsMultiAgentForecaster, batch, *, rounds: int, exec_steps: int,
            z: torch.Tensor | None = None) -> torch.Tensor:
    """Closed-loop multi-agent simulation from a ``stars_ma_batch.GraphBatch``'s observed windows.

    Each round encodes the current scene, predicts ``model.prediction_steps`` ahead, executes the
    first ``exec_steps`` of them for every agent, shifts the window and rebuilds the interaction
    graph from the updated states (``GraphBatch.rebuild``). ``z`` (``[N, Z]``, CVAE only) is each
    agent's intent, held across rounds; a prior sample is drawn from the first round's context
    when it is None. Returns scene-frame positions ``[N, rounds * exec_steps, 2]``.
    """
    if not 1 <= exec_steps <= model.prediction_steps:
        raise ValueError("exec_steps must be between 1 and the decoder horizon")
    window, valid = batch.obs_xy, batch.valid_mask.bool()
    current = batch
    executed = []
    for round_index in range(rounds):
        encoded = model.encode(current)
        if model.latent_dim and z is None:
            z = model.sample_latent(encoded["context"], 1)[0]
        local = model.decode(current, encoded, None if z is None else z[None])[0]
        scene = to_scene(local[:, :exec_steps], current.origin, current.heading)
        executed.append(scene)
        if round_index + 1 < rounds:
            steps = window.shape[1]
            window = torch.cat([window, scene], dim=1)[:, -steps:]
            valid = torch.cat([valid, torch.ones_like(scene[..., 0], dtype=torch.bool)], dim=1)[:, -steps:]
            current = batch.rebuild(window, valid)
    return torch.cat(executed, dim=1)
