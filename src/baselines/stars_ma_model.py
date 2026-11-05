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
  once per rollout and held across autoregressive rounds. ``diffusion=True`` replaces the GRU with
  a per-agent denoising diffusion model over the future (``DiffusionDecoder``; teacher-forced
  forecasting only, no closed-loop rollout).

Node type 0 is reserved for a robot agent; pedestrians are type 1. A robot node (with its own
features and, later, its planned trajectory) can join the same graph without changing the
pedestrian pathway.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from baselines.stars import upstream_model_module
from baselines.stars_data import EDGE_FEATURE_DIM
from baselines.stars_ma_graph import NODE_FEATURE_DIM, build_graph, to_local, to_scene

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
    "social_layer": True,
    "representation": "mu",  # STARS's node representation (encode_nodes); "hidden" uses the pre-head state
    "dropout": 0.1,
    "diffusion": False,       # True = DiffusionDecoder (latent_dim must be 0)
    "diffusion_steps": 100,   # DDPM training steps T (cosine schedule)
    "sample_steps": 10,       # DDIM steps at sampling time
    "diffusion_scale": 0.1,   # metres; constant-velocity residual steps / this have ~unit scale on STEPS
    "diffusion_social_layers": 0,  # > 0: joint (scene-level) diffusion, social layers inside the denoiser
    "goal": False,            # True: the decoder context also holds each agent's goal (goal_features)
}
GOAL_DIM = 4
GOAL_SCALE = 10.0  # metres per unit of the goal features


def goal_features(data) -> torch.Tensor:
    """``[N, 4]`` goal input: goal minus current position in the agent's local frame (dx, dy), its
    distance (all / ``GOAL_SCALE``) and valid; zeros where ``data.goal`` is missing or invalid."""
    goal = getattr(data, "goal", None)
    if goal is None:
        return data.origin.new_zeros(data.origin.shape[0], GOAL_DIM)
    valid = goal[:, 2:3] > 0
    local = to_local(goal[:, None, :2].to(data.origin.dtype), data.origin, data.heading)[:, 0]
    local = torch.where(valid, local / GOAL_SCALE, torch.zeros_like(local))
    return torch.cat([local, torch.linalg.norm(local, dim=-1, keepdim=True), valid.to(local.dtype)], dim=-1)


def _retime(transformer: nn.Module, steps: int) -> nn.Module:
    """Resize an upstream ``TemporalEdgeTransformer``'s positional embedding from 40 to ``steps``."""
    hidden = transformer.pos_embed.shape[-1]
    transformer.pos_embed = nn.Parameter(torch.zeros(1, steps, hidden))
    return transformer


# Fused attention kernels cap one call at 65,535 sequences (with dropout); a large batch has
# hundreds of thousands of edge sequences, so the temporal transformers run in chunks.
TEMPORAL_CHUNK = 32768


def _temporal(transformer: nn.Module, x: torch.Tensor, padding: torch.Tensor, checkpoint: bool = False) -> torch.Tensor:
    """Run a temporal transformer over ``x [S, T, F]`` in chunks; ``checkpoint`` recomputes activations
    in the backward pass instead of storing them (much less memory, ~1/3 more compute)."""
    def run(chunk, mask):
        if checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as recompute

            return recompute(lambda a, b: transformer(a, src_key_padding_mask=b), chunk, mask, use_reentrant=False)
        return transformer(chunk, src_key_padding_mask=mask)

    if x.shape[0] <= TEMPORAL_CHUNK:
        return run(x, padding)
    return torch.cat([run(x[i:i + TEMPORAL_CHUNK], padding[i:i + TEMPORAL_CHUNK])
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
        self.checkpoint_temporal = False  # runtime memory option, not part of the recipe
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
            hidden = hidden + self.node_proj(_temporal(self.node_encoder, data.node_attr, node_pad, self.checkpoint_temporal))
        source, destination = data.edge_index
        if source.numel():
            edge_valid = (data.valid_mask[source] * data.valid_mask[destination]).bool()
            edge_emb = _temporal(base.edge_encoder, data.edge_attr, ~edge_valid, self.checkpoint_temporal)
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

    def loss(self, data, beta: float, node_weight: float = 1.0) -> dict[str, torch.Tensor]:
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
        return {"loss": edge + node_weight * node + beta * kl, "edge": edge, "node": node, "kl": kl}


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
        context = representation_dim + NODE_FEATURE_DIM + (GOAL_DIM if recipe["goal"] else 0)
        if self.latent_dim:
            self.prior = _mlp(context, hidden, 2 * self.latent_dim, recipe["dropout"])
            self.future_encoder = _mlp(3 * prediction_steps, hidden, hidden, recipe["dropout"])
            self.posterior = _mlp(context + hidden, hidden, 2 * self.latent_dim, recipe["dropout"])
        self.input = nn.Linear(context + self.latent_dim, hidden)
        if recipe["social_layer"]:
            from torch_geometric.nn import TransformerConv

            self.social = TransformerConv(hidden, hidden // num_heads, heads=num_heads, concat=True, beta=True,
                                          dropout=recipe["dropout"], edge_dim=edge_dim)
            self.social_norm = nn.LayerNorm(hidden)
        else:
            self.social = None
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
        if self.social is not None and edge_index.numel():
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


class DiffusionDecoder(nn.Module):
    """Per-agent DDPM over the future (MID-style, Gu et al. CVPR 2022), on TrajectoryDecoder's context.

    The context and the one social ``TransformerConv`` round are TrajectoryDecoder's, giving a
    per-agent hidden ``h``. The diffused variable ``x0`` is the agent's 12 per-step displacements in
    its local frame minus constant-velocity steps (``last_velocity * dt``), divided by
    ``diffusion_scale``. Cosine noise schedule (Nichol & Dhariwal 2021) with ``diffusion_steps``
    steps, epsilon prediction by a residual MLP on ``[x_t, t embedding, h]``, masked MSE loss;
    deterministic DDIM sampling (eta 0) with ``sample_steps`` steps. Agents are sampled
    independently given their socially mixed ``h``.

    With ``diffusion_social_layers`` N > 0 the decoder is *joint* (MotionDiffuser-style, Jiang et
    al. CVPR 2023): N social ``TransformerConv`` layers over the scene graph (encoder edge
    embeddings) sit inside the denoiser, after the first N MLP blocks, so every agent's noisy future
    attends to its neighbours' at every step. Training draws one noise level per graph (the scene
    is denoised coherently); sampling runs each of the K chains over the whole scene at once, so
    sample k is one mutually consistent joint future.
    """

    def __init__(self, representation_dim: int, edge_dim: int, prediction_steps: int, *, num_heads: int,
                 recipe: dict | None = None) -> None:
        super().__init__()
        recipe = {**DECODER_RECIPE, **(recipe or {})}
        self.recipe, self.latent_dim = recipe, 0
        hidden, self.prediction_steps = int(recipe["hidden_dim"]), int(prediction_steps)
        self.scale, self.sample_steps = float(recipe["diffusion_scale"]), int(recipe["sample_steps"])
        self.input = nn.Linear(representation_dim + NODE_FEATURE_DIM + (GOAL_DIM if recipe["goal"] else 0), hidden)
        if recipe["social_layer"]:
            from torch_geometric.nn import TransformerConv

            self.social = TransformerConv(hidden, hidden // num_heads, heads=num_heads, concat=True, beta=True,
                                          dropout=recipe["dropout"], edge_dim=edge_dim)
            self.social_norm = nn.LayerNorm(hidden)
        else:
            self.social = None
        size = 2 * self.prediction_steps
        self.time = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.denoise_in = nn.Linear(size + 2 * hidden, hidden)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                                  nn.Linear(hidden, hidden)) for _ in range(3))
        self.denoise_out = nn.Linear(hidden, size)
        self.joint = int(recipe["diffusion_social_layers"])
        if self.joint:
            from torch_geometric.nn import TransformerConv

            self.denoise_social = nn.ModuleList(
                TransformerConv(hidden, hidden // num_heads, heads=num_heads, concat=True, beta=True,
                                dropout=recipe["dropout"], edge_dim=edge_dim) for _ in range(self.joint))
            self.denoise_social_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(self.joint))
        steps = int(recipe["diffusion_steps"])
        f = torch.cos((torch.arange(steps + 1, dtype=torch.float64) / steps + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, 0).float(), persistent=False)

    def context(self, representation: torch.Tensor, node_attr: torch.Tensor) -> torch.Tensor:
        return torch.cat([representation, node_attr[:, -1]], dim=-1)

    def hidden(self, context: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input(context))
        if self.social is not None and edge_index.numel():
            hidden = self.social_norm(hidden + self.social(hidden, edge_index, edge_attr=edge_emb))
        return hidden.float()

    def epsilon(self, x: torch.Tensor, t: torch.Tensor, hidden: torch.Tensor, edge_index: torch.Tensor | None = None,
                edge_emb: torch.Tensor | None = None) -> torch.Tensor:
        """Predicted noise ``[M, 2P]`` for noisy futures ``x [M, 2P]`` at integer steps ``t [M]`` (the
        joint decoder also mixes agents over ``edge_index`` / ``edge_emb``)."""
        half = hidden.shape[-1] // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
        angle = t.to(x.dtype)[:, None] * freq[None]
        y = self.denoise_in(torch.cat([x, self.time(torch.cat([angle.sin(), angle.cos()], -1)), hidden], -1))
        for i, block in enumerate(self.blocks):
            y = y + block(y)
            if i < self.joint and edge_index is not None and edge_index.numel():
                y = self.denoise_social_norm[i](y + self.denoise_social[i](y, edge_index, edge_attr=edge_emb.float()))
        return self.denoise_out(F.gelu(y))

    def loss(self, hidden: torch.Tensor, future_local: torch.Tensor, future_valid: torch.Tensor,
             last_velocity: torch.Tensor, dt: float, edge_index: torch.Tensor | None = None,
             edge_emb: torch.Tensor | None = None, graph: torch.Tensor | None = None) -> torch.Tensor:
        """Masked epsilon MSE over valid displacement steps (both ends of a step observed)."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            future = torch.nan_to_num(future_local.float())
            previous = torch.cat([torch.zeros_like(future[:, :1]), future[:, :-1]], dim=1)
            valid = future_valid & torch.cat([torch.ones_like(future_valid[:, :1]), future_valid[:, :-1]], dim=1)
            mask = valid[..., None].expand_as(future).flatten(1).float()
            x0 = ((future - previous - last_velocity.float()[:, None] * dt) / self.scale).flatten(1) * mask
            if self.joint and graph is not None:   # one noise level per scene
                t = torch.randint(0, len(self.alpha_bar), (int(graph.max()) + 1,), device=x0.device)[graph]
            else:
                t = torch.randint(0, len(self.alpha_bar), (len(x0),), device=x0.device)
            noise = torch.randn_like(x0)
            a = self.alpha_bar[t][:, None]
            x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
            error = (self.epsilon(x_t, t, hidden, edge_index, edge_emb) - noise).square() * mask
            return error.sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor, last_velocity: torch.Tensor, dt: float, samples: int,
               edge_index: torch.Tensor | None = None, edge_emb: torch.Tensor | None = None,
               noise: torch.Tensor | None = None) -> torch.Tensor:
        """Local-frame future positions ``[K, N, P, 2]`` from K DDIM chains (per agent, or per scene for
        the joint decoder: the K copies of the graph are batched as one disconnected graph)."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            N, P = len(hidden), self.prediction_steps
            h = hidden.repeat(samples, 1)
            x = noise.reshape(samples * N, 2 * P).clone() if noise is not None else torch.randn(samples * N, 2 * P, device=h.device)
            if self.joint and edge_index is not None:
                offset = (torch.arange(samples, device=h.device) * N).repeat_interleave(edge_index.shape[1])
                edge_index = edge_index.repeat(1, samples) + offset
                edge_emb = edge_emb.float().repeat(samples, 1)
            else:
                edge_index = edge_emb = None
            times = torch.linspace(len(self.alpha_bar) - 1, 0, self.sample_steps, device=h.device).round().long().unique_consecutive()
            for i, t in enumerate(times):
                a = self.alpha_bar[t]
                eps = self.epsilon(x, t.expand(len(x)), h, edge_index, edge_emb)
                x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-10.0, 10.0)
                a_next = self.alpha_bar[times[i + 1]] if i + 1 < len(times) else torch.ones_like(a)
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            steps = x.view(samples, N, P, 2) * self.scale + last_velocity.float()[None, :, None] * dt
            return steps.cumsum(dim=2)


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
        self.diffusion = bool(self.decoder_recipe["diffusion"])
        if self.diffusion and self.decoder_recipe["latent_dim"]:
            raise ValueError("The diffusion decoder has no CVAE latent (latent_dim must be 0)")
        decoder = DiffusionDecoder if self.diffusion else TrajectoryDecoder
        self.decoder = decoder(representation_dim, self.encoder.hidden_dim, prediction_steps,
                               num_heads=self.encoder.recipe["num_heads"], recipe=self.decoder_recipe)
        self.freeze_encoder = False

    @property
    def latent_dim(self) -> int:
        return self.decoder.latent_dim

    @property
    def stochastic(self) -> bool:
        """Whether ``predict`` draws K different futures (CVAE or diffusion)."""
        return self.diffusion or bool(self.latent_dim)

    def encode(self, data) -> dict[str, torch.Tensor]:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            out = self.encoder(data)
        representation = out["mu"] if self.decoder_recipe["representation"] == "mu" else out["hidden"]
        context = self.decoder.context(representation, data.node_attr)
        if self.decoder_recipe["goal"]:
            context = torch.cat([context, goal_features(data).to(context.dtype)], dim=-1)
        return {"context": context, "edge_emb": out["edge_emb"]}

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
        if self.diffusion:
            raise ValueError("The diffusion decoder only supports predict() (no closed-loop rollout or crowd)")
        velocity = data.node_attr[:, -1, 2:4]
        latents = [None] if z is None else list(z)
        return torch.stack([
            self.decoder(encoded["context"], velocity, data.edge_index, encoded["edge_emb"], self.dt, z=sample)
            for sample in latents
        ])

    def training_forward(self, data, future_local: torch.Tensor, future_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Teacher-forced prediction with posterior latents; returns ``prediction [N, P, 2]`` and ``kl [N]``."""
        encoded = self.encode(data)
        if self.diffusion:
            hidden = self.decoder.hidden(encoded["context"], data.edge_index, encoded["edge_emb"])
            graph = getattr(data, "batch", None)
            denoise = self.decoder.loss(hidden, future_local, future_valid, data.node_attr[:, -1, 2:4], self.dt,
                                        data.edge_index, encoded["edge_emb"], graph)
            return {"prediction": None, "kl": None, "z": None, "denoise": denoise}
        if not self.latent_dim:
            return {"prediction": self.decode(data, encoded, None)[0], "kl": None, "z": None}
        q_mu, q_logvar = self.decoder.posterior_params(encoded["context"], future_local, future_valid)
        p_mu, p_logvar = self.decoder.prior_params(encoded["context"])
        z = q_mu + torch.exp(0.5 * q_logvar) * torch.randn_like(q_mu)
        # Agents with no supervised future step cannot inform a posterior; use the prior for them.
        has_future = future_valid.any(dim=-1, keepdim=True)
        prior_z = p_mu + torch.exp(0.5 * p_logvar) * torch.randn_like(p_mu)
        z = torch.where(has_future, z, prior_z.detach())
        kl = 0.5 * (p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu) ** 2) / p_logvar.exp() - 1).sum(-1)
        return {"prediction": self.decode(data, encoded, z[None])[0], "kl": kl * has_future.squeeze(-1), "z": z}

    @torch.no_grad()
    def predict(self, data, samples: int = 1) -> torch.Tensor:
        """Local-frame predictions ``[K, N, P, 2]``; the deterministic decoder always returns K=1."""
        encoded = self.encode(data)
        if self.diffusion:
            hidden = self.decoder.hidden(encoded["context"], data.edge_index, encoded["edge_emb"])
            return self.decoder.sample(hidden, data.node_attr[:, -1, 2:4], self.dt, samples, data.edge_index,
                                       encoded["edge_emb"])
        return self.decode(data, encoded, self.sample_latent(encoded["context"], samples))


def _rebuild(batch, window: torch.Tensor, valid: torch.Tensor, *, dt: float, knn: int | None, radius: float | None):
    """New batched graph from per-node scene-frame windows ``[N, T, 2]``, graph by graph."""
    from torch_geometric.data import Batch, Data

    graphs = []
    pointer = batch.ptr.tolist()
    for start, stop in zip(pointer[:-1], pointer[1:]):
        graph = build_graph(window[start:stop].transpose(0, 1), valid[start:stop].T, dt=dt, knn=knn, radius=radius)
        graphs.append(Data(num_nodes=stop - start, **graph))
    return Batch.from_data_list(graphs)


def rollout(model: StarsMultiAgentForecaster, batch, *, rounds: int, exec_steps: int, knn: int | None,
            radius: float | None, z: torch.Tensor | None = None, keep_graph: bool = False) -> torch.Tensor:
    """Closed-loop multi-agent simulation from a batch's observed windows.

    Each round encodes the current scene, predicts ``model.prediction_steps`` ahead, executes the
    first ``exec_steps`` of them for every agent, shifts the window and rebuilds the interaction
    graph from the updated states. ``z`` (``[N, Z]``, CVAE only) is each agent's intent, held
    across rounds; a prior sample is drawn from the first round's context when it is None.
    Returns scene-frame positions ``[N, rounds * exec_steps, 2]``. With ``keep_graph`` the rollout
    is differentiable (multi-step training); otherwise it runs without gradients.
    """
    if model.diffusion:
        raise ValueError("Closed-loop rollout is not implemented for the diffusion decoder")
    if not 1 <= exec_steps <= model.prediction_steps:
        raise ValueError("exec_steps must be between 1 and the decoder horizon")
    with torch.set_grad_enabled(keep_graph and torch.is_grad_enabled()):
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
                if hasattr(batch, "rebuild"):  # stars_ma_batch.GraphBatch: all graphs at once on the device
                    current = batch.rebuild(window, valid)
                else:
                    current = _rebuild(batch, window, valid, dt=model.dt, knn=knn, radius=radius)
        return torch.cat(executed, dim=1)
