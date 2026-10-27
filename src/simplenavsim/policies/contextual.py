"""Context-dependent controller settings: a small MLP adjusts the social controller's tuned
parameters at every control step, from the same egocentric observation the learned policy uses.

The social controller (``SocialNavigationPolicy``) does the planning and keeps its hard safety
constraint; the network only moves each tuned parameter within its CMA-ES search range, in that
range's unit (log or linear) scale: ``unit = clamp(base_unit + max_shift * tanh(MLP(features)))``.
The last layer starts at zero, so the initial policy *is* the base policy (e.g. v4) and evolution
strategies improve from there. Context features use constant-velocity pedestrian extrapolation, so
the controller's own (STARS) predictor and its forecast schedule are untouched. Decision 92.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from simplenavsim.policies.base import ConstantVelocityPredictor, Policy
from simplenavsim.policies.learned import OBS_DIM, LearnedConfig, LearnedPolicy

TUNED = ("goal_weight", "pedestrian_weight", "disruption_weight", "preferred_speed", "personal_space_radius",
         "prediction_horizon", "smoothness_weight", "speed_scale", "retreat_scale", "alignment_weight")


class ContextualPolicy(Policy):
    name = "contextual"
    parameters = ()

    def __init__(self, controller, base_values: dict[str, float], *, hidden: int = 32, max_shift: float = 0.5,
                 tuned=TUNED, learned_config: LearnedConfig | None = None) -> None:
        self.controller, self.hidden, self.max_shift = controller, int(hidden), float(max_shift)
        self.tuned = tuple(tuned)
        self.base_values = {**controller.default_parameters(), **base_values}
        self.observer = LearnedPolicy(learned_config, ConstantVelocityPredictor())   # features only
        self.shapes = [(OBS_DIM, self.hidden), (self.hidden,), (self.hidden, len(self.tuned)), (len(self.tuned),)]
        self.dim = sum(int(np.prod(s)) for s in self.shapes)
        specs = {p.name: p for p in controller.parameters}
        self._specs = [specs[name] for name in self.tuned]
        self._index = [controller.parameter_names.index(name) for name in self.tuned]
        self.weights: np.ndarray | None = None
        self.last_info: dict = {}

    @property
    def predictor(self):
        return self.controller.predictor

    def reset(self, batch_size: int) -> None:
        self.controller.reset(batch_size)

    def default_parameters(self) -> dict[str, float]:
        return {}

    def parameter_tensor(self, values, batch: int, device="cpu") -> torch.Tensor:
        if isinstance(values, dict):
            if self.weights is None:
                raise ValueError("ContextualPolicy has no weights loaded")
            values = self.weights
        if isinstance(values, list):
            return torch.as_tensor(np.stack([np.asarray(v, np.float32) for v in values]), device=device)
        return torch.as_tensor(np.asarray(values, np.float32), device=device)[None].expand(batch, -1)

    def init_weights(self, seed: int = 0) -> np.ndarray:
        """Random first layer, zero output layer: the initial policy equals the base policy."""
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0, 1 / math.sqrt(OBS_DIM), self.shapes[0])
        return np.concatenate([w1.ravel(), np.zeros(self.dim - w1.size)]).astype(np.float32)

    def _unit_to_value(self, unit: torch.Tensor, spec) -> torch.Tensor:
        unit = unit.clamp(0.0, 1.0)
        if spec.log:
            return torch.exp(math.log(spec.lower) + unit * (math.log(spec.upper) - math.log(spec.lower)))
        return spec.lower + unit * (spec.upper - spec.lower)

    def controller_parameters(self, obs, theta: torch.Tensor) -> torch.Tensor:
        """``[B, n]`` controller parameters for this step."""
        B = theta.shape[0]
        features, _ = self.observer.features(obs)
        start = 0
        parts = []
        for shape in self.shapes:
            n = int(np.prod(shape))
            parts.append(theta[:, start:start + n].reshape(B, *shape))
            start += n
        w1, b1, w2, b2 = parts
        shift = self.max_shift * torch.tanh(torch.einsum("bh,bhk->bk", torch.tanh(torch.einsum("bi,bih->bh", features, w1) + b1), w2) + b2)
        params = self.controller.parameter_tensor(self.base_values, B, theta.device).clone()
        for k, (spec, column) in enumerate(zip(self._specs, self._index)):
            base_unit = spec.to_unit(self.base_values[spec.name])
            params[:, column] = self._unit_to_value(base_unit + shift[:, k], spec)
        return params

    def act(self, obs, params: torch.Tensor) -> torch.Tensor:
        command = self.controller.act(obs, self.controller_parameters(obs, params.float()))
        self.last_info = self.controller.last_info
        return command
