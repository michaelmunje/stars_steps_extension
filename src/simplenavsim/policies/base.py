"""Policy interface, tunable parameter specifications and pedestrian predictors."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import yaml

from simplenavsim.core.state import Observation


@dataclass(frozen=True)
class ParameterSpec:
    """One tunable policy parameter; ``log`` parameters are searched on a log scale."""

    name: str
    default: float
    lower: float
    upper: float
    log: bool = False
    description: str = ""

    def to_unit(self, value: float) -> float:
        if self.log:
            return (math.log(value) - math.log(self.lower)) / (math.log(self.upper) - math.log(self.lower))
        return (value - self.lower) / (self.upper - self.lower)

    def from_unit(self, unit: float) -> float:
        unit = min(1.0, max(0.0, float(unit)))
        if self.log:
            return math.exp(math.log(self.lower) + unit * (math.log(self.upper) - math.log(self.lower)))
        return self.lower + unit * (self.upper - self.lower)


class Policy(ABC):
    """Maps a batch of observations and per-environment parameters ``[B, n]`` to ``(v, w)`` commands."""

    name = "base"
    parameters: tuple[ParameterSpec, ...] = ()

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def default_parameters(self) -> dict[str, float]:
        return {p.name: p.default for p in self.parameters}

    def parameter_tensor(self, values: dict[str, float] | list[dict[str, float]], batch: int, device="cpu") -> torch.Tensor:
        """``[batch, n]`` parameters from one dict (broadcast) or one dict per row; missing names use defaults."""
        rows = values if isinstance(values, list) else [values] * batch
        if len(rows) != batch:
            raise ValueError(f"Expected {batch} parameter sets, got {len(rows)}")
        unknown = {key for row in rows for key in row} - set(self.parameter_names)
        if unknown:
            raise ValueError(f"Unknown {self.name} parameters: {sorted(unknown)}")
        defaults = self.default_parameters()
        return torch.tensor([[float(row.get(name, defaults[name])) for name in self.parameter_names] for row in rows],
                            dtype=torch.float32, device=device)

    @abstractmethod
    def act(self, obs: Observation, params: torch.Tensor) -> torch.Tensor:
        """``[B, 2]`` linear and angular velocity commands."""

    def reset(self, batch_size: int) -> None:
        """Clear any internal state (called at episode start)."""


def save_policy(path, policy: Policy, values: dict[str, float], **extra) -> None:
    with open(path, "w") as stream:
        yaml.safe_dump({"policy": policy.name, "parameters": {k: float(v) for k, v in values.items()}, **extra},
                       stream, sort_keys=False)


# Controller terms removed after they did not help (decisions 78-82); older files saved them as 0.
RETIRED = {"idle_turn_weight", "side_commit_weight", "pass_side_weight"}


def load_policy_file(path) -> tuple[str, dict[str, float]]:
    with open(path) as stream:
        data = yaml.safe_load(stream)
    values = {k: float(v) for k, v in (data.get("parameters") or {}).items()}
    if any(values.get(k, 0.0) != 0.0 for k in RETIRED):
        raise ValueError(f"{path}: uses a retired controller term {sorted(RETIRED & set(values))}")
    return data["policy"], {k: v for k, v in values.items() if k not in RETIRED}


class PedestrianPredictor(ABC):
    """Forecasts visible pedestrians' positions ``[B, P, H, 2]`` at ``dt`` intervals."""

    @abstractmethod
    def predict(self, obs: Observation, steps: int, dt: float) -> torch.Tensor: ...


class ConstantVelocityPredictor(PedestrianPredictor):
    def predict(self, obs: Observation, steps: int, dt: float) -> torch.Tensor:
        t = torch.arange(1, steps + 1, device=obs.ped_pos.device, dtype=obs.ped_pos.dtype) * dt
        return obs.ped_pos[:, :, None] + obs.ped_vel[:, :, None] * t[None, None, :, None]
