"""YAML configuration: simulator, pedestrian model, policy and optimisation settings.

A top-level config (e.g. ``simplenavsim/configs/cmaes_crowd_v4.yaml``) may reference the component configs by
path (``simulator: simulator.yaml``, relative to the referencing file) or inline them as mappings.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
import yaml

from simplenavsim.environments.lightweight.env import LightweightEnv, RobotConfig, SimConfig
from simplenavsim.pedestrians.reactive_replay import ReactiveReplayModel, ReactiveReplayParams
from simplenavsim.pedestrians.replay import ReplayModel
from simplenavsim.pedestrians.social_force import SocialForceModel, SocialForceParams
from simplenavsim.policies.goal_seeking import make_policy
from simplenavsim.policies.social_navigation import ControllerConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "simplenavsim" / "configs"


def load_yaml(path) -> dict:
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def resolve(config: dict, base: Path) -> dict:
    """Replace ``simulator`` / ``pedestrians`` / ``policy`` paths by their loaded mappings."""
    out = dict(config)
    for key in ("simulator", "pedestrians", "policy"):
        value = out.get(key)
        if isinstance(value, str):
            path = Path(value)
            out[key] = load_yaml(path if path.is_absolute() else base / path)
    return out


def load_config(path) -> dict:
    path = Path(path)
    return resolve(load_yaml(path), path.parent)


def _dataclass(cls, values: dict | None):
    values = dict(values or {})
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**values)


def make_pedestrian_model(config: dict | None):
    config = dict(config or {"model": "reactive_replay"})
    name = config.pop("model", "reactive_replay")
    forces = _dataclass(SocialForceParams, config.pop("social_force", None))
    if name == "replay":
        return ReplayModel()
    if name == "social_force":
        return SocialForceModel(forces)
    if name == "reactive_replay":
        reactive = config.pop("reactive_replay", None) or {}
        return ReactiveReplayModel(ReactiveReplayParams(forces=forces, **reactive))
    if name == "stars_crowd":
        from simplenavsim.pedestrians.stars_crowd import StarsCrowdModel

        spec = dict(config.pop("stars_crowd", None) or {})
        path = Path(spec.pop("checkpoint"))
        path = path if path.is_absolute() else CONFIG_DIR.parents[1] / path
        return StarsCrowdModel(path, **spec)
    if name == "stars_social":
        from simplenavsim.pedestrians.stars_crowd import StarsSocialModel

        # Shares the crowd's checkpoint and seeds; the robot acts through forces, not the graph.
        spec = {k: v for k, v in (config.pop("stars_crowd", None) or {}).items() if k != "include_robot"}
        spec.update(config.pop("stars_social", None) or {})
        path = Path(spec.pop("checkpoint"))
        path = path if path.is_absolute() else CONFIG_DIR.parents[1] / path
        return StarsSocialModel(path, forces=forces, **spec)
    raise ValueError(f"Unknown pedestrian model {name!r}")


def make_env(simulator: dict | None, pedestrians: dict | None) -> LightweightEnv:
    simulator = simulator or {}
    return LightweightEnv(make_pedestrian_model(pedestrians), _dataclass(RobotConfig, simulator.get("robot")),
                          _dataclass(SimConfig, simulator.get("sim")))


def make_predictor(spec: dict | None, device="cpu"):
    """``{type: constant_velocity}`` (default) or ``{type: stars, checkpoint, mode, samples, predict_every}``."""
    spec = dict(spec or {})
    kind = spec.pop("type", "constant_velocity")
    if kind == "constant_velocity":
        return None
    if kind == "stars":
        from simplenavsim.predictors.stars import StarsForecastPredictor

        path = Path(spec.pop("checkpoint"))
        path = path if path.is_absolute() else CONFIG_DIR.parents[1] / path
        return StarsForecastPredictor(path, device=device, **spec)
    raise ValueError(f"Unknown predictor {kind!r}")


def make_controller(policy: dict | None, name: str | None = None, robot: RobotConfig | None = None, predictor=None):
    policy = policy or {}
    controller = dict(policy.get("controller") or {})
    if robot is not None:  # the controller plans with the simulated robot's limits
        controller.setdefault("max_speed", robot.max_speed)
        controller.setdefault("reverse_speed", max(0.0, -robot.min_speed))
        controller.setdefault("max_angular_speed", robot.max_angular_speed)
        controller.setdefault("max_acceleration", robot.max_acceleration)
        controller.setdefault("max_angular_acceleration", robot.max_angular_acceleration)
        controller.setdefault("robot_radius", robot.radius)
    return make_policy(name or policy.get("name", "social"), config=_dataclass(ControllerConfig, controller),
                       predictor=predictor)


def device_from(value: str | None) -> torch.device:
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
