"""Shared construction for the ``seanjax/scripts`` command-line tools."""
from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path

from seanjax.core.config import CONFIG_DIR, device_from, load_config, make_controller, make_env
from seanjax.optimization.cmaes import CMAESConfig
from seanjax.optimization.evaluator import Evaluator, Objective
from seanjax.policies.base import load_policy_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = CONFIG_DIR / "cmaes.yaml"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def check_backend(backend: str) -> None:
    if backend == "isaac_lab":
        raise SystemExit("The Isaac Lab backend is Phase 3 and not implemented yet; use --backend lightweight "
                         "(see docs/decisions.md for the Isaac Sim status on this cluster).")
    if backend != "lightweight":
        raise SystemExit(f"Unknown backend {backend!r}")


def recording_pool(config: dict):
    """A RecordingPool from ``recorded:`` ({source: steps|csv, path, limit_per_split, max_pedestrians})."""
    spec = config.get("recorded")
    if not spec:
        return None
    from seanjax.datasets.trajectory_loader import RecordingPool

    path = Path(spec.get("path", "datasets/steps"))
    path = path if path.is_absolute() else ROOT / path
    options = {k: spec[k] for k in ("max_pedestrians", "min_pedestrians") if k in spec}
    if spec.get("source", "steps") == "csv":
        return RecordingPool.from_csv(path, **options)
    return RecordingPool.from_steps(path, limit_per_split=spec.get("limit_per_split"), **options)


def build(config_path=None, *, policy: str | None = None, pedestrian_model: str | None = None, device=None):
    """``(config, env, policy, evaluator, pool, device)`` from a top-level YAML config.

    ``policy`` is a policy name (``social`` / ``goal_seeking``) or a saved policy YAML whose
    parameters are returned in ``config["policy_parameters"]``.
    """
    config = load_config(config_path or DEFAULT_CONFIG)
    simulator, pedestrians = config.get("simulator") or {}, dict(config.get("pedestrians") or {})
    if pedestrian_model:
        pedestrians["model"] = pedestrian_model
    device = device_from(device or simulator.get("device"))
    env = make_env(simulator, pedestrians)
    policy_config = config.get("policy") or {}
    values = dict(policy_config.get("parameters") or {})
    name = policy_config.get("name", "social")
    if policy and Path(policy).suffix in {".yaml", ".yml"}:
        name, loaded = load_policy_file(policy)
        values.update(loaded)
    elif policy:
        name = policy
    controller = make_controller(policy_config, name, env.robot)
    config["policy_parameters"] = {**controller.default_parameters(), **values}
    objective = Objective(**(config.get("objective") or {}))
    evaluator = Evaluator(env, controller, objective, device=device, max_envs=config.get("max_envs", 4096))
    return config, env, controller, evaluator, recording_pool(config), device


def cmaes_config(config: dict, **overrides) -> CMAESConfig:
    values = dict(config.get("cmaes") or {})
    values.update({k: v for k, v in overrides.items() if v is not None})
    names = {f.name for f in fields(CMAESConfig)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"cmaes: unknown keys {sorted(unknown)}")
    return CMAESConfig(**values)
