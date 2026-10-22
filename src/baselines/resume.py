"""Crash-safe continuation of an existing training stage.

Resume checkpoints are deliberately distinct from transfer-initialization checkpoints:
the former restore the complete optimizer/scheduler/training state, while the latter load
model weights only and start a fresh optimization stage.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ``_clean`` is the single JSON serialisation contract for every file in a run directory
# (status.py), and config.json -- the source of a checkpoint's embedded ``run_config`` --
# is written through it.  The resume guard reuses it so a live value and the recorded one
# are compared in the same canonical form (Path vs str, tuple vs list, dataclass vs dict).
from baselines.status import _clean

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Which run-configuration fields a resume may legitimately change
# ---------------------------------------------------------------------------
# The guard is deliberately INVERTED (L-005/L-017): ``resume_guard_config`` compares
# *every* field a trainer records in its run configuration and exempts only the names
# listed below.  A newly added trainer argument is therefore guarded by default, instead
# of being silently unguarded until somebody remembers to extend an inclusion list --
# which is exactly how ``--validation-mode`` went unguarded in all three trainers.
#
# Adding a name here is a deliberate scientific statement: "changing this field halfway
# through a run cannot change what the run measures." Do NOT add a field that selects,
# sizes, orders or truncates the training or validation population (``max_scenes``,
# ``max_recordings``, ``max_val_batches``, ``validation_mode``, ...) or that defines a
# metric (``validation_k``, ``validation_stride``, ``validation_max_scenes``).
#
# Two classes are exempt, and only these two:
#
# 1. Runtime/placement/logging knobs, which change how the same computation is executed.
# 2. *Path-shaped references* whose content is separately digested by a guarded field --
#    a registry alias and an absolute path denote the same file, so the path string is not
#    the scientific fact.  ``transfer_config`` is covered by ``transfer_optimization_hash``,
#    ``upstream_source``/``upstream_config`` by ``upstream_commit``, ``forecast_config`` by
#    the guarded ``forecasting_config`` dict and ``forecast_preprocessing_hash``, and the transfer
#    ``checkpoint`` by the guarded ``pretraining_seed`` /
#    ``pretraining_best_validation_metric`` read out of it (a resume never reloads those
#    weights: the model state comes from the resume checkpoint).
#
# Caveat worth knowing rather than hiding: ``num_workers`` is exempt as a throughput knob,
# but ``seed_worker`` seeds each loader worker from its own index, so changing it does
# perturb the augmentation RNG stream after the resume boundary.  It changes neither the
# population nor the metric, which is the bar for this list.
#
# Fields that ``RunStatus.__init__`` adds to config.json (``git_commit``, ``git_dirty``,
# the ``environment_info()`` snapshot, ``canonical_*``) are not part of any trainer's
# ``run_config`` dict and so are not guarded at all -- deliberately, since a resume after
# a commit, a library upgrade or a machine change must stay possible.
RESUME_EXEMPT_FIELDS: frozenset[str] = frozenset(
    {
        # 1. runtime / placement / logging
        "resume",  # the --resume flag itself: None on the original launch, "auto" now
        "device",
        "device_realized",
        "num_workers",
        "status_every",
        "iterations",  # a stop condition, like a SIGINT -- not a definition of the run
        "list_forecast_presets",  # introspection flags; both exit before training
        "show_resolved_forecast_config",
        # 2. path-shaped references whose content is digested by a guarded field
        "run_dir",
        "checkpoint",
        "checkpoint_reference",
        "pretraining_checkpoint",
        "transfer_config",
        "forecast_config",
        "upstream_source",
        "upstream_config",
    }
)

# Per-trainer additions to the exemption list.  Same rule as above: a name here asserts
# that changing the field mid-run is scientifically harmless.
RESUME_EXEMPT_FIELDS_BY_TRAINER: dict[str, frozenset[str]] = {
    "moflow": frozenset(),
    "social_vae": frozenset(),
    "trajectron": frozenset(
        {
            # Chooses between rebuilding the prepared environments and reading the
            # persistent cache; either way the environments themselves are identical.
            "no_environment_cache",
        }
    ),
    # Bypasses the finetune checkpoint-metadata guard at startup. It gates a
    # startup check rather than the experiment, and a resume never reloads those
    # weights.
    "_shared_transfer": frozenset({"allow_legacy_checkpoint"}),
}


def resume_guard_config(run_config: dict[str, Any], trainer: str) -> dict[str, Any]:
    """Return the run-configuration fields a resume must match, by exempting the rest.

    ``trainer`` selects the per-trainer additions in
    :data:`RESUME_EXEMPT_FIELDS_BY_TRAINER`; an unknown name is a programming error and
    raises rather than silently guarding everything.
    """
    if trainer not in RESUME_EXEMPT_FIELDS_BY_TRAINER:
        raise KeyError(
            f"Unknown trainer {trainer!r} for the resume guard; known: "
            f"{sorted(RESUME_EXEMPT_FIELDS_BY_TRAINER)}"
        )
    exempt = (
        RESUME_EXEMPT_FIELDS
        | RESUME_EXEMPT_FIELDS_BY_TRAINER[trainer]
        | RESUME_EXEMPT_FIELDS_BY_TRAINER["_shared_transfer"]
    )
    return {key: value for key, value in run_config.items() if key not in exempt}


def _as_recorded(value: Any) -> Any:
    """Canonicalise a value into the form config.json records it in.

    The embedded ``run_config`` is ``json.loads`` of a file written by ``atomic_json``,
    so a live tuple, ``Path`` or dataclass must be put through the same transformation
    before being compared, or a resume would be refused over a representation
    difference. Applied to both sides, so the comparison is of recorded meaning.
    """
    return json.loads(json.dumps(_clean(value), sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class ResumeState:
    checkpoint: Path
    payload: dict[str, Any]
    epoch: int
    iteration: int
    batch_in_epoch: int
    epoch_complete: bool
    best: float
    bad_epochs: int
    best_epoch: int | None
    # Guarded fields this checkpoint predates: recorded and warned about rather than
    # refused. See load_resume_state.
    unguarded_fields: tuple[str, ...] = field(default_factory=tuple)


def add_resume_argument(parser) -> None:
    parser.add_argument(
        "--resume",
        "--resuming",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help=(
            "Continue this run with full model/optimizer/scheduler state. Without a path, "
            "load RUN_DIR/checkpoints/latest.pt. --resuming is a compatibility alias."
        ),
    )


def resolve_resume_checkpoint(run_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(run_dir) / "checkpoints" / "latest.pt" if value == "auto" else Path(value).expanduser()
    if path.is_dir():
        path = path / "checkpoints" / "latest.pt"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def _optimizer_to_device(optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.to(device)


def load_resume_state(
    checkpoint: Path,
    *,
    stage: str,
    optimization: dict[str, Any],
    expected_run_config: dict[str, Any],
    model_key: str = "model",
) -> ResumeState:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {model_key, "optimizer", "epoch", "iteration", "stage", "optimization"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Resume checkpoint {checkpoint} is missing {missing}")
    if payload["stage"] != stage:
        raise ValueError(
            f"Resume stage mismatch: checkpoint={payload['stage']!r}, requested={stage!r}"
        )
    if payload["optimization"] != optimization:
        raise ValueError(
            "Resume optimization does not match the checkpoint. Reuse the original learning "
            "rate, batch size, epoch budget, scheduler, and early-stopping settings."
        )
    saved_config = payload.get("run_config")
    if not isinstance(saved_config, dict) or not saved_config:
        raise ValueError("Resume checkpoint has no embedded run configuration")
    mismatches = []
    unguarded = []
    for key, expected in expected_run_config.items():
        if key not in saved_config:
            # This checkpoint was written before the field existed, so it records no
            # value to compare against.  Treated as "unknown", not as a mismatch: the
            # alternative -- comparing a missing key as None -- would refuse every
            # in-flight run's checkpoint the moment a trainer gains an argument, and the
            # error would name a value the checkpoint never held.  The field is instead
            # reported here and logged by the trainer, so the gap is visible in
            # training.log rather than silent.  Newer checkpoints record the field and
            # are fully guarded.  A field explicitly recorded as null IS guarded.
            unguarded.append(key)
            continue
        if _as_recorded(saved_config[key]) != _as_recorded(expected):
            mismatches.append(f"{key}: checkpoint={saved_config[key]!r}, requested={expected!r}")
    if mismatches:
        raise ValueError("Resume run configuration mismatch: " + "; ".join(mismatches))
    if unguarded:
        logger.warning(
            "Resume checkpoint %s predates %d guarded run-configuration field(s) and "
            "records no value for them, so they could not be checked: %s",
            checkpoint,
            len(unguarded),
            ", ".join(sorted(unguarded)),
        )
    stopping = payload.get("early_stopping") or {}
    selection = payload.get("checkpoint_selection") or {}
    best = stopping.get("best", selection.get("value"))
    return ResumeState(
        checkpoint=checkpoint,
        payload=payload,
        epoch=int(payload["epoch"]),
        iteration=int(payload["iteration"]),
        batch_in_epoch=int(payload.get("batch_in_epoch", 0)),
        epoch_complete=bool(payload.get("epoch_complete", True)),
        best=float("inf") if best is None else float(best),
        bad_epochs=int(stopping.get("bad_epochs", 0)),
        best_epoch=selection.get("best_epoch"),
        unguarded_fields=tuple(sorted(unguarded)),
    )


def restore_optimizer_scheduler_state(
    state: ResumeState, optimizer, scheduler, device: torch.device
) -> None:
    optimizer.load_state_dict(state.payload["optimizer"])
    _optimizer_to_device(optimizer, device)
    saved_scheduler = state.payload.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("Resume scheduler presence does not match the current configuration")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)


def restore_training_state(state: ResumeState, model, optimizer, scheduler, device: torch.device) -> None:
    model.load_state_dict(state.payload["model"])
    restore_optimizer_scheduler_state(state, optimizer, scheduler, device)


def checkpoint_resume_fields(
    *,
    stopper,
    batch_in_epoch: int,
    epoch_complete: bool,
    loader_generator_epoch_state: torch.Tensor | None,
    loader_generator_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "early_stopping": {
            "best": None if stopper.best == float("inf") else stopper.best,
            "bad_epochs": stopper.bad_epochs,
            "patience": stopper.patience,
            "min_delta": stopper.min_delta,
        },
        "batch_in_epoch": int(batch_in_epoch),
        "epoch_complete": bool(epoch_complete),
        "loader_generator_epoch_state": loader_generator_epoch_state,
        "loader_generator_state": loader_generator_state,
        "rng_state": capture_rng_state(),
    }
