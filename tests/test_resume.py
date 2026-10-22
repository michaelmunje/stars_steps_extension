from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from baselines import resume
from baselines.early_stopping import EarlyStopping
from baselines.resume import (
    add_resume_argument,
    checkpoint_resume_fields,
    load_resume_state,
    resolve_resume_checkpoint,
    restore_training_state,
    resume_guard_config,
)


def _checkpoint(path: Path, *, stage: str = "pretrain") -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    stopper = EarlyStopping(5, 1e-4)
    stopper.update(0.75)
    run_config = {
        "model": "test",
        "stage": stage,
        "seed": 3,
        "forecast_preprocessing_hash": "sha256:test",
    }
    generator = torch.Generator().manual_seed(7)
    epoch_state = generator.get_state()
    torch.randperm(10, generator=generator)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "stage": stage,
            "epoch": 4,
            "iteration": 42,
            "optimization": {"optimizer": "Adam", "learning_rate": 1e-3},
            "checkpoint_selection": {"value": 0.75, "best_epoch": 3},
            "run_config": run_config,
            **checkpoint_resume_fields(
                stopper=stopper,
                batch_in_epoch=2,
                epoch_complete=False,
                loader_generator_epoch_state=epoch_state,
                loader_generator_state=generator.get_state(),
            ),
        },
        path,
    )
    return model, optimizer


def test_resume_and_resuming_are_equivalent_cli_aliases(tmp_path: Path):
    parser = argparse.ArgumentParser()
    add_resume_argument(parser)
    assert parser.parse_args(["--resume"]).resume == "auto"
    assert parser.parse_args(["--resuming"]).resume == "auto"
    explicit = tmp_path / "checkpoint.pt"
    assert parser.parse_args(["--resuming", str(explicit)]).resume == str(explicit)


def test_auto_resume_resolves_latest_checkpoint(tmp_path: Path):
    latest = tmp_path / "checkpoints" / "latest.pt"
    latest.parent.mkdir()
    latest.touch()
    assert resolve_resume_checkpoint(tmp_path, "auto") == latest.resolve()


def test_resume_restores_full_training_state(tmp_path: Path):
    path = tmp_path / "latest.pt"
    original, original_optimizer = _checkpoint(path)
    expected_config = {
        "model": "test",
        "stage": "pretrain",
        "seed": 3,
        "forecast_preprocessing_hash": "sha256:test",
    }
    state = load_resume_state(
        path,
        stage="pretrain",
        optimization={"optimizer": "Adam", "learning_rate": 1e-3},
        expected_run_config=expected_config,
    )
    assert (state.epoch, state.iteration, state.batch_in_epoch, state.epoch_complete) == (4, 42, 2, False)
    assert state.best == pytest.approx(0.75)
    assert state.bad_epochs == 0
    assert state.best_epoch == 3

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=9e-2)
    restore_training_state(state, restored, restored_optimizer, None, torch.device("cpu"))
    for expected, actual in zip(original.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    assert restored_optimizer.state_dict()["state"].keys() == original_optimizer.state_dict()["state"].keys()
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_resume_rejects_changed_training_contract(tmp_path: Path):
    path = tmp_path / "latest.pt"
    _checkpoint(path)
    with pytest.raises(ValueError, match="optimization does not match"):
        load_resume_state(
            path,
            stage="pretrain",
            optimization={"optimizer": "Adam", "learning_rate": 2e-3},
            expected_run_config={},
        )
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_resume_state(
            path,
            stage="pretrain",
            optimization={"optimizer": "Adam", "learning_rate": 1e-3},
            expected_run_config={"seed": 99},
        )


# ---------------------------------------------------------------------------
# Inverted resume guard (L-005 / L-017)
# ---------------------------------------------------------------------------
# The guard compares every recorded run-configuration field except the explicit
# exemptions in baselines.resume.  These tests pin the three properties that matter:
# population/metric-affecting fields refuse a resume, a *newly added* field is guarded
# without anyone editing an inclusion list, and the exempt operational fields do not
# refuse a resume.

# Fields RunStatus.__init__ adds to config.json after the trainer builds run_config, so
# they are never part of a trainer's run_config dict and never guarded (src/baselines/status.py).
RUNSTATUS_ADDED_FIELDS = frozenset(
    {
        "git_commit",
        "git_dirty",
        "python_version",
        "platform",
        "torch_version",
        "cuda_available",
        "cuda_device_name",
        "canonical_release",
        "canonical_schema_version",
        "canonical_hz",
    }
)

# One representative run_config per trainer, carrying the fields L-017 identified as
# unguarded-and-material plus the operational fields that must stay exempt.
TRAINER_RUN_CONFIGS = {
    "moflow": {
        "model": "moflow_teacher",
        "stage": "finetune",
        "seed": 0,
        "data_seed": 42,
        "validation_mode": "grouped",
        "max_scenes": None,
        "max_val_batches": None,
        "forecast_preprocessing_hash": "sha256:abc",
        "num_workers": 4,
        "device": "cuda",
        "device_realized": "cuda",
        "status_every": 100,
        "run_dir": "experiments/pretrained/run",
        "resume": None,
    },
    "social_vae": {
        "model": "social_vae",
        "stage": "finetune",
        "seed": 0,
        "data_seed": 42,
        "validation_mode": "grouped",
        "max_val_batches": None,
        "hidden_dim": 256,
        "neighborhood_radius": 2.0,
        "k": 20,
        "forecast_preprocessing_hash": "sha256:abc",
        "num_workers": 2,
        "device": "cuda",
        "device_realized": "cuda",
        "status_every": 25,
        "run_dir": "experiments/pretrained/run",
        "resume": None,
    },
    "trajectron": {
        "stage": "finetune",
        "seed": 0,
        "data_seed": 42,
        "pretrain_dataset": "ptdpes",
        "fold": "eth",
        "batch_size": 128,
        "validation_mode": "grouped",
        "max_recordings": None,
        "validation_k": 20,
        "validation_stride": 20,
        "validation_max_scenes": 10,
        "forecast_preprocessing_hash": "sha256:abc",
        "upstream_commit": "1031c7b",
        "no_environment_cache": False,
        "device": "cuda",
        "device_realized": "cuda",
        "status_every": 100,
        "run_dir": "experiments/pretrained/run",
        "resume": None,
    },
}

# (trainer, field, recorded value, value requested on the resume). Every one of these
# changes the training population, the validation population, or the definition of the
# checkpoint-selection metric, so every one must refuse the resume.
POPULATION_AFFECTING_CHANGES = [
    ("moflow", "validation_mode", "grouped", "none"),
    ("moflow", "max_scenes", None, 500),
    ("moflow", "max_val_batches", None, 10),
    ("social_vae", "validation_mode", "grouped", "none"),
    ("social_vae", "max_val_batches", None, 10),
    ("trajectron", "validation_mode", "grouped", "none"),
    ("trajectron", "validation_mode", "none", "grouped"),
    ("trajectron", "max_recordings", None, 3),
    ("trajectron", "validation_k", 20, 5),
    ("trajectron", "validation_stride", 20, 1),
    ("trajectron", "validation_max_scenes", 10, 50),
]


def _guarded_checkpoint(path: Path, run_config: dict, *, model_key: str = "model") -> None:
    """A minimal resumable checkpoint carrying an arbitrary embedded run configuration."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.save(
        {
            model_key: model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "stage": run_config["stage"],
            "epoch": 7,
            "iteration": 70,
            "optimization": {"optimizer": "Adam", "learning_rate": 1e-3},
            "run_config": run_config,
        },
        path,
    )


def _load(path: Path, run_config: dict, trainer: str, *, model_key: str = "model"):
    return load_resume_state(
        path,
        stage=run_config["stage"],
        optimization={"optimizer": "Adam", "learning_rate": 1e-3},
        expected_run_config=resume_guard_config(run_config, trainer),
        model_key=model_key,
    )


@pytest.mark.parametrize("trainer,field,recorded,requested", POPULATION_AFFECTING_CHANGES)
def test_resume_refuses_a_changed_population_or_metric_field(
    tmp_path: Path, trainer: str, field: str, recorded, requested
):
    recorded_config = {**TRAINER_RUN_CONFIGS[trainer], field: recorded}
    requested_config = {**recorded_config, field: requested}
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)

    # The recorded configuration resumes cleanly, so the refusal below is attributable
    # to this one field and not to an unrelated mismatch.
    assert _load(path, recorded_config, trainer).epoch == 7
    with pytest.raises(ValueError, match=f"configuration mismatch: {field}:"):
        _load(path, requested_config, trainer)


def test_every_population_affecting_field_is_in_the_guarded_set():
    for trainer, field, recorded, _requested in POPULATION_AFFECTING_CHANGES:
        guarded = resume_guard_config({**TRAINER_RUN_CONFIGS[trainer], field: recorded}, trainer)
        assert field in guarded, f"{field} is not guarded for {trainer}"


def test_a_newly_added_argument_is_guarded_without_editing_any_list():
    """The inversion's whole point: a field nobody has heard of is guarded by default."""
    for trainer, run_config in TRAINER_RUN_CONFIGS.items():
        invented = {**run_config, "max_agents_per_scene_that_nobody_exempted": 4}
        guarded = resume_guard_config(invented, trainer)
        assert "max_agents_per_scene_that_nobody_exempted" in guarded
        # and nothing in the module mentions it, i.e. no list was edited to achieve that
        assert "max_agents_per_scene_that_nobody_exempted" not in (
            resume.RESUME_EXEMPT_FIELDS | set().union(*resume.RESUME_EXEMPT_FIELDS_BY_TRAINER.values())
        )


def test_a_newly_added_argument_refuses_a_resume_when_it_differs(tmp_path: Path):
    recorded_config = {**TRAINER_RUN_CONFIGS["trajectron"], "synthetic_population_cap": 100}
    requested_config = {**recorded_config, "synthetic_population_cap": 7}
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config, model_key="models")
    assert _load(path, recorded_config, "trajectron", model_key="models").epoch == 7
    with pytest.raises(ValueError, match="configuration mismatch: synthetic_population_cap:"):
        _load(path, requested_config, "trajectron", model_key="models")


@pytest.mark.parametrize("trainer", sorted(TRAINER_RUN_CONFIGS))
def test_resume_accepts_changed_operational_fields(tmp_path: Path, trainer: str):
    """num_workers and the other exempt runtime knobs must not refuse a resume."""
    recorded_config = TRAINER_RUN_CONFIGS[trainer]
    resumed_config = {
        **recorded_config,
        "num_workers": recorded_config.get("num_workers", 4) + 5,
        "device": "cpu",
        "device_realized": "cpu",
        "status_every": 1,
        "resume": "auto",  # None on the original launch, "auto" on every resume
        "run_dir": "/somewhere/else/run",
        "iterations": 12,
    }
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    state = _load(path, resumed_config, trainer)
    assert (state.epoch, state.iteration) == (7, 70)
    assert state.unguarded_fields == ()


def test_a_field_the_checkpoint_predates_is_reported_not_refused(tmp_path: Path, caplog):
    """An older checkpoint records no value for a later field, so it cannot be compared.

    Refusing it would make every trainer argument addition break in-flight runs, and the
    error would name a value the checkpoint never held. It is reported instead.
    """
    recorded_config = dict(TRAINER_RUN_CONFIGS["moflow"])
    requested_config = {**recorded_config, "field_added_after_this_checkpoint": "grouped"}
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    with caplog.at_level("WARNING", logger="baselines.resume"):
        state = _load(path, requested_config, "moflow")
    assert state.unguarded_fields == ("field_added_after_this_checkpoint",)
    assert "field_added_after_this_checkpoint" in caplog.text


def test_a_field_recorded_as_null_is_still_guarded(tmp_path: Path):
    """Absent is "unknown"; an explicitly recorded null is a value and is compared."""
    recorded_config = {**TRAINER_RUN_CONFIGS["moflow"], "validation_mode": None}
    requested_config = {**recorded_config, "validation_mode": "grouped"}
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    with pytest.raises(ValueError, match="configuration mismatch: validation_mode:"):
        _load(path, requested_config, "moflow")


def test_guard_compares_recorded_meaning_not_python_representation(tmp_path: Path):
    """config.json round-trips through JSON, so tuple/list and Path/str must compare equal."""
    recorded_config = {
        **TRAINER_RUN_CONFIGS["moflow"],
        "source_datasets": ["ptdpes"],
        "upstream_source": "references/MoFlow",
    }
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    resumed_config = {**recorded_config, "source_datasets": ("ptdpes",)}
    assert _load(path, resumed_config, "moflow").epoch == 7
    with pytest.raises(ValueError, match="configuration mismatch: source_datasets:"):
        _load(path, {**recorded_config, "source_datasets": ("ptdpes", "atc")}, "moflow")


def test_resume_guard_rejects_an_unknown_trainer():
    with pytest.raises(KeyError, match="Unknown trainer"):
        resume_guard_config({"stage": "pretrain"}, "no_such_trainer")


LIVE_CHECKPOINTS = [
    ("experiments/pretrained/socialvae_transfer_v2_ptdpes_0/checkpoints/best.pt", "social_vae", "model"),
    ("experiments/pretrained/trajectron_transfer_v2_ptdpes_1/checkpoints/best.pt", "trajectron", "models"),
]


@pytest.mark.parametrize("relative,trainer,model_key", LIVE_CHECKPOINTS)
def test_live_checkpoints_stay_resumable_under_the_inverted_guard(relative, trainer, model_key):
    """Regression floor: the inverted guard must not refuse an in-flight run's checkpoint."""
    path = Path(__file__).resolve().parents[1] / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not present in this checkout")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    recorded = payload["run_config"]
    # What the trainer would rebuild on resume: the recorded configuration minus the
    # fields RunStatus adds afterwards, with the operational differences a real resume
    # actually has.
    live = {key: value for key, value in recorded.items() if key not in RUNSTATUS_ADDED_FIELDS}
    live["resume"] = "auto"
    state = load_resume_state(
        path,
        stage=payload["stage"],
        optimization=payload["optimization"],
        expected_run_config=resume_guard_config(live, trainer),
        model_key=model_key,
    )
    assert state.unguarded_fields == ()
    assert state.epoch >= 1
    # and the fields L-017 named are genuinely guarded for this real configuration
    guarded = resume_guard_config(live, trainer)
    for field in ("validation_mode", "forecast_preprocessing_hash"):
        assert field in guarded
