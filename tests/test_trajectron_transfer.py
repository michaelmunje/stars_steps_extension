"""Unit tests for the Trajectron++ finetune checkpoint guard and seed policy.

These exercise ``validate_trajectron_finetune_checkpoint`` and
``resolve_trajectron_seed`` directly with synthetic in-memory payloads: no
real checkpoints, datasets, or GPU are required.
"""
import unittest
from pathlib import Path

from types import SimpleNamespace
from unittest import mock

from baselines import train_trajectron
from baselines.train_trajectron import (
    UPSTREAM_COMMIT,
    _build_environments,
    resolve_trajectron_seed,
    validate_trajectron_finetune_checkpoint,
)

EXPECTED_HASH = "deadbeefcafef00d"


def _valid_payload(**overrides):
    payload = {
        "stage": "pretrain",
        "upstream_commit": UPSTREAM_COMMIT,
        "run_config": {"forecast_preprocessing_hash": EXPECTED_HASH, "seed": 7},
        "checkpoint_selection": {"metric": "val_minADE", "value": 1.23, "best_epoch": 5},
    }
    payload.update(overrides)
    return payload


class ValidateTrajectronFinetuneCheckpointTests(unittest.TestCase):
    def _check(self, payload, checkpoint_name="best.pt"):
        validate_trajectron_finetune_checkpoint(
            payload, Path(checkpoint_name),
            expected_hash=EXPECTED_HASH, expected_commit=UPSTREAM_COMMIT,
        )

    def test_valid_payload_passes(self):
        self._check(_valid_payload())  # must not raise

    def test_rejects_non_best_checkpoint_name(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(_valid_payload(), checkpoint_name="latest.pt")
        self.assertIn("best.pt", str(ctx.exception))
        self.assertIn("latest.pt", str(ctx.exception))

    def test_rejects_non_pretrain_stage(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(_valid_payload(stage="finetune"))
        self.assertIn("pretrain", str(ctx.exception))

    def test_rejects_forecast_preprocessing_hash_mismatch(self):
        payload = _valid_payload(run_config={"forecast_preprocessing_hash": "wronghash", "seed": 7})
        with self.assertRaises(ValueError) as ctx:
            self._check(payload)
        message = str(ctx.exception)
        self.assertIn("wronghash", message)
        self.assertIn(EXPECTED_HASH, message)

    def test_rejects_upstream_commit_mismatch(self):
        payload = _valid_payload(upstream_commit="0000000000000000000000000000000000000")
        with self.assertRaises(ValueError) as ctx:
            self._check(payload)
        message = str(ctx.exception)
        self.assertIn("0000000000000000000000000000000000000", message)
        self.assertIn(UPSTREAM_COMMIT, message)

    def test_rejects_missing_checkpoint_selection_value(self):
        payload = _valid_payload(checkpoint_selection={"metric": "val_minADE", "value": None, "best_epoch": None})
        with self.assertRaises(ValueError) as ctx:
            self._check(payload)
        self.assertIn("validation metric", str(ctx.exception))


class ResolveTrajectronSeedTests(unittest.TestCase):
    def test_finetune_with_no_explicit_seed_inherits_pretraining_seed(self):
        payload = {"run_config": {"seed": 7}}
        self.assertEqual(resolve_trajectron_seed(None, "finetune", payload), 7)

    def test_explicit_seed_always_wins(self):
        payload = {"run_config": {"seed": 7}}
        self.assertEqual(resolve_trajectron_seed(3, "finetune", payload), 3)
        self.assertEqual(resolve_trajectron_seed(3, "pretrain", None), 3)

    def test_scratch_with_no_explicit_seed_defaults_to_zero(self):
        self.assertEqual(resolve_trajectron_seed(None, "scratch", None), 0)

    def test_pretrain_with_no_explicit_seed_defaults_to_zero(self):
        self.assertEqual(resolve_trajectron_seed(None, "pretrain", None), 0)

    def test_finetune_without_checkpoint_payload_defaults_to_zero(self):
        self.assertEqual(resolve_trajectron_seed(None, "finetune", None), 0)


class BuildEnvironmentsValidationPopulationTests(unittest.TestCase):
    """N-016: a target stage (scratch/finetune) with no validation population
    must hard-error rather than silently training with best.pt = last epoch.
    """

    def _fake_env(self, n_scenes):
        return SimpleNamespace(scenes=[SimpleNamespace() for _ in range(n_scenes)])

    def test_finetune_with_empty_validation_population_raises(self):
        args = SimpleNamespace(stage="finetune", fold="eth", validation_mode="grouped",
                                max_recordings=None, data_seed=42)
        with mock.patch.object(train_trajectron, "fold_environment",
                                side_effect=[self._fake_env(3), self._fake_env(0)]):
            with self.assertRaises(RuntimeError) as ctx:
                _build_environments(args, {}, run=None)
        self.assertIn("No validation population", str(ctx.exception))

    def test_finetune_with_none_validation_environment_raises(self):
        args = SimpleNamespace(stage="finetune", fold="eth", validation_mode="grouped",
                                max_recordings=None, data_seed=42)
        with mock.patch.object(train_trajectron, "fold_environment",
                                side_effect=[self._fake_env(3), None]):
            with self.assertRaises(RuntimeError) as ctx:
                _build_environments(args, {}, run=None)
        self.assertIn("No validation population", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
