"""Tests for the mlmodels src infrastructure -- the save_trained_run adapter.

The thin training slice: save_trained_run builds a minimal real SurrogateRegressor
FROM a run's certified config and persists its UNTRAINED weights as ``{run_id}.ckpt``
under the checkpoint dir, returning ``Ok(Checkpoint)`` -- so the composition seat's
happy path is reachable with a real torch artifact on disk. No training happens yet
(no fit, no data); the multi-epoch training proof stays in test_e2e against a
test-owned adapter, and the EXPAND slice grows this adapter to actually fit over real
data. One assert per test.
"""

from pathlib import Path

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
    RunID,
    TrainingConfig,
    configure_run,
)
from surrogate_models.mlmodels.infrastructure import save_trained_run


def _configured_run() -> ConfiguredRun:
    """A certified ConfiguredRun, built directly for the adapter tests."""
    return configure_run(RunID("smoke"), TrainingConfig(1, 0.01, 2, "sgd"))


def test_save_trained_run_writes_the_named_checkpoint(tmp_path: Path) -> None:
    save_trained_run(tmp_path, _configured_run())
    assert (tmp_path / "smoke.ckpt").exists() is True


def test_save_trained_run_returns_the_written_checkpoint(tmp_path: Path) -> None:
    result = save_trained_run(tmp_path, _configured_run())
    assert result.unwrap() == Checkpoint(str(tmp_path / "smoke.ckpt"))
