"""Tests for the mlmodels infrastructure -- the Lightning-backed TrainFn adapter.

train_with_lightning runs a real one-epoch CPU Trainer over the GIVEN stub model and
tiny stub data, writing a checkpoint named by the run id under the checkpoint dir.
This is an integration test that exercises real Lightning: success folds into
Ok(Checkpoint), any failure onto the ErrorInfo rail. One assert per test.
"""

from pathlib import Path

from surrogate_models.mlmodels.domain import Checkpoint, RunID, TrainingConfig
from surrogate_models.mlmodels.infrastructure import (
    build_stub_model,
    stub_training_data,
    train_with_lightning,
)


def test_train_with_lightning_is_ok(tmp_path: Path) -> None:
    result = train_with_lightning(
        tmp_path,
        RunID("smoke"),
        build_stub_model(),
        stub_training_data(),
        TrainingConfig(1),
    )
    assert result.is_ok() is True


def test_train_with_lightning_writes_the_checkpoint_file(tmp_path: Path) -> None:
    train_with_lightning(
        tmp_path,
        RunID("smoke"),
        build_stub_model(),
        stub_training_data(),
        TrainingConfig(1),
    )
    assert (tmp_path / "smoke.ckpt").exists()


def test_train_with_lightning_returns_the_checkpoint_location(tmp_path: Path) -> None:
    result = train_with_lightning(
        tmp_path,
        RunID("smoke"),
        build_stub_model(),
        stub_training_data(),
        TrainingConfig(1),
    )
    assert result.unwrap() == Checkpoint(str(tmp_path / "smoke.ckpt"))
