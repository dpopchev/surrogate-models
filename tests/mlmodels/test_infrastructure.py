"""Tests for the mlmodels infrastructure -- the Lightning-backed TrainFn adapter.

build_stub_model builds the model FROM the certified config, so its optimiser (Adam
or SGD) and learning rate match the run. train_with_lightning wraps the handed-in raw
dataset in a DataLoader at the run's batch_size and runs a real one-epoch CPU Trainer
over the GIVEN model, writing a checkpoint named by the run id under the checkpoint
dir. Integration tests exercise real Lightning: success folds into Ok(Checkpoint), any
failure onto the ErrorInfo rail. One assert per test.
"""

from pathlib import Path
from typing import cast

import lightning
import torch

from surrogate_models.mlmodels.domain import Checkpoint, RunID, TrainingConfig
from surrogate_models.mlmodels.infrastructure import (
    build_stub_model,
    stub_training_data,
    train_with_lightning,
)


def _config(
    max_epochs: int = 1,
    learning_rate: float = 0.01,
    batch_size: int = 2,
    optimizer: str = "sgd",
) -> TrainingConfig:
    """A valid TrainingConfig, built directly for the adapter tests."""
    return TrainingConfig(max_epochs, learning_rate, batch_size, optimizer)


# --- build_stub_model: the optimiser is configured FROM the certified config ---


def test_build_stub_model_uses_sgd_by_default() -> None:
    model = cast(lightning.LightningModule, build_stub_model(_config(optimizer="sgd")))
    assert isinstance(model.configure_optimizers(), torch.optim.SGD)


def test_build_stub_model_uses_adam_when_configured() -> None:
    model = cast(lightning.LightningModule, build_stub_model(_config(optimizer="adam")))
    assert isinstance(model.configure_optimizers(), torch.optim.Adam)


def test_build_stub_model_sets_the_learning_rate() -> None:
    built = build_stub_model(_config(learning_rate=0.5))
    model = cast(lightning.LightningModule, built)
    optimizer = cast(torch.optim.Optimizer, model.configure_optimizers())
    assert optimizer.param_groups[0]["lr"] == 0.5


# --- train_with_lightning: real one-epoch Trainer over the shell-built loader ---


def test_train_with_lightning_is_ok(tmp_path: Path) -> None:
    result = train_with_lightning(
        tmp_path,
        RunID("smoke"),
        build_stub_model(_config()),
        stub_training_data(),
        _config(),
    )
    assert result.is_ok() is True


def test_train_with_lightning_writes_the_checkpoint_file(tmp_path: Path) -> None:
    train_with_lightning(
        tmp_path,
        RunID("smoke"),
        build_stub_model(_config()),
        stub_training_data(),
        _config(),
    )
    assert (tmp_path / "smoke.ckpt").exists()


def test_train_with_lightning_returns_the_checkpoint_location(tmp_path: Path) -> None:
    result = train_with_lightning(
        tmp_path,
        RunID("smoke"),
        build_stub_model(_config()),
        stub_training_data(),
        _config(),
    )
    assert result.unwrap() == Checkpoint(str(tmp_path / "smoke.ckpt"))


def test_train_with_lightning_runs_with_adam_and_full_batch(tmp_path: Path) -> None:
    config = _config(optimizer="adam", batch_size=4)
    result = train_with_lightning(
        tmp_path,
        RunID("smoke"),
        build_stub_model(config),
        stub_training_data(),
        config,
    )
    assert result.is_ok() is True
