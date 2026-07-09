"""End-to-end proof of the mlmodels application port with test-owned infrastructure.

The src training adapter is a not-yet-implemented placeholder (the real regressor is a
later slice), so this proof brings its OWN infrastructure: StubRegressor is a trivial
1->1 linear model built FROM a run's certified config, and stub_save_trained_run fuses
it with a tiny deterministic dataset into a real one-epoch CPU Lightning run that
writes {run_id}.ckpt -- satisfying the application's SaveTrainedRunFn when curried on
the checkpoint dir. The tests then drive the SAME production wiring the real adapter
will use: the application handler directly (Ok TrainedRun carrying the checkpoint) and
the __main__ composition seat (the checkpoint the run actually wrote to disk).

A second, fuller proof shows real WORK: stub_full_training_run runs a MULTI-epoch CPU
run, writing one checkpoint per epoch (a "step") into ``{checkpoint_dir}/steps`` and
recording the per-epoch training loss via _LossRecorder, so the tests can assert both
that the steps land on disk and that the loss actually falls -- the model is learning,
not merely producing a file. All of this is test scaffolding, never shipped in src.
One assert per test.
"""

from functools import partial
from pathlib import Path
from typing import Any

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from surrogate_models.mlmodels.__main__ import train_run
from surrogate_models.mlmodels.application import TrainRun, handle_train_run
from surrogate_models.mlmodels.domain import Checkpoint, ConfiguredRun, TrainedRun
from surrogate_models.railway_adts import fmap_error, safe


class StubRegressor(lightning.LightningModule):
    """A trivial 1->1 linear regressor built FROM a run's config -- test model only.

    Exists only to prove the training slice end to end; it configures its optimiser
    (Adam or plain SGD) at the run's learning rate. Not a neutron-star model.
    """

    def __init__(self, learning_rate: float, optimizer_name: str) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the target from a batch of single-feature inputs."""
        prediction: torch.Tensor = self.linear(features)
        return prediction

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run one MSE optimisation step over a ``(features, targets)`` batch.

        Keeps the ``*args`` supertype signature so the override stays Liskov-compatible
        under mypy strict; the batch is the first positional argument.
        """
        features, targets = args[0]
        loss: torch.Tensor = nn.functional.mse_loss(self.linear(features), targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the run's optimiser at its learning rate -- Adam or plain SGD."""
        match self.optimizer_name:
            case "adam":
                return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            case _:
                return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


class _LossRecorder(lightning.Callback):
    """Records the mean training loss at the end of each epoch -- the work witness.

    Reads the epoch-aggregated ``train_loss`` the model logs and appends it, so a test
    can compare the first and last epoch and prove the loss actually fell. Keeps the
    supertype hook signature (``trainer``, ``pl_module``) for Liskov under mypy strict.
    """

    def __init__(self) -> None:
        super().__init__()
        self.epoch_losses: list[float] = []

    def on_train_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Append this epoch's aggregated training loss."""
        self.epoch_losses.append(float(trainer.callback_metrics["train_loss"]))


def _stub_training_data() -> TensorDataset:
    """A deterministic 4-row single-feature dataset (target = 2*x + 1)."""
    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    return TensorDataset(features, targets)


@safe(Exception, fmap_error(lambda cause: cause, code="TRAINING_FAILED"))
def stub_save_trained_run(checkpoint_dir: Path, run: ConfiguredRun) -> Checkpoint:
    """Train a stub model for ``run`` and persist ``{run_id}.ckpt`` -- the test adapter.

    Builds a StubRegressor from the run's certified config, wraps the stub dataset in a
    DataLoader at the run's batch_size, runs a one-epoch quiet CPU Trainer, saves the
    checkpoint under ``checkpoint_dir`` named by the run id, and returns the Checkpoint.
    ``@safe`` folds any failure into an ErrorInfo cause. Curried on ``checkpoint_dir``
    it satisfies the application's SaveTrainedRunFn.
    """
    model = StubRegressor(run.config.learning_rate, run.config.optimizer)
    loader = DataLoader(_stub_training_data(), batch_size=run.config.batch_size)
    trainer = lightning.Trainer(
        max_epochs=run.config.max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"{run.run_id}.ckpt"
    trainer.save_checkpoint(target)
    return Checkpoint(str(target))


@safe(Exception, fmap_error(lambda cause: cause, code="TRAINING_FAILED"))
def stub_full_training_run(
    recorder: _LossRecorder, checkpoint_dir: Path, run: ConfiguredRun
) -> Checkpoint:
    """Train ``run`` across epochs, saving one step checkpoint per epoch.

    The fuller test adapter. Like stub_save_trained_run but proves real WORK end to end:
    a ModelCheckpoint saves
    the model into ``{checkpoint_dir}/steps`` at every epoch (the "steps" that land on
    disk), and ``recorder`` captures the per-epoch loss so a test can assert it fell.
    The final ``{run_id}.ckpt`` is still returned as the run's Checkpoint. Seeded for a
    deterministic descent. Curried on ``recorder`` + ``checkpoint_dir`` it satisfies the
    application's SaveTrainedRunFn.
    """
    torch.manual_seed(0)
    model = StubRegressor(run.config.learning_rate, run.config.optimizer)
    loader = DataLoader(_stub_training_data(), batch_size=run.config.batch_size)
    steps = ModelCheckpoint(
        dirpath=str(checkpoint_dir / "steps"),
        filename="epoch-{epoch}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    trainer = lightning.Trainer(
        max_epochs=run.config.max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[steps, recorder],
    )
    trainer.fit(model, loader)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"{run.run_id}.ckpt"
    trainer.save_checkpoint(target)
    return Checkpoint(str(target))


def _cmd(run_id: str = "e2e") -> TrainRun:
    """A valid TrainRun for the end-to-end path."""
    return TrainRun(
        run_id, max_epochs=1, learning_rate=0.01, batch_size=2, optimizer="sgd"
    )


_FULL_EPOCHS = 10


def _full_cmd(run_id: str = "full") -> TrainRun:
    """A valid TrainRun for the fuller multi-epoch path (Adam descends fast)."""
    return TrainRun(
        run_id,
        max_epochs=_FULL_EPOCHS,
        learning_rate=0.05,
        batch_size=2,
        optimizer="adam",
    )


# --- application API end to end: handler + test-owned adapter ---


def test_handler_returns_a_trained_run(tmp_path: Path) -> None:
    save = partial(stub_save_trained_run, tmp_path)
    result = handle_train_run(save_trained_run=save, cmd=_cmd())
    assert isinstance(result.unwrap(), TrainedRun)


def test_handler_trained_run_points_at_the_written_checkpoint(tmp_path: Path) -> None:
    save = partial(stub_save_trained_run, tmp_path)
    result = handle_train_run(save_trained_run=save, cmd=_cmd())
    assert result.unwrap().checkpoint == Checkpoint(str(tmp_path / "e2e.ckpt"))


# --- composition seat end to end: __main__.train_run + injected adapter ---


def test_seat_returns_the_checkpoint_location(tmp_path: Path) -> None:
    location = train_run(
        save_trained_run=partial(stub_save_trained_run, tmp_path), cmd=_cmd()
    )
    assert location == str(tmp_path / "e2e.ckpt")


def test_seat_writes_the_checkpoint_file(tmp_path: Path) -> None:
    train_run(save_trained_run=partial(stub_save_trained_run, tmp_path), cmd=_cmd())
    assert (tmp_path / "e2e.ckpt").exists()


# --- fuller proof: a multi-epoch run does real work, its steps land on disk ---


def test_full_run_completes_with_a_trained_run(tmp_path: Path) -> None:
    save = partial(stub_full_training_run, _LossRecorder(), tmp_path)
    result = handle_train_run(save_trained_run=save, cmd=_full_cmd())
    assert isinstance(result.unwrap(), TrainedRun)


def test_full_run_writes_a_step_checkpoint_per_epoch(tmp_path: Path) -> None:
    save = partial(stub_full_training_run, _LossRecorder(), tmp_path)
    handle_train_run(save_trained_run=save, cmd=_full_cmd())
    assert len(list((tmp_path / "steps").glob("*.ckpt"))) == _FULL_EPOCHS


def test_full_run_reduces_the_training_loss(tmp_path: Path) -> None:
    recorder = _LossRecorder()
    save = partial(stub_full_training_run, recorder, tmp_path)
    handle_train_run(save_trained_run=save, cmd=_full_cmd())
    assert recorder.epoch_losses[-1] < recorder.epoch_losses[0]
