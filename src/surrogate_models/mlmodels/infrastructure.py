"""Mlmodels infrastructure -- the imperative shell of the mlmodels context.

Supplies the impure training the pure core must not touch: the Lightning-backed
TrainFn adapter that runs a Trainer over the GIVEN model and writes the checkpoint,
folding any failure onto the railway as an ErrorInfo. A stub LightningModule and a
tiny stub dataset back the thin training slice; the real 28-feature surrogate
regressor and real data are a later slice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import lightning
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from surrogate_models.mlmodels.domain import Checkpoint, RunID, TrainingConfig
from surrogate_models.railway_adts import fmap_error, safe

logger = logging.getLogger(__name__)


class _StubRegressor(lightning.LightningModule):
    """A trivial 1->1 linear regressor -- the stub model built FROM the run's config.

    Deliberately minimal: it exists only to prove the training slice end to end (an
    epoch writes a checkpoint), NOT to model neutron stars. It configures its optimiser
    from the certified ``learning_rate`` + ``optimizer_name`` the composition root
    hands it, and logs its per-step loss to the progress bar so a notebook run shows
    live movement. The real 28-feature surrogate regressor is a later slice.
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
        """Run one optimisation step: MSE between the prediction and the target.

        Signature mirrors LightningModule's ``*args`` contract so the override stays
        type-compatible; the batch is a ``(features, targets)`` pair from the
        DataLoader. Logs ``train_loss`` to the progress bar so live loss shows during
        training (e.g. in a notebook).
        """
        features, targets = args[0]
        loss: torch.Tensor = nn.functional.mse_loss(self.linear(features), targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the run's optimiser at its learning rate -- Adam or plain SGD.

        The name is already certified to SUPPORTED_OPTIMIZERS by the domain, so the
        default arm handles ``sgd``; only ``adam`` needs its own case.
        """
        match self.optimizer_name:
            case "adam":
                return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            case _:
                return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


def build_stub_model(config: TrainingConfig) -> object:
    """Build the stub regressor FROM the certified config -- the BuildModelFn GIVEN.

    Reads ``learning_rate`` + ``optimizer`` off the certified config so the model's
    optimiser matches the run, and returns it as an opaque object; the application
    never inspects it, only routes it to :func:`train_with_lightning`.
    """
    return _StubRegressor(config.learning_rate, config.optimizer)


def stub_training_data() -> object:
    """Build a tiny 4-row TensorDataset -- stand-in training data for the smoke.

    A deterministic single-feature -> single-target set (target = 2*x + 1). Handed in
    RAW (a Dataset, not a DataLoader) so the shell wraps it in a DataLoader at the
    run's ``batch_size`` -- batching is thus a certified-config knob. Opaque to the
    application, which hands it straight through to the trainer.
    """
    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    return TensorDataset(features, targets)


@safe(Exception, fmap_error(lambda cause: cause, code="TRAINING_FAILED"))
def train_with_lightning(
    checkpoint_dir: Path,
    run_id: RunID,
    model: object,
    data: object,
    config: TrainingConfig,
    *,
    enable_progress: bool = False,
) -> Checkpoint:
    """Train ``model`` for ``config.max_epochs`` and write ``{run_id}.ckpt``.

    The TrainFn adapter. ``checkpoint_dir`` leads so the composition root binds it (and
    ``enable_progress``) via ``partial`` to yield the application's ``TrainFn``. Wraps
    the handed-in raw dataset in a ``DataLoader`` at ``config.batch_size``, runs a CPU
    Lightning Trainer over the GIVEN model (its own metric logging and checkpointing
    disabled), then saves the checkpoint explicitly under ``checkpoint_dir`` named by
    the run id, and returns the Checkpoint pointing at it. ``enable_progress`` turns on
    Lightning's live progress bar + model summary (bound True at the root for
    interactive/notebook runs; left False so the test suite stays quiet).

    ``@safe`` makes this the railway boundary: on success ``Ok(Checkpoint)``; ANY
    exception (a bad model/data shape, an optimiser error, a disk failure) is caught
    and folded into ``Err(ErrorInfo(code="TRAINING_FAILED"))`` -- the structured cause
    crosses UP, never the raw exception. The catch is broad (``Exception``) because a
    training failure is a recoverable run outcome, not a boot fault.
    """
    logger.info("training run %s for %d epoch(s)", run_id, config.max_epochs)
    loader = DataLoader(cast(TensorDataset, data), batch_size=config.batch_size)
    trainer = lightning.Trainer(
        max_epochs=config.max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=enable_progress,
        enable_model_summary=enable_progress,
    )
    trainer.fit(cast(lightning.LightningModule, model), loader)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"{run_id}.ckpt"
    trainer.save_checkpoint(target)
    logger.info("wrote checkpoint %s", target)
    return Checkpoint(str(target))
