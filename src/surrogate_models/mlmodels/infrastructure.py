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
    """A trivial 1->1 linear regressor -- the stub model the smoke GIVES the trainer.

    Deliberately minimal: it exists only to prove the training slice end to end (one
    epoch writes a checkpoint), NOT to model neutron stars. The real 28-feature
    surrogate regressor is a later slice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the target from a batch of single-feature inputs."""
        prediction: torch.Tensor = self.linear(features)
        return prediction

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run one optimisation step: MSE between the prediction and the target.

        Signature mirrors LightningModule's ``*args`` contract so the override stays
        type-compatible; the batch is a ``(features, targets)`` pair from the stub
        DataLoader.
        """
        features, targets = args[0]
        return nn.functional.mse_loss(self.linear(features), targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use plain SGD -- enough to take a step in the smoke."""
        return torch.optim.SGD(self.parameters(), lr=0.01)


def build_stub_model() -> object:
    """Build the stub regressor -- the BuildModelFn the composition root GIVES.

    Returns a fresh ``_StubRegressor`` as an opaque object; the application never
    inspects it, only routes it to :func:`train_with_lightning`.
    """
    return _StubRegressor()


def stub_training_data() -> object:
    """Build a tiny 4-row DataLoader -- stand-in training data for the smoke.

    A deterministic single-feature -> single-target set (target = 2*x + 1), enough for
    one training epoch. Opaque to the application, which hands it straight through to
    the trainer.
    """
    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    return DataLoader(TensorDataset(features, targets), batch_size=2)


@safe(Exception, fmap_error(lambda cause: cause, code="TRAINING_FAILED"))
def train_with_lightning(
    checkpoint_dir: Path,
    run_id: RunID,
    model: object,
    data: object,
    config: TrainingConfig,
) -> Checkpoint:
    """Train ``model`` for ``config.max_epochs`` and write ``{run_id}.ckpt``.

    The TrainFn adapter. ``checkpoint_dir`` leads so the composition root binds it
    with ``partial(train_with_lightning, settings.mlmodels.checkpoint_dir)`` to yield
    the application's ``TrainFn``. Runs a CPU Lightning Trainer over the GIVEN model
    and data (logging, progress bar, and model summary silenced, its own checkpointing
    disabled), then saves the checkpoint explicitly under ``checkpoint_dir`` named by
    the run id, and returns the Checkpoint pointing at it.

    ``@safe`` makes this the railway boundary: on success ``Ok(Checkpoint)``; ANY
    exception (a bad model/data shape, an optimiser error, a disk failure) is caught
    and folded into ``Err(ErrorInfo(code="TRAINING_FAILED"))`` -- the structured cause
    crosses UP, never the raw exception. The catch is broad (``Exception``) because a
    training failure is a recoverable run outcome, not a boot fault.
    """
    logger.info("training run %s for %d epoch(s)", run_id, config.max_epochs)
    trainer = lightning.Trainer(
        max_epochs=config.max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(cast(lightning.LightningModule, model), cast(Any, data))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"{run_id}.ckpt"
    trainer.save_checkpoint(target)
    logger.info("wrote checkpoint %s", target)
    return Checkpoint(str(target))
