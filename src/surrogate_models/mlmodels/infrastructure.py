"""Mlmodels infrastructure -- the imperative shell of the mlmodels context.

Provides the src implementation of the application's write port: save_trained_run
builds a run's model FROM its certified config and persists a checkpoint. It knows the
domain only enough to read a ConfiguredRun and hand back a Checkpoint -- it never
exposes a domain type outward.

This is the THIN training slice: the adapter persists the model's UNTRAINED weights
(no fit, no data, no Trainer yet) so the composition seat's happy path is reachable
with a real torch artifact on disk. The initial model is a minimal single-feature
linear regressor; the EXPAND slice widens it to the real feature count and trains it
over prepared data (replacing torch.save with a fitted Trainer checkpoint). The
context's multi-epoch training proof lives in tests, wiring the SAME application port
to a test-owned adapter that actually fits. Curried on ``checkpoint_dir`` at the
composition root, save_trained_run satisfies the application's SaveTrainedRunFn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightning
import torch
from torch import nn

from surrogate_models.mlmodels.domain import Checkpoint, ConfiguredRun, TrainingConfig
from surrogate_models.railway_adts import fmap_error, safe

logger = logging.getLogger(__name__)


class SurrogateRegressor(lightning.LightningModule):
    """A minimal single-feature linear regressor built FROM a run's config.

    The mlmodels context's initial surrogate model -- one ``1->1`` linear layer that
    remembers the run's optimiser choice and learning rate so a later fit trains under
    the certified config. The thin slice persists this model UNTRAINED, so it holds
    only what serialising its ``state_dict`` needs; the EXPAND slice widens it to the
    real feature count and adds the forward pass, training step, and optimiser under
    its own tests. Not the final neutron-star architecture -- the honest starting point.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.learning_rate = config.learning_rate
        self.optimizer_name = config.optimizer


@safe(Exception, fmap_error(lambda cause: cause, code="TRAINING_FAILED"))
def save_trained_run(checkpoint_dir: Path, run: ConfiguredRun) -> Checkpoint:
    """Build ``run``'s model and persist ``{run_id}.ckpt`` -- the src SaveTrainedRunFn.

    ``checkpoint_dir`` leads so the composition root binds it via ``partial`` to yield
    the application's ``SaveTrainedRunFn``. THIN slice: it builds a SurrogateRegressor
    from the run's certified config and writes its UNTRAINED weights
    (``torch.save(state_dict)``) to ``{run_id}.ckpt`` under the checkpoint dir, then
    returns the Checkpoint pointing at that path -- no fit, no data, no Trainer yet, so
    the seat's happy path is reachable with a real artifact on disk. ``@safe`` folds any
    I/O or serialisation failure into an ErrorInfo cause (``TRAINING_FAILED``). When the
    EXPAND slice trains for real, its fitted Trainer checkpoint replaces this body and
    every ring above stays unchanged.
    """
    logger.info("save_trained_run: run %s under %s", run.run_id, checkpoint_dir)
    model = SurrogateRegressor(run.config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"{run.run_id}.ckpt"
    torch.save(model.state_dict(), target)
    return Checkpoint(str(target))
