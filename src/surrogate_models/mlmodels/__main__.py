"""Mlmodels context root -- composition of the training slice.

The context's composition seat and smoke facade: it reads settings via
get_settings(), binds the Lightning infrastructure adapter to this context's OWN
application handler (DIP, currying the checkpoint dir), GIVES the stub model, and runs
one epoch on stub data -- never touching the domain ring directly. Returns the
checkpoint location the trained run wrote.
"""

from __future__ import annotations

import logging
from functools import partial

from surrogate_models.config import get_settings
from surrogate_models.mlmodels.application import TrainRun, handle_train_run
from surrogate_models.mlmodels.infrastructure import (
    build_stub_model,
    stub_training_data,
    train_with_lightning,
)

logger = logging.getLogger(__name__)

_STUB_RUN_ID = "stub-run"


def train_stub_run(
    max_epochs: int = 5,
    learning_rate: float = 0.01,
    batch_size: int = 2,
    optimizer: str = "sgd",
) -> str:
    """Train the stub model and return the checkpoint location, showing live progress.

    The notebook-facing facade. Binds the Lightning adapter to the handler -- currying
    the checkpoint dir from settings AND ``enable_progress=True`` so an interactive run
    shows a live progress bar with per-step loss -- GIVES the stub model built from the
    certified config, and runs the requested epochs on stub data. Every hyperparameter
    is a keyword with a sensible default so a notebook caller can tweak
    epochs/lr/batch/optimizer and watch. ``unwrap`` raises at this outer edge if
    training fails -- carrying the ErrorInfo cause (e.g. an unsupported optimizer name).
    """
    settings = get_settings()
    logger.info("composing stub run under %s", settings.mlmodels.checkpoint_dir)
    train = partial(
        train_with_lightning, settings.mlmodels.checkpoint_dir, enable_progress=True
    )
    cmd = TrainRun(
        run_id=_STUB_RUN_ID,
        data=stub_training_data(),
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer=optimizer,
    )
    return (
        handle_train_run(build_model=build_stub_model, train=train, cmd=cmd)
        .unwrap()
        .checkpoint.location
    )
