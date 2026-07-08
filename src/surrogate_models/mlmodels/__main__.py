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


def train_stub_run() -> str:
    """Train the stub model one epoch and return the checkpoint location.

    Bind the Lightning adapter to the handler (currying the checkpoint dir from
    settings), GIVE the stub model via build_stub_model, and run one epoch on stub
    data. ``unwrap`` raises at this outer edge if training fails -- carrying the
    ErrorInfo cause.
    """
    settings = get_settings()
    logger.info("composing stub run under %s", settings.mlmodels.checkpoint_dir)
    train = partial(train_with_lightning, settings.mlmodels.checkpoint_dir)
    cmd = TrainRun(run_id=_STUB_RUN_ID, data=stub_training_data(), max_epochs=1)
    return (
        handle_train_run(build_model=build_stub_model, train=train, cmd=cmd)
        .unwrap()
        .checkpoint.location
    )
