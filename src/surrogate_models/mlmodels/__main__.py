"""Mlmodels context root -- the composition seat for the training slice.

Exposes train_run: the context's public entry point that wires an INJECTED
SaveTrainedRunFn adapter into the application's train handler and projects the result
to the checkpoint location -- a bare ``str``, so the seat never names a domain type
(the composition seal). The ready-to-use adapter is supplied by whoever composes the
context: the src ``infrastructure.save_trained_run`` (curried on a checkpoint dir)
once the real training slice lands, or a test-owned adapter for the end-to-end proof.
The higher build-up (settings wiring, a top-level project main, a delivery edge) is
added as the system grows.
"""

from __future__ import annotations

import logging

from surrogate_models.mlmodels.application import (
    SaveTrainedRunFn,
    TrainRun,
    handle_train_run,
)

logger = logging.getLogger(__name__)


def train_run(*, save_trained_run: SaveTrainedRunFn, cmd: TrainRun) -> str:
    """Train a run via the injected adapter and return its checkpoint location.

    The context's composition seat: hands ``cmd`` to the application handler with the
    GIVEN ``save_trained_run`` and unwraps the terminal run's checkpoint location (a
    bare ``str`` -- the seat projects away the domain aggregate so it never imports a
    domain type). ``unwrap`` raises at this outer edge if configuration or training
    fails, carrying the ErrorInfo cause. The adapter is injected by the composer -- the
    src infrastructure adapter once the real training slice lands, or a test adapter.
    """
    logger.info("composing train run %s", cmd.run_id)
    return (
        handle_train_run(save_trained_run=save_trained_run, cmd=cmd)
        .unwrap()
        .checkpoint.location
    )
