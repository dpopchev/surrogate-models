"""Mlmodels application -- CQRS handlers over the mlmodels domain.

Orchestrates training by composing injected callables (DIP): the model is GIVEN as
a BuildModelFn and training runs through a TrainFn, both supplied by infrastructure
at the composition root, so the application drives a run's lifecycle without ever
importing torch or doing I/O. The train command handler is added under TDD, kept as
ordered CQRS sections.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    InvalidRunID,
    InvalidTrainingConfig,
    RunID,
    TrainedRun,
    TrainingConfig,
    complete_training,
    configure_run,
    make_runid,
    make_training_config,
)
from surrogate_models.railway_adts import ErrorInfo, Result

logger = logging.getLogger(__name__)

# --- (1) Commands ---

# Opaque edge payloads the application ROUTES to the injected trainer but never
# inspects: the built model (a torch/Lightning object) and the handed-in training
# data (a tensor/frame). Typed as ``object`` so the application ring stays free of
# any torch import -- the concrete types live only in the infrastructure adapter.
TrainableModel: TypeAlias = object
TrainingData: TypeAlias = object

# Injected ports (DIP via callables), supplied by infrastructure at the composition
# root. BuildModelFn is HOW THE MODEL IS GIVEN -- the root closes over the concrete
# LightningModule factory and the application only calls it. TrainFn runs the
# training for a given RunID (which names the checkpoint file) and returns the
# produced Checkpoint, folding a training failure into an ErrorInfo cause (the @safe
# boundary lives in infrastructure, not here).
BuildModelFn = Callable[[], TrainableModel]
TrainFn = Callable[
    [RunID, TrainableModel, TrainingData, TrainingConfig],
    Result[Checkpoint, ErrorInfo],
]


@dataclass(frozen=True, slots=True, eq=False)
class TrainRun:
    """Command: configure and train a run on the handed-in ``data``.

    Carries only edge data -- a requested ``run_id`` (a bare ``str`` the handler
    certifies via the domain :func:`make_runid`), the training ``data`` (an opaque
    payload routed to the injected trainer), and the ``max_epochs`` budget (certified
    via :func:`make_training_config`). ``eq=False`` keeps identity equality so a
    command holding an arbitrary data payload (e.g. a tensor) never triggers an
    ambiguous elementwise ``==``. The model factory and the trainer are injected into
    the handler, not the command.
    """

    run_id: str
    data: TrainingData
    max_epochs: int


# The union of every mlmodels write command -- one member today, dispatched by
# match/case at the delivery edge. Grows as commands are added (no base class).
type MLModelCommand = TrainRun


@dataclass(frozen=True, slots=True)
class FailTrainRun:
    """Failure of handle_train_run, carrying its structured ``cause``.

    A single application error for every rail: an invalid id, an invalid config, and
    a training-boundary failure all fold into one ``ErrorInfo`` cause, so the failure
    crosses UP the ring as a serializable descriptor -- never the domain type or a
    lower-layer exception.
    """

    cause: ErrorInfo


def _fail_from_invalid_id(error: InvalidRunID) -> FailTrainRun:
    """Lift a domain id-validation rejection into the handler's own error.

    The offending id string stays in the domain error; the boundary reports a stable
    ``code`` plus message, never the domain type. Logged here at the boundary (the
    domain core stays free of logging).
    """
    logger.warning("run id rejected: %r is not a valid id", error.run_id)
    return FailTrainRun(
        ErrorInfo(code="INVALID_RUN_ID", message=f"invalid run id: {error.run_id!r}")
    )


def _fail_from_invalid_config(error: InvalidTrainingConfig) -> FailTrainRun:
    """Lift a domain config-validation rejection into the handler's own error.

    The offending epoch count stays in the domain error; the boundary reports a
    stable ``code`` plus message. Logged here at the boundary.
    """
    logger.warning(
        "training config rejected: max_epochs=%d must be >= 1", error.max_epochs
    )
    return FailTrainRun(
        ErrorInfo(
            code="INVALID_TRAINING_CONFIG",
            message=f"max_epochs must be >= 1: {error.max_epochs}",
        )
    )


def handle_train_run(
    *, build_model: BuildModelFn, train: TrainFn, cmd: TrainRun
) -> Result[TrainedRun, FailTrainRun]:
    """Configure a run from ``cmd``, train the GIVEN model, and record the checkpoint.

    Re-wraps the edge ``run_id`` and ``max_epochs`` into domain value objects via the
    smart constructors (an invalid id or epoch count short-circuits to
    ``FailTrainRun`` and neither the model nor the trainer is touched), builds the
    injected model, runs the injected ``train`` on it, and folds the produced
    Checkpoint into the terminal TrainedRun via the domain
    :func:`complete_training` transition. A training-boundary ``ErrorInfo`` folds into
    ``FailTrainRun``. Pure railway composition -- no branching on the ``Result``.
    """
    return (
        make_runid(cmd.run_id)
        .fmap_err(_fail_from_invalid_id)
        .and_then(
            lambda run_id: (
                make_training_config(cmd.max_epochs)
                .fmap_err(_fail_from_invalid_config)
                .fmap(lambda config: configure_run(run_id, config))
            )
        )
        .and_then(
            lambda run: (
                train(run.run_id, build_model(), cmd.data, run.config)
                .fmap_err(FailTrainRun)
                .fmap(lambda checkpoint: complete_training(run, checkpoint))
            )
        )
    )
