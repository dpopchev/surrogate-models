"""Mlmodels application -- CQRS handlers over the mlmodels domain.

Orchestrates training by composing an injected callable (DIP): training a run is
modeled as SAVING its aggregate through a SaveTrainedRunFn, supplied by
infrastructure at the composition root, so the application drives a run's lifecycle
without ever importing torch or doing I/O. The train command handler is kept as
ordered CQRS sections.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
    InvalidBatchSize,
    InvalidLearningRate,
    InvalidMaxEpochs,
    InvalidOptimizer,
    InvalidRunID,
    RunID,
    RunSummaryDTO,
    TrainedRun,
    TrainingConfigError,
    complete_training,
    configure_run,
    make_runid,
    make_training_config,
)
from surrogate_models.railway_adts import ErrorInfo, Result

logger = logging.getLogger(__name__)

# --- (1) Commands ---

# Injected write port (DIP via callables), supplied by infrastructure at the
# composition root. Training a run is modeled as SAVING its aggregate: the shell is
# GIVEN the certified ConfiguredRun, builds the concrete model FROM the aggregate's
# config internally (the application never sees a torch object), runs the training,
# writes the checkpoint named by the run id, and returns the produced Checkpoint --
# folding any training failure into an ErrorInfo cause (the @safe boundary lives in
# infrastructure, not here). Mirrors the datasets SaveDatasetFn shape (the aggregate
# in, a Result out); it yields the Checkpoint the save produced rather than None. The
# concrete model + data live only in the infrastructure adapter -- no torch type
# crosses into this ring, and no model/data payload rides on the command.
SaveTrainedRunFn = Callable[[ConfiguredRun], Result[Checkpoint, ErrorInfo]]


@dataclass(frozen=True, slots=True)
class TrainRun:
    """Command: configure and train a run under the given hyperparameters.

    Carries only edge primitives -- a requested ``run_id`` (a bare ``str`` the handler
    certifies via the domain :func:`make_runid`) and the four raw training knobs
    (``max_epochs``, ``learning_rate``, ``batch_size``, ``optimizer``) the handler
    certifies together via :func:`make_training_config`. No model or data object rides
    on the command: the shell builds the model FROM the certified config and sources
    its own data, so the command stays pure primitives. The trainer
    (:data:`SaveTrainedRunFn`) is injected into the handler, not the command.
    """

    run_id: str
    max_epochs: int
    learning_rate: float
    batch_size: int
    optimizer: str


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


def _config_failure(code: str, message: str) -> FailTrainRun:
    """Log a rejected training knob at the boundary and fold it into FailTrainRun."""
    logger.warning("training config rejected: %s", message)
    return FailTrainRun(ErrorInfo(code=code, message=message))


def _fail_from_invalid_config(error: TrainingConfigError) -> FailTrainRun:
    """Lift a domain config-validation rejection into the handler's own error.

    Exhaustively matches the per-field rejection VALUE (never isinstance) and reports
    a stable ``code`` plus message for the offending knob. The offending value stays
    in the domain error; only the descriptor crosses up the ring.
    """
    match error:
        case InvalidMaxEpochs(max_epochs=value):
            return _config_failure(
                "INVALID_MAX_EPOCHS", f"max_epochs must be >= 1: {value}"
            )
        case InvalidLearningRate(learning_rate=value):
            return _config_failure(
                "INVALID_LEARNING_RATE", f"learning_rate must be > 0: {value}"
            )
        case InvalidBatchSize(batch_size=value):
            return _config_failure(
                "INVALID_BATCH_SIZE", f"batch_size must be >= 1: {value}"
            )
        case InvalidOptimizer(optimizer=value):
            return _config_failure(
                "INVALID_OPTIMIZER", f"unsupported optimizer: {value!r}"
            )
        case _:
            assert_never(error)


def handle_train_run(
    *, save_trained_run: SaveTrainedRunFn, cmd: TrainRun
) -> Result[TrainedRun, FailTrainRun]:
    """Configure a run from ``cmd``, train + persist it, and record the checkpoint.

    Re-wraps the edge ``run_id`` and the four training knobs into domain value objects
    via the smart constructors (an invalid id or any bad knob short-circuits to
    ``FailTrainRun`` and the trainer is never touched), forms the ConfiguredRun
    aggregate, and hands it to the injected ``save_trained_run`` -- which builds the
    run's model from its certified config, trains it, and persists the checkpoint. The
    produced Checkpoint folds into the terminal TrainedRun via the domain
    :func:`complete_training` transition; a training-boundary ``ErrorInfo`` folds into
    ``FailTrainRun``. Pure railway composition -- no branching on the ``Result``.
    """
    return (
        make_runid(cmd.run_id)
        .fmap_err(_fail_from_invalid_id)
        .and_then(
            lambda run_id: (
                make_training_config(
                    cmd.max_epochs, cmd.learning_rate, cmd.batch_size, cmd.optimizer
                )
                .fmap_err(_fail_from_invalid_config)
                .fmap(lambda config: configure_run(run_id, config))
            )
        )
        .and_then(
            lambda run: (
                save_trained_run(run)
                .fmap_err(FailTrainRun)
                .fmap(lambda checkpoint: complete_training(run, checkpoint))
            )
        )
    )


# --- (2) Queries ---

# Read-side port (DIP via callables): FindRunSummaryFn returns the read model keyed by
# RunID -- a RunSummaryDTO projected straight from the run's {run_id}.json manifest --
# BYPASSING the TrainingRun aggregate and its certification. FIND a view to show it (the
# read counterpart of the write side's LOAD-to-act-on-it). The DTO is a legal query
# read model at the I/O boundary; a lookup miss or read failure folds into an ErrorInfo
# cause. Never returns an aggregate -- that would be the read-model bypass smell
# (CQRS_004).
FindRunSummaryFn = Callable[[RunID], Result[RunSummaryDTO, ErrorInfo]]


@dataclass(frozen=True, slots=True)
class GetRunSummary:
    """Query: fetch the read-model summary of a persisted run by its id.

    Carries only edge data -- the run id to look up, as a bare ``str`` (an edge DTO
    holds primitives; the handler re-wraps it into a ``RunID`` via the domain
    :func:`make_runid` before the read port is touched). The read port is injected into
    the handler, not the query.
    """

    run_id: str


# The union of every mlmodels read query -- one member today, dispatched by match/case
# at the delivery edge. Grows as queries are added (no base class).
type MLModelQuery = GetRunSummary


@dataclass(frozen=True, slots=True)
class FailGetRunSummary:
    """Failure of handle_get_run_summary, carrying its structured ``cause``.

    An invalid id and a read-boundary ``ErrorInfo`` both fold into one application
    error, so the failure crosses UP the ring as a serializable descriptor -- never the
    domain type or a lower-layer exception.
    """

    cause: ErrorInfo


def _fail_get_from_invalid_id(error: InvalidRunID) -> FailGetRunSummary:
    """Lift a domain id-validation rejection into the query handler's own error.

    The offending id string stays in the domain error; the boundary reports a stable
    ``code`` plus message, never the domain type -- symmetric with the write side's
    :func:`_fail_from_invalid_id`. The rejection is logged here at the boundary (the
    domain core stays free of logging).
    """
    logger.warning("run id rejected on read: %r is not a valid id", error.run_id)
    return FailGetRunSummary(
        ErrorInfo(code="INVALID_RUN_ID", message=f"invalid run id: {error.run_id!r}")
    )


def handle_get_run_summary(
    *, find: FindRunSummaryFn, query: GetRunSummary
) -> Result[RunSummaryDTO, FailGetRunSummary]:
    """Fetch the read-model summary for ``query.run_id`` from the run manifest.

    Re-wraps the edge ``str`` id into a ``RunID`` via the domain :func:`make_runid` (an
    invalid id short-circuits to ``FailGetRunSummary`` and the read port is never
    touched), then binds the injected ``FindRunSummaryFn`` and returns its RunSummaryDTO
    on the success rail -- the read model projected from the manifest, never the
    TrainedRun aggregate (loading an aggregate only to show it is the read-model bypass
    smell, CQRS_004). A read-boundary ``ErrorInfo`` folds into ``FailGetRunSummary``.
    Pure railway composition -- no branching on the ``Result``.
    """
    return (
        make_runid(query.run_id)
        .fmap_err(_fail_get_from_invalid_id)
        .and_then(lambda run_id: find(run_id).fmap_err(FailGetRunSummary))
    )
