"""Mlmodels domain -- the pure functional core of the mlmodels bounded context.

Holds the TrainingRun aggregate as states-as-types (ConfiguredRun -> TrainedRun),
the run identity (RunID) and training configuration (TrainingConfig) as validated
value objects, the Checkpoint reference a completed run points at, and the
validation errors returned when an id or a training knob is malformed. No torch, no
I/O, no raise: the model is GIVEN to the shell as an injected callable and the
training itself happens there; the domain only records a run's certified
configuration and the checkpoint the shell produced. Failures travel the railway as
Result values.

The smart constructors (make_runid, make_training_config) and the pure transitions
(configure_run, complete_training) are added under TDD. A TrainedRun is reachable
only through complete_training applied to a ConfiguredRun, so every trained run in
the system was a validly configured run first -- the "trained but never configured"
and "trained without an artifact" states are unrepresentable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType

from surrogate_models.railway_adts import Err, Ok, Result

RunID = NewType("RunID", str)

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class InvalidRunID:
    """Validation failure: a candidate id is not a valid RunID.

    Carries the offending ``run_id`` string -- the value that failed certification --
    per the boundary rule that a pure validation failure holds the offending value.
    """

    run_id: str


@dataclass(frozen=True, slots=True)
class InvalidMaxEpochs:
    """Validation failure: the epoch budget is below the one-epoch minimum.

    Carries the offending ``max_epochs`` -- the value that failed certification.
    """

    max_epochs: int


@dataclass(frozen=True, slots=True)
class InvalidLearningRate:
    """Validation failure: the learning rate is not strictly positive.

    Carries the offending ``learning_rate`` -- the value that failed certification.
    """

    learning_rate: float


@dataclass(frozen=True, slots=True)
class InvalidBatchSize:
    """Validation failure: the batch size is below one.

    Carries the offending ``batch_size`` -- the value that failed certification.
    """

    batch_size: int


@dataclass(frozen=True, slots=True)
class InvalidOptimizer:
    """Validation failure: the optimizer name is not a supported choice.

    Carries the offending ``optimizer`` -- the value that failed certification.
    """

    optimizer: str


# Every way make_training_config can reject a candidate configuration -- one typed
# member per field, each carrying its own offending value. The application lifts this
# union into a single coded FailTrainRun via an exhaustive match at the boundary.
type TrainingConfigError = (
    InvalidMaxEpochs | InvalidLearningRate | InvalidBatchSize | InvalidOptimizer
)

# The optimiser names the domain certifies. Authoritative here (the pure core owns the
# vocabulary); infrastructure maps each name to its concrete torch optimiser class.
SUPPORTED_OPTIMIZERS: frozenset[str] = frozenset({"sgd", "adam"})


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """A certified training configuration -- the knobs a run trains under.

    Every field is certified by make_training_config: ``max_epochs`` (passes over the
    data, >= 1), ``learning_rate`` (the optimiser step size, strictly > 0),
    ``batch_size`` (rows per gradient step, >= 1), and ``optimizer`` (which optimiser
    to use, one of SUPPORTED_OPTIMIZERS). The shell reads every field FROM this one
    certified object -- the model's optimiser from ``learning_rate`` + ``optimizer``,
    the DataLoader from ``batch_size``, the Trainer from ``max_epochs`` -- so an
    invalid combination cannot reach training.
    """

    max_epochs: int
    learning_rate: float
    batch_size: int
    optimizer: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A reference to the trained-model artifact the shell persisted.

    ``location`` is the filesystem path of the Lightning ``.ckpt`` the trainer wrote.
    The domain treats it as an opaque handle -- it records WHERE the trained weights
    live without doing any I/O itself.
    """

    location: str


@dataclass(frozen=True, slots=True)
class ConfiguredRun:
    """The initial TrainingRun state: a run certified and ready to train.

    Pairs the run identity with its certified TrainingConfig. The only way to reach
    the terminal TrainedRun is complete_training applied to a ConfiguredRun -- so an
    un-configured run can never be trained.
    """

    run_id: RunID
    config: TrainingConfig


@dataclass(frozen=True, slots=True)
class TrainedRun:
    """The terminal TrainingRun state: a run whose training produced a checkpoint.

    Reachable only via complete_training(ConfiguredRun, Checkpoint), so a TrainedRun
    always carries the checkpoint its training wrote -- the illegal "trained but no
    artifact" state is unrepresentable.
    """

    run_id: RunID
    config: TrainingConfig
    checkpoint: Checkpoint


def make_runid(run_id: str) -> Result[RunID, InvalidRunID]:
    """Certify a candidate ``run_id`` string into a RunID, or reject it.

    The SOLE way to build a RunID -- so every RunID in the system has passed this
    check. Pure and total: validate + wrap, no minting. A valid id is a non-empty,
    filesystem-safe stem (``^[A-Za-z0-9._-]+$``), because the id becomes the
    ``{id}.ckpt`` checkpoint filename downstream; anything else (blank, whitespace, a
    path separator) returns ``Err(InvalidRunID)`` carrying the offending string.
    """
    if _RUN_ID_PATTERN.match(run_id):
        return Ok(RunID(run_id))
    return Err(InvalidRunID(run_id))


def make_training_config(
    max_epochs: int,
    learning_rate: float,
    batch_size: int,
    optimizer: str,
) -> Result[TrainingConfig, TrainingConfigError]:
    """Certify the four training knobs, or reject the first invalid field.

    The sole way to build a TrainingConfig. Pure and total: each field is checked in
    turn and the FIRST failure short-circuits to its own typed error carrying the
    offending value -- ``max_epochs`` >= 1 (InvalidMaxEpochs), ``learning_rate`` > 0
    (InvalidLearningRate), ``batch_size`` >= 1 (InvalidBatchSize), and ``optimizer``
    in SUPPORTED_OPTIMIZERS (InvalidOptimizer). Only an all-valid combination yields
    ``Ok(TrainingConfig)``.
    """
    if max_epochs < 1:
        return Err(InvalidMaxEpochs(max_epochs))
    if learning_rate <= 0:
        return Err(InvalidLearningRate(learning_rate))
    if batch_size < 1:
        return Err(InvalidBatchSize(batch_size))
    if optimizer not in SUPPORTED_OPTIMIZERS:
        return Err(InvalidOptimizer(optimizer))
    return Ok(TrainingConfig(max_epochs, learning_rate, batch_size, optimizer))


def configure_run(run_id: RunID, config: TrainingConfig) -> ConfiguredRun:
    """Build the initial ConfiguredRun from a certified id and configuration.

    The aggregate's entry transition. Total: both inputs are already validated value
    objects, so pairing them into the initial state cannot fail -- the failure rails
    live in the smart constructors that produced the inputs.
    """
    return ConfiguredRun(run_id, config)


def complete_training(run: ConfiguredRun, checkpoint: Checkpoint) -> TrainedRun:
    """Advance a ConfiguredRun to the terminal TrainedRun, recording ``checkpoint``.

    The aggregate's completion transition -- the ONLY way to reach TrainedRun, so a
    trained run always carries the checkpoint its training wrote. Total and pure: the
    shell has already produced the checkpoint; this only records the outcome, keeping
    the run's identity and configuration.
    """
    return TrainedRun(run.run_id, run.config, checkpoint)
