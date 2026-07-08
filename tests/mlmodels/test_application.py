"""Tests for the mlmodels application -- handle_train_run across its four rails.

The handler composes two injected callables on the railway: build_model GIVES the
model and train runs it for the run's id, returning the produced Checkpoint (or an
ErrorInfo cause). Ports are injected as typed lambdas -- never a mock. Four rails are
covered: the ok path folds the checkpoint into a TrainedRun; an invalid id and an
invalid config short-circuit to FailTrainRun before training; and a training-boundary
ErrorInfo folds into FailTrainRun. One assert per test.
"""

from surrogate_models.mlmodels.application import (
    FailTrainRun,
    TrainRun,
    handle_train_run,
)
from surrogate_models.mlmodels.domain import (
    Checkpoint,
    RunID,
    TrainedRun,
    TrainingConfig,
)
from surrogate_models.railway_adts import Err, ErrorInfo, Ok, Result

_TRAIN_FAILURE = ErrorInfo(code="TRAINING_FAILED", message="boom")


def _model() -> object:
    """A stand-in for the GIVEN model -- a stable, deterministic string."""
    return "MODEL"


def _unreachable_model() -> object:
    """A build_model that fails the test if the handler builds a model at all."""
    raise AssertionError("build_model must not be called when configuration fails")


def _ok_train(
    run_id: RunID, model: object, data: object, config: TrainingConfig
) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that always succeeds, ignoring its inputs, with a fixed checkpoint."""
    return Ok(Checkpoint("r1.ckpt"))


def _echo_train(
    run_id: RunID, model: object, data: object, config: TrainingConfig
) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that echoes what it received into the checkpoint location.

    Lets one assert prove the run id, the built model, the handed-in data, and the
    CERTIFIED config all reach the trainer -- i.e. the model was truly GIVEN through.
    """
    return Ok(Checkpoint(f"{run_id}-{model}-{data}-e{config.max_epochs}"))


def _failing_train(
    run_id: RunID, model: object, data: object, config: TrainingConfig
) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that fails at the boundary, returning the ErrorInfo cause."""
    return Err(_TRAIN_FAILURE)


def _unreachable_train(
    run_id: RunID, model: object, data: object, config: TrainingConfig
) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that fails the test if the handler trains despite a bad config."""
    raise AssertionError("train must not be called when configuration fails")


# --- ok rail ---


def test_ok_returns_a_trained_run() -> None:
    result = handle_train_run(
        build_model=_model, train=_ok_train, cmd=TrainRun("r1", "DATA", 1)
    )
    assert isinstance(result.unwrap(), TrainedRun)


def test_ok_carries_the_run_id() -> None:
    result = handle_train_run(
        build_model=_model, train=_ok_train, cmd=TrainRun("r1", "DATA", 1)
    )
    assert result.unwrap().run_id == RunID("r1")


def test_ok_carries_the_checkpoint_from_train() -> None:
    result = handle_train_run(
        build_model=_model, train=_ok_train, cmd=TrainRun("r1", "DATA", 1)
    )
    assert result.unwrap().checkpoint == Checkpoint("r1.ckpt")


def test_ok_gives_run_id_model_and_certified_config_to_train() -> None:
    result = handle_train_run(
        build_model=_model, train=_echo_train, cmd=TrainRun("r1", "DATA", 2)
    )
    assert result.unwrap().checkpoint == Checkpoint("r1-MODEL-DATA-e2")


# --- invalid id rail ---


def test_invalid_id_is_err() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=TrainRun("bad id", "DATA", 1),
    )
    assert result.is_err() is True


def test_invalid_id_error_is_fail_train_run() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=TrainRun("bad id", "DATA", 1),
    )
    assert isinstance(result.unwrap_err(), FailTrainRun)


# --- invalid config rail ---


def test_invalid_config_is_err() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=TrainRun("r1", "DATA", 0),
    )
    assert result.is_err() is True


def test_invalid_config_error_is_fail_train_run() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=TrainRun("r1", "DATA", 0),
    )
    assert isinstance(result.unwrap_err(), FailTrainRun)


# --- training-boundary failure rail ---


def test_training_failure_is_err() -> None:
    result = handle_train_run(
        build_model=_model, train=_failing_train, cmd=TrainRun("r1", "DATA", 1)
    )
    assert result.is_err() is True


def test_training_failure_folds_the_cause_into_fail_train_run() -> None:
    result = handle_train_run(
        build_model=_model, train=_failing_train, cmd=TrainRun("r1", "DATA", 1)
    )
    assert result.unwrap_err().cause == _TRAIN_FAILURE
