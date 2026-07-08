"""Tests for the mlmodels application -- handle_train_run across its rails.

The handler composes two injected callables on the railway: build_model GIVES the
model (built FROM the certified config) and train runs it for the run's id, returning
the produced Checkpoint (or an ErrorInfo cause). Ports are injected as typed lambdas
-- never a mock. The rails covered: the ok path folds the checkpoint into a
TrainedRun; an invalid id and any invalid config knob short-circuit to FailTrainRun
before training; and a training-boundary ErrorInfo folds into FailTrainRun. One
assert per test.
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


def _cmd(
    run_id: str = "r1",
    data: object = "DATA",
    max_epochs: int = 1,
    learning_rate: float = 0.01,
    batch_size: int = 2,
    optimizer: str = "sgd",
) -> TrainRun:
    """Build a valid TrainRun; override one knob to probe a rail."""
    return TrainRun(run_id, data, max_epochs, learning_rate, batch_size, optimizer)


def _model(config: TrainingConfig) -> object:
    """A stand-in for the GIVEN model -- a stable string, ignoring the config."""
    return "MODEL"


def _unreachable_model(config: TrainingConfig) -> object:
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
    result = handle_train_run(build_model=_model, train=_ok_train, cmd=_cmd())
    assert isinstance(result.unwrap(), TrainedRun)


def test_ok_carries_the_run_id() -> None:
    result = handle_train_run(build_model=_model, train=_ok_train, cmd=_cmd())
    assert result.unwrap().run_id == RunID("r1")


def test_ok_carries_the_checkpoint_from_train() -> None:
    result = handle_train_run(build_model=_model, train=_ok_train, cmd=_cmd())
    assert result.unwrap().checkpoint == Checkpoint("r1.ckpt")


def test_ok_gives_run_id_model_and_certified_config_to_train() -> None:
    result = handle_train_run(
        build_model=_model, train=_echo_train, cmd=_cmd(max_epochs=2)
    )
    assert result.unwrap().checkpoint == Checkpoint("r1-MODEL-DATA-e2")


# --- invalid id rail ---


def test_invalid_id_is_err() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=_cmd(run_id="bad id"),
    )
    assert result.is_err() is True


def test_invalid_id_error_is_fail_train_run() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=_cmd(run_id="bad id"),
    )
    assert isinstance(result.unwrap_err(), FailTrainRun)


# --- invalid config rail (one per knob folds to FailTrainRun with its own code) ---


def test_invalid_epochs_error_is_fail_train_run() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=_cmd(max_epochs=0),
    )
    assert isinstance(result.unwrap_err(), FailTrainRun)


def test_invalid_epochs_reports_the_epochs_code() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=_cmd(max_epochs=0),
    )
    assert result.unwrap_err().cause.code == "INVALID_MAX_EPOCHS"


def test_invalid_learning_rate_reports_the_learning_rate_code() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=_cmd(learning_rate=0.0),
    )
    assert result.unwrap_err().cause.code == "INVALID_LEARNING_RATE"


def test_invalid_batch_size_reports_the_batch_size_code() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=_cmd(batch_size=0),
    )
    assert result.unwrap_err().cause.code == "INVALID_BATCH_SIZE"


def test_invalid_optimizer_reports_the_optimizer_code() -> None:
    result = handle_train_run(
        build_model=_unreachable_model,
        train=_unreachable_train,
        cmd=_cmd(optimizer="rmsprop"),
    )
    assert result.unwrap_err().cause.code == "INVALID_OPTIMIZER"


# --- training-boundary failure rail ---


def test_training_failure_is_err() -> None:
    result = handle_train_run(build_model=_model, train=_failing_train, cmd=_cmd())
    assert result.is_err() is True


def test_training_failure_folds_the_cause_into_fail_train_run() -> None:
    result = handle_train_run(build_model=_model, train=_failing_train, cmd=_cmd())
    assert result.unwrap_err().cause == _TRAIN_FAILURE
