"""Tests for the mlmodels application -- the train command and run-summary query.

The write handler composes one injected callable on the railway: save_trained_run is
GIVEN the certified ConfiguredRun aggregate and returns the produced Checkpoint (or an
ErrorInfo cause). The read handler binds an injected FindRunSummaryFn and returns the
RunSummaryDTO the read side projected. Both ports are injected as typed lambdas -- never
a mock. The rails covered: the ok path folds the checkpoint into a TrainedRun; an
invalid id and any invalid config knob short-circuit to FailTrainRun before training; a
training-boundary ErrorInfo folds into FailTrainRun; and the query returns the found
summary on its ok rail. One assert per test.
"""

from surrogate_models.mlmodels.application import (
    FailGetRunSummary,
    FailTrainRun,
    FindRunSummaryFn,
    GetRunSummary,
    TrainRun,
    handle_get_run_summary,
    handle_train_run,
)
from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
    RunID,
    RunSummaryDTO,
    TrainedRun,
)
from surrogate_models.railway_adts import Err, ErrorInfo, Ok, Result

_TRAIN_FAILURE = ErrorInfo(code="TRAINING_FAILED", message="boom")


def _cmd(
    run_id: str = "r1",
    max_epochs: int = 1,
    learning_rate: float = 0.01,
    batch_size: int = 2,
    optimizer: str = "sgd",
) -> TrainRun:
    """Build a valid TrainRun; override one knob to probe a rail."""
    return TrainRun(run_id, max_epochs, learning_rate, batch_size, optimizer)


def _ok_save(run: ConfiguredRun) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that always succeeds, ignoring its input, with a fixed checkpoint."""
    return Ok(Checkpoint("r1.ckpt"))


def _echo_save(run: ConfiguredRun) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that echoes the certified run into the checkpoint location.

    Lets one assert prove the run id AND the certified config (epochs, optimizer) reach
    the trainer -- i.e. the whole ConfiguredRun aggregate was truly GIVEN through.
    """
    return Ok(
        Checkpoint(f"{run.run_id}-e{run.config.max_epochs}-{run.config.optimizer}")
    )


def _failing_save(run: ConfiguredRun) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that fails at the boundary, returning the ErrorInfo cause."""
    return Err(_TRAIN_FAILURE)


def _unreachable_save(run: ConfiguredRun) -> Result[Checkpoint, ErrorInfo]:
    """A trainer that fails the test if the handler trains despite a bad config/id."""
    raise AssertionError("save_trained_run must not be called when configuration fails")


# --- ok rail ---


def test_ok_returns_a_trained_run() -> None:
    result = handle_train_run(save_trained_run=_ok_save, cmd=_cmd())
    assert isinstance(result.unwrap(), TrainedRun)


def test_ok_carries_the_run_id() -> None:
    result = handle_train_run(save_trained_run=_ok_save, cmd=_cmd())
    assert result.unwrap().run_id == RunID("r1")


def test_ok_carries_the_checkpoint_from_save() -> None:
    result = handle_train_run(save_trained_run=_ok_save, cmd=_cmd())
    assert result.unwrap().checkpoint == Checkpoint("r1.ckpt")


def test_ok_gives_the_certified_run_to_save() -> None:
    result = handle_train_run(save_trained_run=_echo_save, cmd=_cmd(max_epochs=2))
    assert result.unwrap().checkpoint == Checkpoint("r1-e2-sgd")


# --- invalid id rail ---


def test_invalid_id_is_err() -> None:
    result = handle_train_run(
        save_trained_run=_unreachable_save, cmd=_cmd(run_id="bad id")
    )
    assert result.is_err() is True


def test_invalid_id_error_is_fail_train_run() -> None:
    result = handle_train_run(
        save_trained_run=_unreachable_save, cmd=_cmd(run_id="bad id")
    )
    assert isinstance(result.unwrap_err(), FailTrainRun)


# --- invalid config rail (one per knob folds to FailTrainRun with its own code) ---


def test_invalid_epochs_error_is_fail_train_run() -> None:
    result = handle_train_run(
        save_trained_run=_unreachable_save, cmd=_cmd(max_epochs=0)
    )
    assert isinstance(result.unwrap_err(), FailTrainRun)


def test_invalid_epochs_reports_the_epochs_code() -> None:
    result = handle_train_run(
        save_trained_run=_unreachable_save, cmd=_cmd(max_epochs=0)
    )
    assert result.unwrap_err().cause.code == "INVALID_MAX_EPOCHS"


def test_invalid_learning_rate_reports_the_learning_rate_code() -> None:
    result = handle_train_run(
        save_trained_run=_unreachable_save, cmd=_cmd(learning_rate=0.0)
    )
    assert result.unwrap_err().cause.code == "INVALID_LEARNING_RATE"


def test_invalid_batch_size_reports_the_batch_size_code() -> None:
    result = handle_train_run(
        save_trained_run=_unreachable_save, cmd=_cmd(batch_size=0)
    )
    assert result.unwrap_err().cause.code == "INVALID_BATCH_SIZE"


def test_invalid_optimizer_reports_the_optimizer_code() -> None:
    result = handle_train_run(
        save_trained_run=_unreachable_save, cmd=_cmd(optimizer="rmsprop")
    )
    assert result.unwrap_err().cause.code == "INVALID_OPTIMIZER"


# --- training-boundary failure rail ---


def test_training_failure_is_err() -> None:
    result = handle_train_run(save_trained_run=_failing_save, cmd=_cmd())
    assert result.is_err() is True


def test_training_failure_folds_the_cause_into_fail_train_run() -> None:
    result = handle_train_run(save_trained_run=_failing_save, cmd=_cmd())
    assert result.unwrap_err().cause == _TRAIN_FAILURE


# --- Queries: handle_get_run_summary returns the projected read model ---


def _find_ok(summary: RunSummaryDTO) -> FindRunSummaryFn:
    """A read port that always returns ``summary``, ignoring the key."""
    return lambda _run_id: Ok(summary)


def _find_err(error: ErrorInfo) -> FindRunSummaryFn:
    """A read port that fails at the boundary, returning the ErrorInfo cause."""
    return lambda _run_id: Err(error)


def _unreachable_find(run_id: RunID) -> Result[RunSummaryDTO, ErrorInfo]:
    """A read port that fails the test if the handler queries despite a bad id."""
    raise AssertionError("find must not be called when the run id is invalid")


def test_get_ok_returns_the_run_summary() -> None:
    summary = RunSummaryDTO("demo", "regressor", "1.0.0")
    result = handle_get_run_summary(find=_find_ok(summary), query=GetRunSummary("demo"))
    assert result.unwrap() == summary


# --- invalid id rail (short-circuits before the read port) ---


def test_get_invalid_id_error_is_fail_get_run_summary() -> None:
    result = handle_get_run_summary(
        find=_unreachable_find, query=GetRunSummary("bad id")
    )
    assert isinstance(result.unwrap_err(), FailGetRunSummary)


def test_get_invalid_id_reports_the_run_id_code() -> None:
    result = handle_get_run_summary(
        find=_unreachable_find, query=GetRunSummary("bad id")
    )
    assert result.unwrap_err().cause.code == "INVALID_RUN_ID"


# --- read-boundary failure rail ---


def test_get_read_failure_folds_the_cause_into_fail_get_run_summary() -> None:
    cause = ErrorInfo(code="RUN_SUMMARY_READ_FAILED", message="boom")
    result = handle_get_run_summary(find=_find_err(cause), query=GetRunSummary("demo"))
    assert result.unwrap_err().cause == cause
