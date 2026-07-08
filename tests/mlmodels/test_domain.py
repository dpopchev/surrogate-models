"""Tests for the mlmodels domain -- the TrainingRun states and smart constructors.

Two smart constructors gate the value objects: make_runid (a non-empty,
filesystem-safe stem ``^[A-Za-z0-9._-]+$``, since the id becomes the ``{id}.ckpt``
filename) and make_training_config (four knobs -- ``max_epochs`` >= 1,
``learning_rate`` > 0, ``batch_size`` >= 1, ``optimizer`` in SUPPORTED_OPTIMIZERS --
each with its own typed rejection carrying the offending value). Two pure transitions
move the aggregate: configure_run builds the initial ConfiguredRun, and
complete_training advances it to the terminal TrainedRun carrying the produced
Checkpoint. One assert per test.
"""

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
    InvalidBatchSize,
    InvalidLearningRate,
    InvalidMaxEpochs,
    InvalidOptimizer,
    InvalidRunID,
    RunID,
    TrainedRun,
    TrainingConfig,
    TrainingConfigError,
    complete_training,
    configure_run,
    make_runid,
    make_training_config,
)
from surrogate_models.railway_adts import Result


def _make_config(
    max_epochs: int = 1,
    learning_rate: float = 0.01,
    batch_size: int = 2,
    optimizer: str = "sgd",
) -> Result[TrainingConfig, TrainingConfigError]:
    """Certify a config; override one knob to probe a rail."""
    return make_training_config(max_epochs, learning_rate, batch_size, optimizer)


def _training_config(max_epochs: int = 1) -> TrainingConfig:
    """Build a valid TrainingConfig value object directly for the transition tests."""
    return TrainingConfig(max_epochs, 0.01, 2, "sgd")


# --- make_runid: the sole str -> RunID smart constructor ---


def test_make_runid_ok_on_valid_id() -> None:
    assert make_runid("run-01").is_ok() is True


def test_make_runid_ok_returns_the_id() -> None:
    assert make_runid("run-01").unwrap() == RunID("run-01")


def test_make_runid_err_on_empty_string() -> None:
    assert make_runid("").is_err() is True


def test_make_runid_err_on_whitespace() -> None:
    assert make_runid("  ").is_err() is True


def test_make_runid_err_on_path_separator() -> None:
    assert make_runid("nested/run").is_err() is True


def test_make_runid_err_is_invalid_runid() -> None:
    assert isinstance(make_runid("").unwrap_err(), InvalidRunID)


def test_make_runid_err_carries_the_offending_string() -> None:
    assert make_runid("bad run").unwrap_err().run_id == "bad run"


# --- make_training_config: certifies four knobs, first bad field wins ---


def test_make_training_config_ok_on_valid_knobs() -> None:
    assert _make_config().is_ok() is True


def test_make_training_config_ok_carries_max_epochs() -> None:
    assert _make_config(max_epochs=5).unwrap().max_epochs == 5


def test_make_training_config_ok_carries_learning_rate() -> None:
    assert _make_config(learning_rate=0.5).unwrap().learning_rate == 0.5


def test_make_training_config_ok_carries_batch_size() -> None:
    assert _make_config(batch_size=8).unwrap().batch_size == 8


def test_make_training_config_ok_carries_optimizer() -> None:
    assert _make_config(optimizer="adam").unwrap().optimizer == "adam"


def test_make_training_config_rejects_zero_epochs() -> None:
    assert _make_config(max_epochs=0).unwrap_err() == InvalidMaxEpochs(0)


def test_make_training_config_rejects_negative_epochs() -> None:
    assert _make_config(max_epochs=-3).unwrap_err() == InvalidMaxEpochs(-3)


def test_make_training_config_rejects_zero_learning_rate() -> None:
    assert _make_config(learning_rate=0.0).unwrap_err() == InvalidLearningRate(0.0)


def test_make_training_config_rejects_negative_learning_rate() -> None:
    assert _make_config(learning_rate=-0.1).unwrap_err() == InvalidLearningRate(-0.1)


def test_make_training_config_rejects_zero_batch_size() -> None:
    assert _make_config(batch_size=0).unwrap_err() == InvalidBatchSize(0)


def test_make_training_config_rejects_unknown_optimizer() -> None:
    assert _make_config(optimizer="rmsprop").unwrap_err() == InvalidOptimizer("rmsprop")


def test_make_training_config_reports_the_first_bad_field() -> None:
    assert _make_config(max_epochs=0, learning_rate=-1.0).unwrap_err() == (
        InvalidMaxEpochs(0)
    )


# --- configure_run: build the initial ConfiguredRun state ---


def test_configure_run_returns_a_configured_run() -> None:
    run = configure_run(RunID("r1"), _training_config(1))
    assert isinstance(run, ConfiguredRun)


def test_configure_run_carries_the_run_id() -> None:
    run = configure_run(RunID("r1"), _training_config(1))
    assert run.run_id == RunID("r1")


def test_configure_run_carries_the_config() -> None:
    run = configure_run(RunID("r1"), _training_config(3))
    assert run.config == _training_config(3)


# --- complete_training: advance ConfiguredRun -> TrainedRun with the checkpoint ---


def test_complete_training_returns_a_trained_run() -> None:
    run = configure_run(RunID("r1"), _training_config(1))
    assert isinstance(complete_training(run, Checkpoint("r1.ckpt")), TrainedRun)


def test_complete_training_carries_the_checkpoint() -> None:
    run = configure_run(RunID("r1"), _training_config(1))
    assert complete_training(run, Checkpoint("r1.ckpt")).checkpoint == Checkpoint(
        "r1.ckpt"
    )


def test_complete_training_preserves_the_run_id() -> None:
    run = configure_run(RunID("r1"), _training_config(1))
    assert complete_training(run, Checkpoint("r1.ckpt")).run_id == RunID("r1")


def test_complete_training_preserves_the_config() -> None:
    run = configure_run(RunID("r1"), _training_config(7))
    assert complete_training(run, Checkpoint("r1.ckpt")).config == _training_config(7)
