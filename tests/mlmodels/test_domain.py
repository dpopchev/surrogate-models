"""Tests for the mlmodels domain -- the TrainingRun states and smart constructors.

Two smart constructors gate the value objects: make_runid (a non-empty,
filesystem-safe stem ``^[A-Za-z0-9._-]+$``, since the id becomes the ``{id}.ckpt``
filename) and make_training_config (``max_epochs >= 1``). Two pure transitions move
the aggregate: configure_run builds the initial ConfiguredRun, and complete_training
advances it to the terminal TrainedRun carrying the produced Checkpoint. One assert
per test.
"""

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
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


# --- make_training_config: max_epochs must be >= 1 ---


def test_make_training_config_ok_on_one() -> None:
    assert make_training_config(1).is_ok() is True


def test_make_training_config_ok_carries_max_epochs() -> None:
    assert make_training_config(5).unwrap().max_epochs == 5


def test_make_training_config_err_on_zero() -> None:
    assert make_training_config(0).is_err() is True


def test_make_training_config_err_on_negative() -> None:
    assert make_training_config(-3).is_err() is True


def test_make_training_config_err_is_invalid_training_config() -> None:
    assert isinstance(make_training_config(0).unwrap_err(), InvalidTrainingConfig)


def test_make_training_config_err_carries_the_offending_value() -> None:
    assert make_training_config(0).unwrap_err().max_epochs == 0


# --- configure_run: build the initial ConfiguredRun state ---


def test_configure_run_returns_a_configured_run() -> None:
    run = configure_run(RunID("r1"), TrainingConfig(1))
    assert isinstance(run, ConfiguredRun)


def test_configure_run_carries_the_run_id() -> None:
    run = configure_run(RunID("r1"), TrainingConfig(1))
    assert run.run_id == RunID("r1")


def test_configure_run_carries_the_config() -> None:
    run = configure_run(RunID("r1"), TrainingConfig(3))
    assert run.config == TrainingConfig(3)


# --- complete_training: advance ConfiguredRun -> TrainedRun with the checkpoint ---


def test_complete_training_returns_a_trained_run() -> None:
    run = configure_run(RunID("r1"), TrainingConfig(1))
    assert isinstance(complete_training(run, Checkpoint("r1.ckpt")), TrainedRun)


def test_complete_training_carries_the_checkpoint() -> None:
    run = configure_run(RunID("r1"), TrainingConfig(1))
    assert complete_training(run, Checkpoint("r1.ckpt")).checkpoint == Checkpoint(
        "r1.ckpt"
    )


def test_complete_training_preserves_the_run_id() -> None:
    run = configure_run(RunID("r1"), TrainingConfig(1))
    assert complete_training(run, Checkpoint("r1.ckpt")).run_id == RunID("r1")


def test_complete_training_preserves_the_config() -> None:
    run = configure_run(RunID("r1"), TrainingConfig(7))
    assert complete_training(run, Checkpoint("r1.ckpt")).config == TrainingConfig(7)
