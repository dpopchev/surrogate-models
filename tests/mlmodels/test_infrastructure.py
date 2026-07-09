"""Tests for the mlmodels src infrastructure -- the save_trained_run placeholder.

The real training adapter (the 28-feature regressor over real data) is a later slice,
so the src save_trained_run is a torch-free placeholder that fails on the training rail
with a stable code. The end-to-end training proof lives in test_e2e, against a
test-owned stub adapter that brings its own infrastructure. One assert per test.
"""

from pathlib import Path

from surrogate_models.mlmodels.domain import (
    ConfiguredRun,
    RunID,
    TrainingConfig,
    configure_run,
)
from surrogate_models.mlmodels.infrastructure import save_trained_run


def _configured_run() -> ConfiguredRun:
    """A certified ConfiguredRun, built directly for the placeholder-adapter tests."""
    return configure_run(RunID("smoke"), TrainingConfig(1, 0.01, 2, "sgd"))


def test_save_trained_run_is_err(tmp_path: Path) -> None:
    result = save_trained_run(tmp_path, _configured_run())
    assert result.is_err() is True


def test_save_trained_run_reports_not_implemented(tmp_path: Path) -> None:
    result = save_trained_run(tmp_path, _configured_run())
    assert result.unwrap_err().code == "TRAINING_NOT_IMPLEMENTED"
