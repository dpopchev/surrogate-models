"""Tests for the mlmodels context root -- the settings-driven train_run facade.

train_run(cmd) is the public composition seat: it reads ``mlmodels.checkpoint_dir``
from get_settings(), curries the src save_trained_run adapter on it, and drives the
train handler, returning the checkpoint location (raising at the outer edge on
failure). The thin training slice has landed, so the facade's happy path is now
reachable: train_run returns the path of the ``{run_id}.ckpt`` the adapter wrote under
the configured dir -- these tests pin that reachable contract AND that the CONFIGURED
checkpoint dir actually reaches the adapter (proving the settings wiring). When the
EXPAND slice trains for real, the first test grows a "trained weights" assertion, but
the path contract holds. Settings are resolved via get_settings(), so each test
isolates the environment (env overrides + cache clear) as tests/datasets/test_main.py
does. One assert per test.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from surrogate_models import TrainRun, train_run
from surrogate_models.config import get_settings

CHECKPOINT_DIR_ENV = "SURROGATE_MODELS__MLMODELS__CHECKPOINT_DIR"
LOGGING_LEVEL_ENV = "SURROGATE_MODELS__LOGGING__LEVEL"
LOGGING_FILE_ENV = "SURROGATE_MODELS__LOGGING__FILE"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give each test a clean, deterministic settings environment.

    Drops the checkpoint env var, points the log file at ``tmp_path`` (so
    get_settings, which train_run calls, never touches ``/var`` while configuring
    logging), moves into an empty working directory (so no real .env /
    surrogate_models.toml is discovered), and clears the process-global get_settings
    cache before and after so no configuration bleeds across tests. On teardown it
    strips any handler get_settings attached to the root logger and restores its
    level, since configure-on-get mutates that process-global logger.
    """
    monkeypatch.delenv(CHECKPOINT_DIR_ENV, raising=False)
    monkeypatch.delenv(LOGGING_LEVEL_ENV, raising=False)
    monkeypatch.setenv(LOGGING_FILE_ENV, str(tmp_path / "log" / "surrogate_models.log"))
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    for handler in root.handlers[:]:
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def _cmd(run_id: str = "manual") -> TrainRun:
    """A valid TrainRun for the settings-driven facade path."""
    return TrainRun(
        run_id, max_epochs=1, learning_rate=0.01, batch_size=2, optimizer="sgd"
    )


def test_train_run_returns_the_written_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setenv(CHECKPOINT_DIR_ENV, str(checkpoints))
    assert train_run(_cmd()) == str(checkpoints / "manual.ckpt")


def test_train_run_binds_the_configured_checkpoint_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setenv(CHECKPOINT_DIR_ENV, str(checkpoints))
    with caplog.at_level(logging.INFO):
        train_run(_cmd())
    assert any(str(checkpoints) in record.getMessage() for record in caplog.records)
