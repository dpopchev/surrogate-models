"""Tests for the app-wide settings shell -- get_settings (pydantic-settings).

get_settings() reads configuration once and caches it. Resolution order, highest
first: OS environment / .env, then the surrogate_models.toml file, then the field
defaults. Every test isolates that resolution: it clears the cache, drops the
datasets-path env var, and runs in a fresh working directory so no real .env or
surrogate_models.toml on disk (nor the Makefile's exported .env) leaks in. One
assert per test.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from surrogate_models.config import get_settings

DATASETS_PATH_ENV = "SURROGATE_MODELS__DATASETS__PATH"
NEUTRON_STARS_SOURCE_ENV = "SURROGATE_MODELS__DATASETS__NEUTRON_STARS_SOURCE"
LOGGING_LEVEL_ENV = "SURROGATE_MODELS__LOGGING__LEVEL"
LOGGING_FILE_ENV = "SURROGATE_MODELS__LOGGING__FILE"
MLMODELS_CHECKPOINT_DIR_ENV = "SURROGATE_MODELS__MLMODELS__CHECKPOINT_DIR"
TOML_FILE = "surrogate_models.toml"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give each test a clean, deterministic settings environment.

    Drops the datasets env vars, points the log file at ``tmp_path`` (so
    get_settings never touches ``/var`` and never opens a shared file), moves into
    an empty working directory (so no real .env / surrogate_models.toml is
    discovered), and clears the get_settings cache before and after -- the cache is
    process-global, so an un-cleared entry would bleed one test's configuration into
    the next. On teardown it also strips any handler get_settings attached to the
    root logger and restores the root level, since configure-on-get mutates that
    process-global logger.
    """
    monkeypatch.delenv(DATASETS_PATH_ENV, raising=False)
    monkeypatch.delenv(NEUTRON_STARS_SOURCE_ENV, raising=False)
    monkeypatch.delenv(LOGGING_LEVEL_ENV, raising=False)
    monkeypatch.setenv(LOGGING_FILE_ENV, str(tmp_path / "log" / "surrogate_models.log"))
    monkeypatch.delenv(MLMODELS_CHECKPOINT_DIR_ENV, raising=False)
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


def test_datasets_path_defaults_under_var_data() -> None:
    assert get_settings().datasets.path == Path("/var/data/surrogate_models/datasets")


def test_env_overrides_datasets_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATASETS_PATH_ENV, "/srv/custom/datasets")
    assert get_settings().datasets.path == Path("/srv/custom/datasets")


def test_toml_sets_datasets_path(tmp_path: Path) -> None:
    (tmp_path / TOML_FILE).write_text('[datasets]\npath = "/srv/from/toml"\n')
    assert get_settings().datasets.path == Path("/srv/from/toml")


def test_env_takes_precedence_over_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / TOML_FILE).write_text('[datasets]\npath = "/srv/from/toml"\n')
    monkeypatch.setenv(DATASETS_PATH_ENV, "/srv/from/env")
    assert get_settings().datasets.path == Path("/srv/from/env")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_neutron_stars_source_defaults_under_data() -> None:
    source = get_settings().datasets.neutron_stars_source
    assert source == Path("/data/neutron-stars/neutron-stars.dat")


def test_env_overrides_neutron_stars_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEUTRON_STARS_SOURCE_ENV, "/data/neutron-stars.dat")
    source = get_settings().datasets.neutron_stars_source
    assert source == Path("/data/neutron-stars.dat")


def test_logging_level_defaults_info() -> None:
    assert get_settings().logging.level == "INFO"


def test_env_overrides_logging_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOGGING_LEVEL_ENV, "debug")
    assert get_settings().logging.level == "DEBUG"


def test_unknown_logging_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOGGING_LEVEL_ENV, "bogus")
    with pytest.raises(ValidationError):
        get_settings()


def test_logging_file_defaults_under_var_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOGGING_FILE_ENV, raising=False)
    assert get_settings().logging.file == Path(
        "/var/data/surrogate_models/log/surrogate_models.log"
    )


def test_env_overrides_logging_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "custom" / "app.log"
    monkeypatch.setenv(LOGGING_FILE_ENV, str(target))
    assert get_settings().logging.file == target


def test_relative_logging_file_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOGGING_FILE_ENV, "relative/app.log")
    with pytest.raises(ValidationError):
        get_settings()


def test_get_settings_configures_root_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOGGING_LEVEL_ENV, "WARNING")
    get_settings()
    assert logging.getLogger().level == logging.WARNING


def test_get_settings_does_not_stack_file_handlers(tmp_path: Path) -> None:
    expected = str(tmp_path / "log" / "surrogate_models.log")
    get_settings()
    get_settings.cache_clear()
    get_settings()
    ours = [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "baseFilename", None) == expected
    ]
    assert len(ours) == 1


def test_mlmodels_checkpoint_dir_defaults_under_var_data() -> None:
    assert get_settings().mlmodels.checkpoint_dir == Path(
        "/var/data/surrogate_models/checkpoints"
    )


def test_env_overrides_mlmodels_checkpoint_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MLMODELS_CHECKPOINT_DIR_ENV, "/srv/custom/ckpt")
    assert get_settings().mlmodels.checkpoint_dir == Path("/srv/custom/ckpt")


def test_relative_mlmodels_checkpoint_dir_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MLMODELS_CHECKPOINT_DIR_ENV, "relative/ckpt")
    with pytest.raises(ValidationError):
        get_settings()
