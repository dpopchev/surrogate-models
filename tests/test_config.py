"""Tests for the app-wide settings shell -- get_settings (pydantic-settings).

get_settings() reads configuration once and caches it. Resolution order, highest
first: OS environment / .env, then the surrogate_models.toml file, then the field
defaults. Every test isolates that resolution: it clears the cache, drops the
datasets-path env var, and runs in a fresh working directory so no real .env or
surrogate_models.toml on disk (nor the Makefile's exported .env) leaks in. One
assert per test.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from surrogate_models.config import get_settings

DATASETS_PATH_ENV = "SURROGATE_MODELS__DATASETS__PATH"
NEUTRON_STARS_SOURCE_ENV = "SURROGATE_MODELS__DATASETS__NEUTRON_STARS_SOURCE"
TOML_FILE = "surrogate_models.toml"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give each test a clean, deterministic settings environment.

    Drops the datasets-path env var, moves into an empty working directory (so no
    real .env / surrogate_models.toml is discovered), and clears the get_settings
    cache before and after -- the cache is process-global, so an un-cleared entry
    would bleed one test's configuration into the next.
    """
    monkeypatch.delenv(DATASETS_PATH_ENV, raising=False)
    monkeypatch.delenv(NEUTRON_STARS_SOURCE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_datasets_path_defaults_under_var_data() -> None:
    assert get_settings().datasets.path == Path("var/data/surrogate_models/datasets")


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


def test_neutron_stars_source_defaults_to_none() -> None:
    assert get_settings().datasets.neutron_stars_source is None


def test_env_overrides_neutron_stars_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEUTRON_STARS_SOURCE_ENV, "/data/neutron-stars.dat")
    source = get_settings().datasets.neutron_stars_source
    assert source == Path("/data/neutron-stars.dat")
