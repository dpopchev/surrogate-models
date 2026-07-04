"""Tests for the datasets context root -- the load_neutron_stars facade.

load_neutron_stars() is the public get-or-build entry point: it reads the stored
``neutron-stars`` frame from ``datasets.path`` when present, and otherwise digests
``datasets.neutron_stars_source``, persists the dataset, and returns the re-read
frame (read-through). It returns a bare DataFrame, raising at the outer edge on
failure. Settings are resolved via get_settings(), so each test isolates the
environment (env overrides + cache clear) exactly as tests/test_config.py does.
One assert per test.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from surrogate_models import load_neutron_stars
from surrogate_models.config import get_settings
from surrogate_models.railway_adts import UnwrapError

DATASETS_PATH_ENV = "SURROGATE_MODELS__DATASETS__PATH"
NEUTRON_STARS_SOURCE_ENV = "SURROGATE_MODELS__DATASETS__NEUTRON_STARS_SOURCE"

ONE_BATCH = """\
# x1 = 0.005, x2 = 1000.0, beta = 0.4, Lambda = 0.5, Pc = 5e-05, choice_theory = 31
rhoc Pc M
5.89e14 5.0e-05 0.56
6.06e14 5.5e-05 0.59
"""


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give each test a clean, deterministic settings environment.

    Drops both datasets env vars, moves into an empty working directory (so no real
    .env / surrogate_models.toml is discovered), and clears the process-global
    get_settings cache before and after so no configuration bleeds across tests.
    """
    monkeypatch.delenv(DATASETS_PATH_ENV, raising=False)
    monkeypatch.delenv(NEUTRON_STARS_SOURCE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_load_neutron_stars_returns_stored_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    frame = pd.DataFrame({"x": [1, 2, 3]})
    frame.to_parquet(store / "neutron-stars.parquet")
    monkeypatch.setenv(DATASETS_PATH_ENV, str(store))
    assert load_neutron_stars().equals(frame)


def test_load_neutron_stars_builds_from_source_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "store"
    source = tmp_path / "neutron-stars.dat"
    source.write_text(ONE_BATCH)
    monkeypatch.setenv(DATASETS_PATH_ENV, str(store))
    monkeypatch.setenv(NEUTRON_STARS_SOURCE_ENV, str(source))
    assert "pc_init" in load_neutron_stars().columns


def test_load_neutron_stars_persists_built_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "store"
    source = tmp_path / "neutron-stars.dat"
    source.write_text(ONE_BATCH)
    monkeypatch.setenv(DATASETS_PATH_ENV, str(store))
    monkeypatch.setenv(NEUTRON_STARS_SOURCE_ENV, str(source))
    load_neutron_stars()
    assert (store / "neutron-stars.parquet").is_file()


def test_load_neutron_stars_raises_when_source_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "store"
    source = tmp_path / "absent.dat"
    monkeypatch.setenv(DATASETS_PATH_ENV, str(store))
    monkeypatch.setenv(NEUTRON_STARS_SOURCE_ENV, str(source))
    with pytest.raises(UnwrapError):
        load_neutron_stars()
