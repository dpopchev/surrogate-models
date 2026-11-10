"""Facts about reading the paper's choices from a TOML file."""

import os
import tomllib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from paperlib.config import (
    PAPER_TOML,
    NetActivation,
    NetLoss,
    load_config,
    load_section,
)
from paperlib.surrogate import Activation, Loss


@pytest.fixture(autouse=True)
def _no_paper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in os.environ if n.startswith("PAPER__")]:
        monkeypatch.delenv(name)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "paper.toml"
    path.write_text(text)
    return path


@pytest.mark.edge
def test_an_empty_file_gives_the_default_network(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")).network.width == 64


@pytest.mark.edge
def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[network]\ncolour = "red"\n')
    with pytest.raises(ValidationError, match="colour"):
        load_config(toml)


@pytest.mark.edge
def test_an_unknown_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="results"):
        load_config(write(tmp_path, "[results]\nx = 1\n"))


@pytest.mark.edge
def test_an_environment_variable_overrides_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPER__PLOT__PALETTE", "deep")
    assert load_config(write(tmp_path, '[plot]\npalette = "muted"\n')).plot.palette == "deep"


@pytest.mark.edge
def test_the_table_layout_defaults_to_small(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")).tables.layout == "small"


@pytest.mark.edge
def test_the_baseline_network_defaults_to_relu(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")).network.activation == "relu"


@pytest.mark.edge
def test_the_baseline_trains_on_one_thread_by_default(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")).network.threads == 1


@pytest.mark.edge
def test_the_baseline_runs_on_six_workers_by_default(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")).network.workers == 6


@pytest.mark.edge
def test_zero_threads_are_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, "[network]\nthreads = 0\n")
    with pytest.raises(ValidationError, match="threads"):
        load_config(toml)


@pytest.mark.edge
def test_an_activation_the_network_cannot_build_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[network]\nactivation = "swish"\n')
    with pytest.raises(ValidationError, match="activation"):
        load_config(toml)


@pytest.mark.edge
def test_a_loss_the_network_cannot_use_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[network]\nloss = "l1"\n')
    with pytest.raises(ValidationError, match="loss"):
        load_config(toml)


@pytest.mark.model
def test_the_activation_vocabulary_matches_the_network() -> None:
    assert get_args(NetActivation) == get_args(Activation)


@pytest.mark.model
def test_the_loss_vocabulary_matches_the_network() -> None:
    assert get_args(NetLoss) == get_args(Loss)


@pytest.mark.edge
def test_a_log_level_other_than_info_or_debug_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[network]\nlog_level = "LOUD"\n')
    with pytest.raises(ValidationError, match="log_level"):
        load_config(toml)


@pytest.mark.edge
def test_the_environment_shortens_the_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPER__NETWORK__MAX_EPOCHS", "2")
    toml = write(tmp_path, "[network]\nmax_epochs = 500\n")
    assert load_config(toml).network.max_epochs == 2


@pytest.mark.edge
def test_the_shipped_paper_toml_sets_the_baseline_network() -> None:
    with PAPER_TOML.open("rb") as handle:
        assert "network" in tomllib.load(handle)


class _Knobs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = 1
    name: str = "a"


@pytest.mark.edge
def test_section_settings_come_from_the_section_toml(tmp_path: Path) -> None:
    toml = write(tmp_path, 'width = 8\nname = "b"\n')
    assert load_section(_Knobs, toml) == _Knobs(width=8, name="b")
