"""Facts about reading the paper's choices from a TOML file."""

import os
import tomllib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from shared.config import (
    PAPER_TOML,
    ROOT,
    NetActivation,
    NetLoss,
    PaperConfig,
    load_config,
    missing_section_folders,
)
from shared.surrogate import Activation, Loss


@pytest.fixture(autouse=True)
def _no_paper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in os.environ if n.startswith("PAPER__")]:
        monkeypatch.delenv(name)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "paper.toml"
    path.write_text(text)
    return path


def test_a_choice_loads_as_its_typed_value(tmp_path: Path) -> None:
    toml = write(tmp_path, '[data_analysis.neutron_stars]\nfigures = ["grid_fill"]\n')
    assert load_config(toml).data_analysis.neutron_stars.figures == ("grid_fill",)


def test_an_empty_file_gives_every_default_figure(tmp_path: Path) -> None:
    assert len(load_config(write(tmp_path, "")).data_analysis.neutron_stars.figures) == 7


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[data_analysis.neutron_stars]\ncolour = "red"\n')
    with pytest.raises(ValidationError, match="colour"):
        load_config(toml)


def test_an_unknown_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="results"):
        load_config(write(tmp_path, "[results]\nx = 1\n"))


def test_a_value_outside_its_literal_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[data_analysis.neutron_stars]\nfigures = ["pie"]\n')
    with pytest.raises(ValidationError, match="figures"):
        load_config(toml)


def test_an_environment_variable_overrides_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPER__PLOT__PALETTE", "deep")
    assert load_config(write(tmp_path, '[plot]\npalette = "muted"\n')).plot.palette == "deep"


class _Leaf(BaseModel):
    x: int = 0


class _Chapter(BaseModel):
    kept: _Leaf = _Leaf()
    lost: _Leaf = _Leaf()


class _Paper(BaseModel):
    chapter: _Chapter = _Chapter()


def test_a_section_without_its_folder_is_reported(tmp_path: Path) -> None:
    (tmp_path / "10_chapter" / "11_kept").mkdir(parents=True)
    assert missing_section_folders(_Paper, tmp_path) == ("chapter.lost",)


def test_the_shipped_paper_toml_selects_the_four_ns_figures_the_text_cites() -> None:
    assert len(load_config().data_analysis.neutron_stars.figures) == 4


def test_the_shipped_paper_toml_selects_the_four_bh_figures() -> None:
    assert len(load_config().data_analysis.black_holes.figures) == 4


def test_the_charge_floor_defaults_to_one_in_a_hundred_thousand() -> None:
    assert PaperConfig().data_analysis.charge_floor == 1e-5


def test_the_shipped_paper_toml_sets_the_charge_floor() -> None:
    assert load_config().data_analysis.charge_floor == 1e-5


def test_the_shipped_paper_toml_selects_every_ns_table() -> None:
    assert len(load_config().data_analysis.neutron_stars.tables) == 3


def test_the_baseline_network_defaults_to_relu(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")).methodology.algorithms.activation == "relu"


def test_the_baseline_trains_on_one_thread_by_default(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")).methodology.algorithms.threads == 1


def test_zero_threads_are_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, "[methodology.algorithms]\nthreads = 0\n")
    with pytest.raises(ValidationError, match="threads"):
        load_config(toml)


def test_an_activation_the_network_cannot_build_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[methodology.algorithms]\nactivation = "swish"\n')
    with pytest.raises(ValidationError, match="activation"):
        load_config(toml)


def test_a_loss_the_network_cannot_use_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[methodology.algorithms]\nloss = "l1"\n')
    with pytest.raises(ValidationError, match="loss"):
        load_config(toml)


def test_the_activation_vocabulary_matches_the_network() -> None:
    assert get_args(NetActivation) == get_args(Activation)


def test_the_loss_vocabulary_matches_the_network() -> None:
    assert get_args(NetLoss) == get_args(Loss)


def test_a_log_level_other_than_info_or_debug_is_rejected(tmp_path: Path) -> None:
    toml = write(tmp_path, '[methodology.algorithms]\nlog_level = "LOUD"\n')
    with pytest.raises(ValidationError, match="log_level"):
        load_config(toml)


def test_the_environment_shortens_the_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPER__METHODOLOGY__ALGORITHMS__MAX_EPOCHS", "2")
    toml = write(tmp_path, "[methodology.algorithms]\nmax_epochs = 500\n")
    assert load_config(toml).methodology.algorithms.max_epochs == 2


def test_the_shipped_paper_toml_sets_the_baseline_network() -> None:
    with PAPER_TOML.open("rb") as handle:
        assert "algorithms" in tomllib.load(handle).get("methodology", {})


def test_every_paper_section_has_its_folder() -> None:
    assert missing_section_folders(PaperConfig, ROOT) == ()
