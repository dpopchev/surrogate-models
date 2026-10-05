"""Facts about reading the paper's choices from a TOML file."""

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from shared.config import ROOT, PaperConfig, load_config, missing_section_folders


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


def test_the_shipped_paper_toml_selects_every_ns_figure() -> None:
    assert len(load_config().data_analysis.neutron_stars.figures) == 7


def test_the_shipped_paper_toml_selects_every_ns_table() -> None:
    assert len(load_config().data_analysis.neutron_stars.tables) == 3


def test_every_paper_section_has_its_folder() -> None:
    assert missing_section_folders(PaperConfig, ROOT) == ()
