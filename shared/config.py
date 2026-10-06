"""The paper's choices: paper.toml read into a typed PaperConfig (E-003).

Each section model is keyed by its chapter folder name without the number prefix
(data_analysis -> 40_data_analysis) and each choice is a Literal scoped to what its module can
draw, so an unknown key or an unsupported value fails at load time. PAPER__<SECTION>__<FIELD>
environment variables override the file.
"""

import tomllib
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from shared.plots import PlotStyle

ROOT = Path(__file__).resolve().parents[1]
PAPER_TOML = ROOT / "paper.toml"

# --- vocabulary and types ---------------------------------------------------------------------

NsFigure = Literal[
    "univariate",
    "charge_target",
    "mass_density",
    "grid_fill",
    "curve_adjacency",
    "univariate_continuous",
    "mass_max",
]
NsTable = Literal["univariate", "charge_correlation", "split_strategies"]


class NeutronStarsSection(BaseModel):
    """Section 4.1: which NS EDA figures and tables eda_neutron_stars.py renders."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    figures: tuple[NsFigure, ...] = get_args(NsFigure)
    tables: tuple[NsTable, ...] = get_args(NsTable)


BhFigure = Literal["univariate_continuous", "mass_radius", "charge_target", "existence_edge"]
BhTable = Literal["univariate", "charge_correlation", "split_strategies"]


class BlackHolesSection(BaseModel):
    """Section 4.2: which BH EDA figures and tables eda_black_holes.py renders."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    figures: tuple[BhFigure, ...] = get_args(BhFigure)
    tables: tuple[BhTable, ...] = get_args(BhTable)


class DataAnalysis(BaseModel):
    """Chapter 40_data_analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    neutron_stars: NeutronStarsSection = NeutronStarsSection()
    black_holes: BlackHolesSection = BlackHolesSection()
    # The floor eps of the charge target log10(max(D, eps)/M) (W-019, E-002 Decision).
    charge_floor: float = 1e-5


# The network vocabulary of shared/surrogate.py, repeated so that loading the config never
# imports torch; test_config.py holds the two equal.
NetActivation = Literal["relu", "gelu", "tanh"]
NetLoss = Literal["mse", "huber"]


class AlgorithmsSection(BaseModel):
    """Section 5.1: the baseline network fit_baseline.py trains -- the H2 and H3 control arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = 64
    depth: int = 3
    activation: NetActivation = "relu"
    loss: NetLoss = "mse"
    lr: float = 1e-3
    max_epochs: int = 500
    batch_size: int = 1024
    patience: int = 20
    valid_fraction: float = 0.2
    seed: int = 20261005
    log_level: Literal["INFO", "DEBUG"] = "INFO"


class Methodology(BaseModel):
    """Chapter 50_methodology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithms: AlgorithmsSection = AlgorithmsSection()


class PaperConfig(BaseSettings):
    """Every choice of the paper: the plot style and one model per chapter."""

    model_config = SettingsConfigDict(
        frozen=True, extra="forbid", env_prefix="PAPER__", env_nested_delimiter="__"
    )

    plot: PlotStyle = PlotStyle()
    data_analysis: DataAnalysis = DataAnalysis()
    methodology: Methodology = Methodology()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let the environment override the file, which load_config passes as init values."""
        return (env_settings, init_settings)


# --- behaviour --------------------------------------------------------------------------------


def _sections(model: type[BaseModel], skip: tuple[str, ...]) -> list[tuple[str, type[BaseModel]]]:
    """The fields of model whose type is itself a model, except those named in skip."""
    return [
        (name, field.annotation)
        for name, field in model.model_fields.items()
        if isinstance(field.annotation, type)
        and issubclass(field.annotation, BaseModel)
        and name not in skip
    ]


def missing_section_folders(
    model: type[BaseModel], root: Path, skip: tuple[str, ...] = ("plot",)
) -> tuple[str, ...]:
    """Return the dotted section keys of model that name no <nn>_<key> folder under root.

    Only fields whose type is itself a model count as sections; plot is a style, not a chapter.
    """
    missing: list[str] = []
    for name, section in _sections(model, skip):
        folders = sorted(root.glob(f"[0-9][0-9]_{name}"))
        if not folders:
            missing.append(name)
            continue
        missing += [f"{name}.{key}" for key in missing_section_folders(section, folders[0], ())]
    return tuple(missing)


def load_config(toml_file: Path = PAPER_TOML) -> PaperConfig:
    """Read toml_file into a PaperConfig; environment variables override the file."""
    with toml_file.open("rb") as handle:
        return PaperConfig(**tomllib.load(handle))
