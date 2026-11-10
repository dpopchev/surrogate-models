"""The paper's choices: paper.toml read into a typed PaperConfig (E-003).

The paper-wide choices only: the plot style, the baseline network and the table layouts. Each
section's own picks live in its toml beside its script, read with load_section (W-136). Each
choice is a Literal scoped to what its module can draw, so an unknown key or an unsupported value
fails at load time. PAPER__<SECTION>__<FIELD> environment variables override the file.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from paperlib.plots import NOTATION_TEX as NOTATION_TEX
from paperlib.plots import PlotStyle

ROOT = Path(__file__).resolve().parents[1]
PAPER_TOML = ROOT / "paper.toml"
# NOTATION_TEX (paperlib.plots): the paper's notation macros, input by the preamble and handed
# to matplotlib by the figure writers (W-098).

# --- vocabulary and types ---------------------------------------------------------------------

# histograms: a row per variable, raw and log10; log_counts: the same with log-y row counts;
# transposed: raw and log10 as two rows, a column per variable (W-112). Both sections.
UnivariateVariant = Literal["histograms", "log_counts", "transposed"]
# annotated: a panel per target, the 4 and 5 figures lines annotated; figures_axis: a right
# axis in significant figures instead; one_panel: both targets' p95 in one axes (W-112).
CeilingsVariant = Literal["annotated", "figures_axis", "one_panel"]

# The network vocabulary of paperlib/surrogate.py, repeated so that loading the config never
# imports torch; test_config.py holds the two equal.
NetActivation = Literal["relu", "gelu", "tanh"]
NetLoss = Literal["mse", "huber"]


class NetworkSettings(BaseModel):
    """The baseline network, paper-wide: fit_baseline.py trains it, the screens read it."""

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
    # torch threads per fit: 1 is fastest for this small network on this CPU (W-036 Finding).
    threads: PositiveInt = 1
    # one-thread worker processes running the fits at once: 6 gives 3.2x the fits per hour of 1
    # (W-072 T-148); 1 for a run whose fit seconds are reported, since sharing slows each fit.
    workers: PositiveInt = 6


# small: booktabs in the small font; aligned: siunitx S columns aligning the decimals (W-114).
TableLayout = Literal["small", "aligned"]
RepresentationTableLayout = Literal["eight_columns", "split", "transposed"]
FamiliesTableLayout = Literal["subscripts", "rows_column"]
TuningTableLayout = Literal["screen_tuned_test", "no_screen", "delta"]
BaselineTableLayout = Literal["one_table", "two_tables"]
DecisionsTableLayout = Literal["wide", "narrow"]


class Tables(BaseModel):
    """The layout of every table and of each results table the paper places (W-114)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout: TableLayout = "small"
    representation: RepresentationTableLayout = "eight_columns"
    families: FamiliesTableLayout = "subscripts"
    tuning: TuningTableLayout = "screen_tuned_test"
    baseline: BaselineTableLayout = "one_table"
    decisions: DecisionsTableLayout = "wide"


class PaperConfig(BaseSettings):
    """Every paper-wide choice: the plot style, the baseline network and the table layouts."""

    model_config = SettingsConfigDict(
        frozen=True, extra="forbid", env_prefix="PAPER__", env_nested_delimiter="__"
    )

    plot: PlotStyle = PlotStyle()
    network: NetworkSettings = NetworkSettings()
    tables: Tables = Tables()

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


def load_config(toml_file: Path = PAPER_TOML) -> PaperConfig:
    """Read toml_file into a PaperConfig; environment variables override the file."""
    with toml_file.open("rb") as handle:
        return PaperConfig(**tomllib.load(handle))


def load_section[M: BaseModel](model: type[M], toml_file: Path) -> M:
    """Read one section's toml_file into model, a section's own settings (W-136)."""
    with toml_file.open("rb") as handle:
        return model(**tomllib.load(handle))
