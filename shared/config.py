"""The paper's choices: paper.toml read into a typed PaperConfig (E-003).

Each section model is keyed by its chapter folder name without the number prefix
(data_analysis -> 40_data_analysis) and each choice is a Literal scoped to what its module can
draw, so an unknown key or an unsupported value fails at load time. PAPER__<SECTION>__<FIELD>
environment variables override the file.
"""

import tomllib
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, PositiveInt
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from shared.plots import NOTATION_TEX as NOTATION_TEX
from shared.plots import PlotStyle

ROOT = Path(__file__).resolve().parents[1]
PAPER_TOML = ROOT / "paper.toml"
# NOTATION_TEX (shared.plots): the paper's notation macros, input by the preamble and handed
# to matplotlib by the figure writers (W-098).

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
# scatter: every row; lines: one line per curve; lambda_inset: by lambda, a beta inset (W-112).
NsChargeTargetVariant = Literal["scatter", "lines", "lambda_inset"]
# by_lambda or by_beta: the curves coloured by that parameter; two_panels: both (W-112).
NsMassDensityVariant = Literal["by_lambda", "by_beta", "two_panels"]
# histograms: a row per variable, raw and log10; log_counts: the same with log-y row counts;
# transposed: raw and log10 as two rows, a column per variable (W-112). Both sections.
UnivariateVariant = Literal["histograms", "log_counts", "transposed"]
# annotated: a panel per target, the 4 and 5 figures lines annotated; figures_axis: a right
# axis in significant figures instead; one_panel: both targets' p95 in one axes (W-112).
CeilingsVariant = Literal["annotated", "figures_axis", "one_panel"]
# histogram_heatmap: distance to M_max and the M_max grid; heatmap_contours: the grid alone
# with M_max contours; log_histogram: the distance alone, log-y (W-112).
NsMassMaxVariant = Literal["histogram_heatmap", "heatmap_contours", "log_histogram"]


class NeutronStarVariants(BaseModel):
    """Section 3.1: which variant of each NS figure the paper places (W-112)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    charge_target: NsChargeTargetVariant = "scatter"
    mass_density: NsMassDensityVariant = "by_lambda"
    univariate_continuous: UnivariateVariant = "histograms"
    mass_max: NsMassMaxVariant = "histogram_heatmap"
    ceilings: CeilingsVariant = "annotated"


class NeutronStarsSection(BaseModel):
    """Section 3.1: which NS EDA figures and tables eda_neutron_stars.py renders."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    figures: tuple[NsFigure, ...] = get_args(NsFigure)
    tables: tuple[NsTable, ...] = get_args(NsTable)
    variants: NeutronStarVariants = NeutronStarVariants()


BhFigure = Literal["univariate_continuous", "mass_radius", "charge_target", "existence_edge"]
BhTable = Literal["univariate", "charge_correlation", "split_strategies"]
# inset: M against r_h, the departure from GR in an inset; departure: M - r_h/2 against r_h;
# both: the two side by side (W-111).
BhMassRadiusVariant = Literal["inset", "departure", "both"]
# two_panels: M against log10 D and log10 D/M; log_d: log10 D alone (W-112).
BhChargeTargetVariant = Literal["two_panels", "log_d"]
# segments: each curve's r_h range; edge_line: r_h,min against beta; zoomed: the segments
# near the edge (W-112).
BhExistenceEdgeVariant = Literal["segments", "edge_line", "zoomed"]


class BlackHoleVariants(BaseModel):
    """Section 3.2: which variant of each BH figure the paper places (W-111)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mass_radius: BhMassRadiusVariant = "inset"
    univariate_continuous: UnivariateVariant = "histograms"
    charge_target: BhChargeTargetVariant = "two_panels"
    existence_edge: BhExistenceEdgeVariant = "segments"
    ceilings: CeilingsVariant = "annotated"


class BlackHolesSection(BaseModel):
    """Section 3.2: which BH EDA figures and tables eda_black_holes.py renders."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    figures: tuple[BhFigure, ...] = get_args(BhFigure)
    tables: tuple[BhTable, ...] = get_args(BhTable)
    variants: BlackHoleVariants = BlackHoleVariants()


class DataAnalysis(BaseModel):
    """Chapter 40_data_analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    neutron_stars: NeutronStarsSection = NeutronStarsSection()
    black_holes: BlackHolesSection = BlackHolesSection()


# The network vocabulary of shared/surrogate.py, repeated so that loading the config never
# imports torch; test_config.py holds the two equal.
NetActivation = Literal["relu", "gelu", "tanh"]
NetLoss = Literal["mse", "huber"]
# ns_mass: the NS mass parity; residual_strip: the same with its residuals below; all_pairs:
# every dataset and target, 2 x 2 (W-113).
AlgParityVariant = Literal["ns_mass", "residual_strip", "all_pairs"]


class AlgorithmsVariants(BaseModel):
    """Section 4.1: which variant of each figure the paper places (W-113)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parity: AlgParityVariant = "ns_mass"


class AlgorithmsSection(BaseModel):
    """Section 4.1: the baseline network fit_baseline.py trains -- the H2 and H3 control arm."""

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
    variants: AlgorithmsVariants = AlgorithmsVariants()


class Methodology(BaseModel):
    """Chapter 50_methodology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithms: AlgorithmsSection = AlgorithmsSection()


FamiliesFigure = Literal["parity", "errors", "scaling"]
# parity: predicted against true; residual: predicted minus true against true (W-113).
FamiliesParityVariant = Literal["parity", "residual"]
# scatter_profiles: a panel per pair and candidate, scatter and profiles; profiles: the
# profiles only, the candidates overlaid, a panel per pair; hexbin: a density per panel (W-113).
FamiliesErrorsVariant = Literal["scatter_profiles", "profiles", "hexbin"]
# figures_and_seconds: per pair the folds' figures and the fit seconds; marker_size: a panel per
# pair, the seconds as marker size; small_multiples: a panel per family (W-113).
FamiliesScalingVariant = Literal["figures_and_seconds", "marker_size", "small_multiples"]


class FamiliesVariants(BaseModel):
    """Section 5.2: which variant of each family-screen figure the paper places (W-113)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parity: FamiliesParityVariant = "parity"
    errors: FamiliesErrorsVariant = "scatter_profiles"
    scaling: FamiliesScalingVariant = "figures_and_seconds"


class FamiliesSection(BaseModel):
    """Section 5.2: which family-screen figures screen_families.py writes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    figures: tuple[FamiliesFigure, ...] = get_args(FamiliesFigure)
    variants: FamiliesVariants = FamiliesVariants()


# dots_and_best: every trial and the best so far; best_only: the best so far alone; regret: the
# best so far minus each trial, log-scaled (W-113).
PrecisionHistoryVariant = Literal["dots_and_best", "best_only", "regret"]
# row: knots, neighbours and the network side by side; grid: 2 x 2, the knots split by
# dataset; no_network: knots and neighbours alone (W-113).
PrecisionRangesVariant = Literal["row", "grid", "no_network"]


class PrecisionVariants(BaseModel):
    """Section 5.3: which variant of each tuning figure the paper places (W-113)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    history: PrecisionHistoryVariant = "dots_and_best"
    ranges: PrecisionRangesVariant = "row"


class PrecisionSection(BaseModel):
    """Section 5.3: the tuning figures tune_survivors.py writes (W-113)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variants: PrecisionVariants = PrecisionVariants()


class Results(BaseModel):
    """Chapter 60_results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    families: FamiliesSection = FamiliesSection()
    precision: PrecisionSection = PrecisionSection()


class PaperConfig(BaseSettings):
    """Every choice of the paper: the plot style and one model per chapter."""

    model_config = SettingsConfigDict(
        frozen=True, extra="forbid", env_prefix="PAPER__", env_nested_delimiter="__"
    )

    plot: PlotStyle = PlotStyle()
    data_analysis: DataAnalysis = DataAnalysis()
    methodology: Methodology = Methodology()
    results: Results = Results()

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
    """Return the dotted chapter and section keys of model that name no <nn>_<key> folder.

    Only fields whose type is itself a model count; plot is a style, not a chapter, and a
    model inside a section (its variants) is a group of choices, not a folder (W-111).
    """
    missing: list[str] = []
    for name, chapter in _sections(model, skip):
        folders = sorted(root.glob(f"[0-9][0-9]_{name}"))
        if not folders:
            missing.append(name)
            continue
        missing += [
            f"{name}.{key}"
            for key, _ in _sections(chapter, ())
            if not any(folders[0].glob(f"[0-9][0-9]_{key}"))
        ]
    return tuple(missing)


def load_config(toml_file: Path = PAPER_TOML) -> PaperConfig:
    """Read toml_file into a PaperConfig; environment variables override the file."""
    with toml_file.open("rb") as handle:
        return PaperConfig(**tomllib.load(handle))
