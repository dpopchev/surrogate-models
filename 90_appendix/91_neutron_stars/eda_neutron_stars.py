"""Exploratory analysis of the prepared NS table behind Section 3.1 (W-017).

The computations are pure functions over the table, each returning a frozen record; their
numbers become \\nsEda... LaTeX macros so the text and the decisions table cite generated values.
The evidence they carry: which columns want log10, whether log10(D/M) is a better charge target
than log10 D, whether beta matters at fixed lambda, how full the (beta, lambda) grid is, and
whether rows of one curve are each other's nearest neighbours (why the split groups by curve).
The dataset-agnostic machinery lives in paperlib/eda.py.
"""

import argparse
import json
import logging
import pickle
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, assert_never, cast, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from paperlib.ceilings import (
    Ceilings,
    ceiling_figure,
    ceiling_macros,
    ceiling_object,
    ceilings_caption,
    decade_object,
    measure,
    uncertainty_macros,
)
from paperlib.config import (
    NOTATION_TEX,
    CeilingsVariant,
    PaperConfig,
    TableLayout,
    UnivariateVariant,
    load_config,
    load_section,
)
from paperlib.eda import (
    Adjacency,
    ColumnSummary,
    RowsPerCurve,
    camel,
    curve_adjacency,
    curve_ids,
    grid_bins,
    group_splitter,
    holdout_mae,
    kfold_mae,
    make_curve_space,
    neighbour_distances,
    orders_of_magnitude,
    rim_mask,
    row_splitter,
    sci_tex,
    summarize,
    text_width_size,
    univariate_caption,
    univariate_figure,
    with_charge_targets,
)
from paperlib.plots import (
    SYMBOLS,
    PlotStyle,
    anchor_color,
    apply_style,
    colormap,
    line_width,
    mark_alpha,
    mark_size,
)
from paperlib.tables import Plain, Table, Text, aligns, fixed, number, render, to_json

logger = logging.getLogger(__name__)

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


SECTION_TOML = Path(__file__).with_name("91_neutron_stars.toml")

CURVE = ("beta", "lambda")
# The probe space: beta and lambda as they are, the central density in log10.
NS_SPACE = make_curve_space({"beta": "raw", "lambda": "raw", "rho_c": "log10"}, CURVE)


@dataclass(frozen=True)
class ChargeTarget:
    """How one candidate charge target correlates with M and with the two free parameters."""

    target: str
    pearson_m: float
    spearman_m: float
    pearson_lambda: float
    pearson_beta: float


@dataclass(frozen=True)
class GridFill:
    """How many cells of the (beta, lambda) product grid hold a curve."""

    n_beta: int
    n_lambda: int
    curves: int

    @property
    def cells(self) -> int:
        """The cells of the product grid of the observed beta and lambda values."""
        return self.n_beta * self.n_lambda

    @property
    def fill(self) -> float:
        """The share of cells holding a curve."""
        return self.curves / self.cells


SplitStrategy = Literal["random_rows", "curves", "beta_lines", "lambda_lines", "rim"]


@dataclass(frozen=True)
class SplitScore:
    """1-nearest-neighbour error of a target when the split holds out what the strategy names.

    random_rows: rows at random; curves: whole (beta, lambda) curves; beta_lines and
    lambda_lines: every curve of a held-out beta or lambda value; rim: the rim curves, predicted
    from the interior ones (extrapolation). NaN when the strategy has nothing to train on.
    """

    strategy: SplitStrategy
    target: str
    mae: float


# The data limits (W-064): macro tags and TeX labels of the targets and of the curve keys varied.
CEILING_TAGS = {"M": "M", "D": "D", "beta": "Beta", "lambda": "Lambda"}
CEILING_LABELS = SYMBOLS.of("M", "D", "beta", "lambda")
CEILINGS_CAPTION = (
    "Neutron stars: the relative error the data allow, at the 95th percentile -- the noise "
    "along a curve (median), a cubic spline through every other row of a curve (along), and a "
    "curve predicted from the two curves on each side in $\\beta$ or in $\\lambda$ (across); "
    "figures is $-\\log_{10}$ of the worse across-curve error."
)
DECADES_CAPTION = (
    "Neutron stars: median relative error of $\\Dch$ predicted from the neighbouring curves, "
    "per decade of $\\Dch$, and the significant figures the worse direction allows."
)
# The ceilings figure's caption: the dataset, the curve keys varied across, the coordinate along.
CEILINGS_FIGURE = ("Neutron stars", "$\\beta$ or in $\\lambda$", "$\\log_{10}\\rhoc$")


# --- pure functions ---------------------------------------------------------------------------


def charge_targets(table: pd.DataFrame) -> tuple[ChargeTarget, ...]:
    """Compare log10_D and log10_D_over_M as charge targets."""
    return tuple(
        ChargeTarget(
            target,
            float(table[target].corr(table["M"])),
            float(table[target].corr(table["M"], method="spearman")),
            float(table[target].corr(table["lambda"])),
            float(table[target].corr(table["beta"])),
        )
        for target in ("log10_D", "log10_D_over_M")
    )


def _beta_share(group: pd.DataFrame, target: str) -> float:
    """The share of target variance at one lambda explained by the beta group means."""
    total = float(((group[target] - group[target].mean()) ** 2).sum())
    means = group.groupby("beta")[target].transform("mean")
    return float(((means - group[target].mean()) ** 2).sum()) / total if total > 0 else np.nan


def beta_share_at_fixed_lambda(table: pd.DataFrame, target: str) -> float:
    """Median over lambda of the share of target variance explained by beta at that lambda."""
    shares = [_beta_share(group, target) for _, group in table.groupby("lambda")]
    return float(np.nanmedian(shares))


def grid_fill(table: pd.DataFrame) -> GridFill:
    """Count the curves on the product grid of the observed beta and lambda values."""
    return GridFill(
        table["beta"].nunique(), table["lambda"].nunique(), table.groupby(list(CURVE)).ngroups
    )


def _score(
    table: pd.DataFrame, target: str, strategy: SplitStrategy, folds: int, seed: int
) -> float:
    """The 1-NN error of one strategy."""
    match strategy:
        case "random_rows":
            return kfold_mae(table, NS_SPACE, target, row_splitter(folds, seed), None)
        case "curves":
            groups = curve_ids(table, NS_SPACE)
            return kfold_mae(table, NS_SPACE, target, group_splitter(folds, seed), groups)
        case "beta_lines":
            groups = table["beta"].to_numpy()
            return kfold_mae(table, NS_SPACE, target, group_splitter(folds, seed), groups)
        case "lambda_lines":
            groups = table["lambda"].to_numpy()
            return kfold_mae(table, NS_SPACE, target, group_splitter(folds, seed), groups)
        case "rim":
            rim = rim_mask(table, "beta", "lambda").to_numpy()
            return holdout_mae(table, NS_SPACE, target, rim)
        case _:
            assert_never(strategy)


def split_strategies(
    table: pd.DataFrame, target: str, folds: int, seed: int
) -> tuple[SplitScore, ...]:
    """1-NN error of target under each split strategy, in SplitStrategy order."""
    return tuple(
        SplitScore(strategy, target, _score(table, target, strategy, folds, seed))
        for strategy in get_args(SplitStrategy)
    )


def pile_up_share(table: pd.DataFrame, window: float) -> float:
    """The share of rows whose M lies within window of their own curve's maximum M."""
    peak = table.groupby(list(CURVE))["M"].transform("max")
    return float(((peak - table["M"]) < window).mean())


SUMMARY_COLUMNS = ("rho_c", "M", "D", "D_over_M")
SPLIT_TARGETS = ("log10_D_over_M", "M")


@dataclass(frozen=True)
class Evidence:
    """Every computed result Section 3.1 cites, computed once for the macros and the tables."""

    rows: int
    density_grid_values: int
    rows_per_curve: RowsPerCurve
    folds: int
    window: float
    mass_max: float
    pile_up: float
    summaries: tuple[ColumnSummary, ...]
    targets: tuple[ChargeTarget, ...]
    beta_share: float
    fill: GridFill
    adjacency: Adjacency
    splits: tuple[SplitScore, ...]


def evidence(table: pd.DataFrame, folds: int, seed: int, window: float) -> Evidence:
    """Compute every Section 3.1 result from a table with the charge targets."""
    sizes = table.groupby(list(CURVE)).size()
    return Evidence(
        len(table),
        table["rho_c"].nunique(),
        RowsPerCurve(int(sizes.min()), int(sizes.max())),
        folds,
        window,
        float(table["M"].max()),
        pile_up_share(table, window),
        summarize(table, SUMMARY_COLUMNS),
        charge_targets(table),
        beta_share_at_fixed_lambda(table, "log10_D_over_M"),
        grid_fill(table),
        curve_adjacency(table, NS_SPACE),
        tuple(s for t in SPLIT_TARGETS for s in split_strategies(table, t, folds, seed)),
    )


def raw_numbers(raw: Mapping[str, int]) -> dict[str, str]:
    """The raw file's counts the preparation kept with the table (its runs, the empty ones,
    its rows), as \\nsEda macros for Section 2.4."""
    return {
        "nsEdaRawRuns": str(raw["runs"]),
        "nsEdaEmptyRuns": str(raw["empty_runs"]),
        "nsEdaRawRows": str(raw["rows"]),
    }


def numbers(found: Evidence) -> dict[str, str]:
    """Every number Section 3.1 cites, keyed by its \\nsEda macro name."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    log_d, log_dm = found.targets
    rho = summary["rho_c", "log10"]
    d = summary["D", "raw"]
    return {
        "nsEdaRows": f"{found.rows}",
        "nsEdaCurves": f"{found.fill.curves}",
        "nsEdaGridCells": f"{found.fill.cells}",
        "nsEdaGridFillPercent": f"{100 * found.fill.fill:.1f}",
        "nsEdaRhocLogMean": f"{rho.mean:.3f}",
        "nsEdaRhocLogMedian": f"{rho.median:.3f}",
        "nsEdaRhocLogStd": f"{rho.std:.3f}",
        "nsEdaRhocRawSkew": f"{summary['rho_c', 'raw'].skew:.2f}",
        "nsEdaRhocLogSkew": f"{rho.skew:.2f}",
        "nsEdaRhocOrders": f"{orders_of_magnitude(summary['rho_c', 'raw']):.1f}",
        "nsEdaDensityGridValues": f"{found.density_grid_values}",
        "nsEdaRowsPerCurveMin": f"{found.rows_per_curve.minimum}",
        "nsEdaRowsPerCurveMax": f"{found.rows_per_curve.maximum}",
        "nsEdaDMin": sci_tex(d.minimum),
        "nsEdaDMax": f"{d.maximum:.3f}",
        "nsEdaDOrders": f"{orders_of_magnitude(d):.1f}",
        "nsEdaCorrMLogD": f"{log_d.spearman_m:.2f}",
        "nsEdaCorrMLogDM": f"{log_dm.spearman_m:.2f}",
        "nsEdaCorrLambdaLogD": f"{log_d.pearson_lambda:.2f}",
        "nsEdaCorrLambdaLogDM": f"{log_dm.pearson_lambda:.2f}",
        "nsEdaPearsonMLogD": f"{log_d.pearson_m:.2f}",
        "nsEdaPearsonMLogDM": f"{log_dm.pearson_m:.2f}",
        "nsEdaCorrBetaLogD": f"{log_d.pearson_beta:.2f}",
        "nsEdaCorrBetaLogDM": f"{log_dm.pearson_beta:.2f}",
        # r squared: the share of the target's variance a straight line in lambda explains.
        "nsEdaLambdaLinearPercentLogD": f"{100 * log_d.pearson_lambda**2:.0f}",
        "nsEdaLambdaLinearPercentLogDM": f"{100 * log_dm.pearson_lambda**2:.0f}",
        "nsEdaBetaSharePercent": f"{100 * found.beta_share:.0f}",
        "nsEdaSameCurvePercent": f"{100 * found.adjacency.same_curve_share:.1f}",
        "nsEdaWithinMedian": f"{found.adjacency.within_median:.3f}",
        "nsEdaAcrossMedian": f"{found.adjacency.across_median:.3f}",
        "nsEdaFolds": f"{found.folds}",
        "nsEdaMassMax": f"{found.mass_max:.2f}",
        "nsEdaPileUpWindow": f"{found.window:.2f}",
        "nsEdaPileUpPercent": f"{100 * found.pile_up:.0f}",
    } | {
        f"nsEdaSplit{camel(s.strategy)}{TARGET_TAGS[s.target]}": _split_error(s)
        for s in found.splits
    }


TABLE_LABELS = SYMBOLS.of("rho_c", "M", "D", "D_over_M")
STRATEGY_LABELS: dict[SplitStrategy, str] = {
    "random_rows": "random rows",
    "curves": "$(\\beta, \\lambda)$ curves (GroupKFold)",
    "beta_lines": "whole $\\beta$ lines",
    "lambda_lines": "whole $\\lambda$ lines",
    "rim": "rim curves, from the interior",
}


def _univariate_table(found: Evidence) -> Table:
    """Table A: range, orders of magnitude and skew, raw against log10."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    rows = tuple(
        (
            Text(TABLE_LABELS[c]),
            number(summary[c, "raw"].minimum),
            number(summary[c, "raw"].maximum),
            fixed(orders_of_magnitude(summary[c, "raw"]), 1),
            fixed(summary[c, "raw"].skew, 2),
            fixed(summary[c, "log10"].skew, 2),
        )
        for c in SUMMARY_COLUMNS
    )
    header = tuple(
        Text(h) for h in ("Variable", "min", "max", "orders", "skew", "skew of $\\log_{10}$")
    )
    caption = (
        "Neutron stars: range, orders of magnitude spanned (orders) and skewness of each "
        "continuous variable, raw and in $\\log_{10}$."
    )
    return Table("ns-univariate", caption, aligns("lrrrrr"), (header,), rows)


def _charge_table(found: Evidence) -> Table:
    """Table B: how each charge target correlates with M and the free parameters."""
    rows = tuple(
        (
            Text(CHARGE_LABELS[t.target]),
            fixed(t.pearson_m, 2),
            fixed(t.spearman_m, 2),
            fixed(t.pearson_lambda, 2),
            fixed(t.pearson_beta, 2),
        )
        for t in found.targets
    )
    header = tuple(
        Text(h)
        for h in (
            "Target",
            "Pearson $M$",
            "Spearman $M$",
            "Pearson $\\lambda$",
            "Pearson $\\beta$",
        )
    )
    caption = "Neutron stars: correlation of the two candidate charge targets."
    return Table("ns-charge-correlation", caption, aligns("lrrrr"), (header,), rows)


def _split_table(found: Evidence) -> Table:
    """Table C: 1-NN error per split strategy and target."""
    error = {(s.strategy, s.target): _split_error(s) for s in found.splits}
    rows = tuple(
        (
            Text(STRATEGY_LABELS[strategy]),
            Plain(error[strategy, "log10_D_over_M"]),
            Plain(error[strategy, "M"]),
        )
        for strategy in get_args(SplitStrategy)
    )
    header = tuple(Text(h) for h in ("Held out", "$\\log_{10}(D/M)$", "$M$"))
    caption = (
        f"Neutron stars: 1-nearest-neighbour mean absolute error under each split "
        f"({found.folds} folds; the rim is predicted from the interior)."
    )
    return Table("ns-split-strategies", caption, aligns("lrr"), (header,), rows)


def table_object(found: Evidence, table: NsTable) -> Table:
    """One table as an object, to store as JSON and to render under any layout."""
    match table:
        case "univariate":
            return _univariate_table(found)
        case "charge_correlation":
            return _charge_table(found)
        case "split_strategies":
            return _split_table(found)
        case _:
            assert_never(table)


def table_tex(found: Evidence, table: NsTable, layout: TableLayout = "small") -> str:
    """The booktabs table environment of one table in the paper's table layout."""
    return render(table_object(found, table), layout)


def table_objects(
    found: Evidence, names: tuple[NsTable, ...], bounds: Ceilings
) -> dict[str, Table]:
    """The section's tables by stored name: the selected ones and the two ceilings tables."""
    return {name: table_object(found, name) for name in names} | {
        "ceilings": ceiling_object("ns-ceilings", CEILINGS_CAPTION, bounds, CEILING_LABELS),
        "ceiling_decades": decade_object(
            "ns-ceiling-decades", DECADES_CAPTION, bounds, CEILING_LABELS
        ),
    }


TARGET_TAGS = {"M": "M", "log10_D_over_M": "DM"}
# The mass errors are an order of magnitude smaller, so they keep one more decimal.
SPLIT_DECIMALS = {"M": 4, "log10_D_over_M": 3}


def _split_error(score: SplitScore) -> str:
    """One split error with its target's decimals, as in the text and Table C alike."""
    return f"{score.mae:.{SPLIT_DECIMALS[score.target]}f}"


# The section's asset folder, and the prefix that keeps its basenames unique in the flat build.
SECTION = "91_neutron_stars"
DEFAULT_VARIANTS = NeutronStarVariants()


def asset(figure: NsFigure) -> str:
    """The stem one figure's image and its .tex wrapper share."""
    return f"{SECTION}_fig_{figure}"


def picked_stem(figure: NsFigure, variants: NeutronStarVariants = DEFAULT_VARIANTS) -> str:
    """The stem of the file the paper includes: the picked variant's PNG under build/figures/,
    or the figure's own stem when it has no variants."""
    variant = getattr(variants, figure, None)
    return asset(figure) if variant is None else f"{asset(figure)}_{variant}"


def _charge_target_caption(variant: NsChargeTargetVariant) -> str:
    """The caption of the charge-target figure in one variant."""
    match variant:
        case "scatter":
            return (
                "Neutron stars: $M$ against $\\log_{10}\\Dch$ (left) and $\\log_{10}(\\Dch/M)$ "
                "(right), coloured by $\\lambda$ (top) and $\\beta$ (bottom)."
            )
        case "lines":
            return (
                "Neutron stars: $M$ against $\\log_{10}\\Dch$ (left) and $\\log_{10}(\\Dch/M)$ "
                "(right), one line per $(\\beta, \\lambda)$ curve coloured by $\\lambda$ (top) "
                "and $\\beta$ (bottom)."
            )
        case "lambda_inset":
            return (
                "Neutron stars: $M$ against $\\log_{10}\\Dch$ (left) and $\\log_{10}(\\Dch/M)$ "
                "(right), coloured by $\\lambda$; the inset colours $\\log_{10}(\\Dch/M)$ by "
                "$\\beta$."
            )
        case _:
            assert_never(variant)


def _mass_density_caption(variant: NsMassDensityVariant) -> str:
    """The caption of the mass-density figure in one variant."""
    curves = "one line per $(\\beta, \\lambda)$ curve"
    ends = "dots mark each curve's last (maximum-mass) point."
    match variant:
        case "by_lambda":
            return (
                f"Neutron stars: $M$ against $\\log_{{10}}\\rhoc$, {curves} coloured by "
                f"$\\lambda$; {ends}"
            )
        case "by_beta":
            return (
                f"Neutron stars: $M$ against $\\log_{{10}}\\rhoc$, {curves} coloured by "
                f"$\\beta$; {ends}"
            )
        case "two_panels":
            return (
                f"Neutron stars: $M$ against $\\log_{{10}}\\rhoc$, {curves} coloured by "
                f"$\\lambda$ (left) and $\\beta$ (right); {ends}"
            )
        case _:
            assert_never(variant)


def _mass_max_caption(variant: NsMassMaxVariant) -> str:
    """The caption of the mass-max figure in one variant."""
    distance = "distance of each row to its own curve's maximum mass, the pile-up window dashed"
    grid = "each curve's maximum mass over the $(\\beta, \\lambda)$ grid"
    match variant:
        case "histogram_heatmap":
            return f"Neutron stars: {distance} (left), and {grid} (right)."
        case "heatmap_contours":
            return f"Neutron stars: {grid}, its contours labelled in $M_\\odot$."
        case "log_histogram":
            return f"Neutron stars: {distance}, the row counts on a log scale."
        case _:
            assert_never(variant)


def caption(figure: NsFigure, variants: NeutronStarVariants = DEFAULT_VARIANTS) -> str:
    """The caption of one figure in its picked variant."""
    match figure:
        case "univariate":
            return (
                "Neutron stars: distributions of each variable, raw (left) and "
                "$\\log_{10}$ (right)."
            )
        case "charge_target":
            return _charge_target_caption(variants.charge_target)
        case "mass_density":
            return _mass_density_caption(variants.mass_density)
        case "grid_fill":
            return (
                "Neutron stars: rows per $(\\beta, \\lambda)$ curve; blank cells hold no converged "
                "curve."
            )
        case "curve_adjacency":
            return (
                "Neutron stars: distance from each row to its nearest neighbour in its own curve "
                "and in another curve, in standardized $(\\beta, \\lambda, \\log_{10}\\rhoc)$."
            )
        case "univariate_continuous":
            return univariate_caption(
                "Neutron stars",
                variants.univariate_continuous,
                ", one bar per value where the variable sits on a grid",
            )
        case "mass_max":
            return _mass_max_caption(variants.mass_max)
        case _:
            assert_never(figure)


def figure_tex(figure: NsFigure, variants: NeutronStarVariants = DEFAULT_VARIANTS) -> str:
    """The figure environment of one figure in its picked variant, placed here or at the top
    of a page (!htb)."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{picked_stem(figure, variants)}}}\n"
        f"\\caption{{{caption(figure, variants)}}}\n"
        f"\\label{{fig:ns-{figure.replace('_', '-')}}}\n"
        "\\end{figure}\n"
    )


LABELS = SYMBOLS.of("beta", "lambda", "rho_c", "M", "D", "D_over_M")


def _univariate(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Histograms of each variable, raw and in log10."""
    fig, axes = plt.subplots(len(LABELS), 2, figsize=text_width_size(1.6), layout="constrained")
    color = anchor_color(style, "neutron_stars")
    for row, (column, label) in enumerate(LABELS.items()):
        raw = table[column].to_numpy(dtype=float)
        limit = style.univariate_grid_values
        axes[row, 0].hist(raw, bins=grid_bins(raw, limit), color=color)
        axes[row, 1].hist(np.log10(raw), bins=grid_bins(np.log10(raw), limit), color=color)
        axes[row, 0].set_xlabel(label)
        axes[row, 1].set_xlabel(f"$\\log_{{10}}$ {label}")
    return fig


def _univariate_continuous(
    table: pd.DataFrame, style: PlotStyle, variant: UnivariateVariant
) -> Figure:
    """Histograms of the continuous variables, raw and in log10, grid-aware bins: rho_c sits on
    a shared grid of a few hundred values, so one bar per value avoids aliasing."""
    color = anchor_color(style, "neutron_stars")
    return univariate_figure(table, SUMMARY_COLUMNS, LABELS, color, style.grid_values, variant)


def _distance_to_peak(ax: Axes, table: pd.DataFrame, style: PlotStyle, window: float) -> None:
    """Rows' distance to their curve's M_max on ax, the pile-up window dashed."""
    peak = table.groupby(list(CURVE))["M"].transform("max")
    ax.hist(peak - table["M"], bins=40, color=anchor_color(style, "neutron_stars"))
    ax.axvline(window, color="black", linestyle="--", linewidth=line_width(style, 0.8))
    ax.set_xlabel(SYMBOLS["M_max_minus_M"])
    ax.set_ylabel("rows")


def _peak_grid(
    fig: Figure, ax: Axes, table: pd.DataFrame, style: PlotStyle, contours: bool
) -> None:
    """Each curve's M_max over the (beta, lambda) grid on ax, with labelled contours if asked."""
    maxima = table.groupby(list(CURVE))["M"].max().unstack("beta")
    mesh = ax.pcolormesh(
        maxima.columns,
        maxima.index,
        maxima.to_numpy(),
        cmap=colormap(style.neutron_stars.lambda_cmap),
        shading="nearest",
    )
    if contours:
        lines = ax.contour(
            maxima.columns,
            maxima.index,
            maxima.to_numpy(),
            colors="white",
            linewidths=line_width(style, 0.6),
        )
        ax.clabel(lines, fontsize="small", fmt="%.2f")
    ax.set_xlabel(LABELS["beta"])
    ax.set_ylabel(LABELS["lambda"])
    fig.colorbar(mesh, ax=ax, label=SYMBOLS["M_max"])


def _mass_max(
    table: pd.DataFrame, style: PlotStyle, window: float, variant: NsMassMaxVariant
) -> Figure:
    """Rows' distance to their curve's M_max beside each curve's M_max over the grid
    (histogram_heatmap), the grid alone with M_max contours (heatmap_contours), or the
    distance alone on log-y counts (log_histogram)."""
    match variant:
        case "histogram_heatmap":
            fig, (left, right) = plt.subplots(
                1, 2, figsize=text_width_size(0.45), layout="constrained"
            )
            _distance_to_peak(left, table, style, window)
            _peak_grid(fig, right, table, style, contours=False)
        case "heatmap_contours":
            fig, ax = plt.subplots(figsize=text_width_size(0.62), layout="constrained")
            _peak_grid(fig, ax, table, style, contours=True)
        case "log_histogram":
            fig, ax = plt.subplots(figsize=text_width_size(0.45), layout="constrained")
            _distance_to_peak(ax, table, style, window)
            ax.set_yscale("log")
        case _:
            assert_never(variant)
    return fig


CHARGE_LABELS = SYMBOLS.of("log10_D", "log10_D_over_M")


def _charge_row(
    fig: Figure,
    axes: np.ndarray,
    table: pd.DataFrame,
    style: PlotStyle,
    column: str,
    cmap: Colormap,
) -> None:
    """One row of the charge-target figure: both targets against M, coloured by column."""
    scatters = [
        ax.scatter(
            table["M"],
            table[target],
            c=table[column],
            cmap=cmap,
            s=mark_size(style, 0.2),
            linewidths=0,
            rasterized=True,
        )
        for ax, target in zip(axes, CHARGE_LABELS, strict=True)
    ]
    for ax, label in zip(axes, CHARGE_LABELS.values(), strict=True):
        ax.set_ylabel(label)
    fig.colorbar(scatters[-1], ax=list(axes), label=LABELS[column])


def _curve_collection(
    table: pd.DataFrame, style: PlotStyle, x: str, y: str, column: str, cmap: Colormap
) -> LineCollection:
    """One line per (beta, lambda) curve of y against x, coloured by column."""
    groups = [g for _, g in table.groupby(list(CURVE))]
    lines = LineCollection(
        [np.column_stack((g[x], g[y])) for g in groups],
        array=np.array([g[column].iloc[0] for g in groups]),
        cmap=cmap,
        linewidths=line_width(style, 0.3),
    )
    lines.set_rasterized(True)
    return lines


def _charge_lines_row(
    fig: Figure,
    axes: np.ndarray,
    table: pd.DataFrame,
    style: PlotStyle,
    column: str,
    cmap: Colormap,
) -> None:
    """One row of the charge-target figure as one line per curve, coloured by column."""
    collections = []
    for ax, (target, label) in zip(axes, CHARGE_LABELS.items(), strict=True):
        collections.append(_curve_collection(table, style, "M", target, column, cmap))
        ax.add_collection(collections[-1])
        ax.autoscale()
        ax.set_ylabel(label)
    fig.colorbar(collections[-1], ax=list(axes), label=LABELS[column])


def _charge_target(table: pd.DataFrame, style: PlotStyle, variant: NsChargeTargetVariant) -> Figure:
    """M against log10 D and log10 D/M, coloured by lambda and by beta: every row (scatter),
    one line per curve (lines), or by lambda with a beta inset (lambda_inset)."""
    lam, beta = colormap(style.neutron_stars.lambda_cmap), colormap(style.neutron_stars.beta_cmap)
    match variant:
        case "scatter":
            fig, axes = plt.subplots(
                2, 2, figsize=text_width_size(0.9), sharex=True, layout="constrained"
            )
            _charge_row(fig, axes[0, :], table, style, "lambda", lam)
            _charge_row(fig, axes[1, :], table, style, "beta", beta)
        case "lines":
            fig, axes = plt.subplots(
                2, 2, figsize=text_width_size(0.9), sharex=True, layout="constrained"
            )
            _charge_lines_row(fig, axes[0, :], table, style, "lambda", lam)
            _charge_lines_row(fig, axes[1, :], table, style, "beta", beta)
        case "lambda_inset":
            fig, axes = plt.subplots(
                1, 2, figsize=text_width_size(0.45), squeeze=False, layout="constrained"
            )
            _charge_row(fig, axes[0, :], table, style, "lambda", lam)
            inset = axes[0, 1].inset_axes((0.58, 0.06, 0.38, 0.38))
            inset.scatter(
                table["M"],
                table["log10_D_over_M"],
                c=table["beta"],
                cmap=beta,
                s=mark_size(style, 0.1),
                linewidths=0,
                rasterized=True,
            )
            # A miniature of the panel for the beta pattern; its axes are the panel's, unlabelled.
            inset.set_title(f"by {LABELS['beta']}", fontsize="small")
            inset.set_xticks([])
            inset.set_yticks([])
        case _:
            assert_never(variant)
    for ax in axes[-1, :]:
        ax.set_xlabel(LABELS["M"])
    return fig


def _density_panel(
    fig: Figure, ax: Axes, table: pd.DataFrame, style: PlotStyle, column: str, cmap: Colormap
) -> None:
    """M against log10 rho_c on ax, one line per curve coloured by column, last points marked."""
    groups = [g for _, g in table.groupby(list(CURVE))]
    lines = LineCollection(
        [np.column_stack((np.log10(g["rho_c"]), g["M"])) for g in groups],
        array=np.array([g[column].iloc[0] for g in groups]),
        cmap=cmap,
        linewidths=line_width(style, 0.3),
    )
    lines.set_rasterized(True)
    ax.add_collection(lines)
    ends = pd.DataFrame([g.iloc[-1] for g in groups])
    ax.scatter(
        np.log10(ends["rho_c"]),
        ends["M"],
        c=ends[column],
        cmap=cmap,
        s=mark_size(style, 1.5),
        zorder=3,
    )
    ax.autoscale()
    ax.set_xlabel(SYMBOLS["log10_rho_c"])
    ax.set_ylabel(LABELS["M"])
    fig.colorbar(lines, ax=ax, label=LABELS[column])


def _mass_density(table: pd.DataFrame, style: PlotStyle, variant: NsMassDensityVariant) -> Figure:
    """M against log10 rho_c, one line per curve with its last point marked, coloured by
    lambda (by_lambda), by beta (by_beta), or the two side by side (two_panels)."""
    lam, beta = colormap(style.neutron_stars.lambda_cmap), colormap(style.neutron_stars.beta_cmap)
    match variant:
        case "by_lambda":
            fig, ax = plt.subplots(figsize=text_width_size(0.62), layout="constrained")
            _density_panel(fig, ax, table, style, "lambda", lam)
        case "by_beta":
            fig, ax = plt.subplots(figsize=text_width_size(0.62), layout="constrained")
            _density_panel(fig, ax, table, style, "beta", beta)
        case "two_panels":
            fig, (left, right) = plt.subplots(
                1, 2, figsize=text_width_size(0.45), sharey=True, layout="constrained"
            )
            _density_panel(fig, left, table, style, "lambda", lam)
            _density_panel(fig, right, table, style, "beta", beta)
        case _:
            assert_never(variant)
    return fig


def _grid_fill(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Rows per (beta, lambda) curve on the product grid; empty cells stay blank."""
    fig, ax = plt.subplots(figsize=text_width_size(0.62), layout="constrained")
    counts = table.groupby(list(CURVE)).size().unstack("beta")
    cmap = colormap(style.neutron_stars.beta_cmap)
    mesh = ax.pcolormesh(
        counts.columns, counts.index, counts.to_numpy(), cmap=cmap, shading="nearest"
    )
    ax.set_xlabel(LABELS["beta"])
    ax.set_ylabel(LABELS["lambda"])
    fig.colorbar(mesh, ax=ax, label="rows per curve")
    return fig


def _curve_adjacency(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Histograms of the nearest within-curve and across-curve distances."""
    fig, ax = plt.subplots(figsize=text_width_size(0.5), layout="constrained")
    found = neighbour_distances(table, NS_SPACE)
    cmap = colormap(style.neutron_stars.beta_cmap)
    edges = np.geomspace(
        min(found.within.min(), found.across.min()), max(found.within.max(), found.across.max()), 50
    )
    alpha = mark_alpha(style, 0.8)
    ax.hist(found.within, bins=edges, color=cmap(0.8), alpha=alpha, label="own curve")
    ax.hist(found.across, bins=edges, color=cmap(0.35), alpha=alpha, label="other curve")
    ax.set_xscale("log")
    ax.set_xlabel("nearest-neighbour distance (standardized)")
    ax.set_ylabel("rows")
    ax.legend()
    return fig


def draw(
    figure: NsFigure,
    table: pd.DataFrame,
    style: PlotStyle,
    window: float,
    variants: NeutronStarVariants = DEFAULT_VARIANTS,
) -> Figure:
    """Draw one NS EDA figure from a table with the charge targets in its picked variant;
    window is the pile-up window the mass-max figure marks."""
    match figure:
        case "univariate":
            return _univariate(table, style)
        case "charge_target":
            return _charge_target(table, style, variants.charge_target)
        case "mass_density":
            return _mass_density(table, style, variants.mass_density)
        case "grid_fill":
            return _grid_fill(table, style)
        case "curve_adjacency":
            return _curve_adjacency(table, style)
        case "univariate_continuous":
            return _univariate_continuous(table, style, variants.univariate_continuous)
        case "mass_max":
            return _mass_max(table, style, window, variants.mass_max)
        case _:
            assert_never(figure)


def ceiling_colors(style: PlotStyle) -> dict[str, Any]:
    """The colour of each curve key's line on the ceilings figure, from the NS colormaps (W-097:
    the palette's index 1 is the BH anchor colour)."""
    return {
        "beta": colormap(style.neutron_stars.beta_cmap)(0.6),
        "lambda": colormap(style.neutron_stars.lambda_cmap)(0.6),
    }


# --- shell ------------------------------------------------------------------------------------


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The prepared table, the asset dir and the probe knobs; mk/paper.mk passes every knob."""
    parser = argparse.ArgumentParser(prog="eda_neutron_stars.py")
    parser.add_argument("table", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--folds", type=int, required=True, help="probe folds of the split")
    parser.add_argument("--seed", type=int, required=True, help="the split's seed")
    parser.add_argument("--window", type=float, required=True, help="pile-up window in M")
    parser.add_argument(
        "--state",
        type=Path,
        help="the state dir: the tables and numbers are stored as JSON under <state>/<section>/",
    )
    return parser.parse_args(argv)


def _bounds(table: pd.DataFrame) -> Ceilings:
    """The data ceilings of M and D along log10 rho_c, across beta and lambda."""
    return measure(
        table.assign(log10_rho_c=np.log10(table["rho_c"])), CURVE, "log10_rho_c", ("M", "D"), "D"
    )


def _ceilings(bounds: Ceilings, style: PlotStyle, variant: CeilingsVariant) -> Figure:
    """The ceilings figure in one variant."""
    labels, along = CEILING_LABELS, SYMBOLS["log10_rho_c"]
    return ceiling_figure(bounds, labels, along, ceiling_colors(style), variant)


@dataclass(frozen=True)
class FigureData:
    """What the figures plot, stored by the main run for the figures mode: the table with its
    charge targets, the pile-up window and the ceiling bounds."""

    table: pd.DataFrame
    window: float
    bounds: Ceilings


def _figure_data_path(state: Path) -> Path:
    """Where the main run stores the figure data, machine-local interim."""
    return state / SECTION / "figures.pkl"


def _figure_arguments(argv: list[str]) -> argparse.Namespace:
    """The figures mode: the out dir and the state dir the main run stored its data in."""
    parser = argparse.ArgumentParser(prog="eda_neutron_stars.py")
    parser.add_argument("--figures", type=Path, required=True, help="draw every variant here")
    parser.add_argument("--state", type=Path, required=True, help="the state dir of the main run")
    return parser.parse_args(argv)


def _draw_figures(out: Path, data: FigureData, style: PlotStyle) -> None:
    """Every variant of every figure with variants as <out>/<section>_fig_<name>_<variant>.png,
    drawn from the stored data alone."""
    out.mkdir(parents=True, exist_ok=True)
    drawers: dict[str, partial[Figure]] = {}
    for name, field in NeutronStarVariants.model_fields.items():
        for variant in get_args(field.annotation):
            if name == "ceilings":
                drawers[f"{SECTION}_fig_ceilings_{variant}"] = partial(
                    _ceilings, data.bounds, style, variant
                )
                continue
            if name not in get_args(NsFigure):
                raise ValueError(f"variants field {name} names no NS figure")
            figure = cast(NsFigure, name)
            chosen = NeutronStarVariants(**{name: variant})
            drawers[f"{asset(figure)}_{variant}"] = partial(
                draw, figure, data.table, style, data.window, chosen
            )
    for stem, drawer in drawers.items():
        fig = drawer()
        fig.savefig(out / f"{stem}.png", dpi=style.dpi)
        plt.close(fig)
    logger.info("done: %d NS figure variants -> %s", len(drawers), out)


def _store(state: Path, tables: Mapping[str, Table], macros: Mapping[str, str]) -> None:
    """Store the tables as JSON by name and the numbers under the section's state folder, for
    paperlib/tools/render_assets.py to write the assets from; an earlier run's tables go."""
    folder = state / SECTION / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*.json"):
        stale.unlink()
    for name, table in tables.items():
        (folder / f"{name}.json").write_text(to_json(table))
    (state / SECTION / "numbers.json").write_text(json.dumps(dict(macros)))


def _write_ceilings(
    out: Path, bounds: Ceilings, style: PlotStyle, variant: CeilingsVariant
) -> None:
    """Write the figure of the data limits."""
    caption_ = ceilings_caption(*CEILINGS_FIGURE, variant)
    (out / f"{SECTION}_fig_ceilings.tex").write_text(
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{SECTION}_fig_ceilings_{variant}}}\n"
        f"\\caption{{{caption_}}}\n\\label{{fig:ns-ceilings}}\n\\end{{figure}}\n"
    )


def main(
    argv: list[str],
    config: PaperConfig | None = None,
    section: NeutronStarsSection | None = None,
) -> None:
    """Write the selected figures and tables, the numbers and the figures .tex into the section's
    folder of the asset dir, after emptying that folder of an earlier run -- or, given --figures,
    only every figure variant as its own PNG, from the data the main run stored."""
    if "--figures" in argv:
        figure_args = _figure_arguments(argv)
        config = config or load_config()
        apply_style(config.plot, NOTATION_TEX.read_text())
        data = pickle.loads(_figure_data_path(figure_args.state).read_bytes())
        _draw_figures(figure_args.figures / SECTION / "figures", data, config.plot)
        return
    args = _arguments(argv)
    source, out = args.table, args.assets / SECTION
    assert source.is_file(), f"prepared NS table not found: {source}"
    config = config or load_config()
    apply_style(config.plot, NOTATION_TEX.read_text())
    prepared = pd.read_parquet(source)
    raw = {str(key): int(value) for key, value in prepared.attrs.items()}
    assert {"runs", "empty_runs", "rows"} <= raw.keys(), (
        f"{source} carries no raw counts -- re-run make data, the preparation writes them"
    )
    table = with_charge_targets(prepared)
    out.mkdir(parents=True, exist_ok=True)
    for stale in (p for p in out.iterdir() if p.is_file()):  # not figures/
        stale.unlink()
    section = section or load_section(NeutronStarsSection, SECTION_TOML)
    for figure in section.figures:
        if picked_stem(figure, section.variants) == asset(figure):
            # No variants: the figure has no file under build/figures, so it is drawn here.
            fig = draw(figure, table, config.plot, args.window, section.variants)
            fig.savefig(out / f"{asset(figure)}.png", dpi=config.plot.dpi)
            plt.close(fig)
        (out / f"{asset(figure)}.tex").write_text(figure_tex(figure, section.variants))
    found = evidence(table, args.folds, args.seed, args.window)
    bounds = _bounds(table)
    # The ceilings as uncertainties at the median mass, at M_max and at the median charge (W-076).
    typical = {
        "M": {"": float(table["M"].median()), "Max": float(table["M"].max())},
        "D": {"": float(table["D"].median())},
    }
    macros = (
        numbers(found)
        | raw_numbers(raw)
        | ceiling_macros("nsEda", bounds, CEILING_TAGS)
        | uncertainty_macros("nsEda", bounds, CEILING_TAGS, typical, "D")
    )
    assert args.state is not None, "--state is required"
    _store(args.state, table_objects(found, section.tables, bounds), macros)
    _figure_data_path(args.state).write_bytes(pickle.dumps(FigureData(table, args.window, bounds)))
    _write_ceilings(out, bounds, config.plot, section.variants.ceilings)
    logger.info(
        "done: NS figures %s -> %s, tables %s and numbers -> %s",
        section.figures,
        out,
        section.tables,
        args.state / SECTION,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
