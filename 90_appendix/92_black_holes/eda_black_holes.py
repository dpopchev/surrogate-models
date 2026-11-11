"""Exploratory analysis of the prepared BH table behind Section 3.2 (W-018).

The BH data are 21 dense beta curves over the horizon radius r_h. The computations ask what the
NS analysis asked, and find different answers: how closely M follows r_h, how much beta moves M
and D at fixed r_h, where each curve starts (the existence edge), which charge target the data
support, and which split the curves support. The mass is raw M throughout (developer). Numbers
become \\bhEda... macros; the dataset-agnostic machinery lives in paperlib/eda.py.
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
from typing import Literal, assert_never, cast, get_args

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
    group_splitter,
    holdout_mae,
    kfold_mae,
    make_curve_space,
    number_tex,
    orders_of_magnitude,
    row_splitter,
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
    mark_size,
)
from paperlib.tables import Table, Text, aligns, fixed, number, render, to_json

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

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


SECTION_TOML = Path(__file__).with_name("92_black_holes.toml")

CURVE = ("beta",)
BH_SPACE = make_curve_space({"beta": "raw", "r_h": "raw"}, CURVE)
CHARGE_TARGETS = ("D", "log10_D", "D_over_M", "log10_D_over_M")


@dataclass(frozen=True)
class ChargeTarget:
    """How one candidate charge target correlates with M, beta and r_h."""

    target: str
    pearson_m: float
    spearman_m: float
    pearson_beta: float
    pearson_rh: float


@dataclass(frozen=True)
class BetaEffect:
    """How far beta moves a target at fixed r_h: the median spread across beta at one r_h value
    over the target's total spread (0 when beta does not matter)."""

    target: str
    spread_ratio: float


@dataclass(frozen=True)
class CurveStart:
    """Where one beta curve starts: its beta and its smallest r_h."""

    beta: float
    rh_min: float


@dataclass(frozen=True)
class ExistenceEdge:
    """Where each beta curve starts, ordered by beta."""

    starts: tuple[CurveStart, ...]


BhSplitStrategy = Literal["random_rows", "curves", "outer_curves"]


@dataclass(frozen=True)
class SplitScore:
    """1-nearest-neighbour error of a target when the split holds out what the strategy names.

    random_rows: rows at random; curves: whole beta curves (GroupKFold); outer_curves: the
    smallest and largest beta curves, predicted from the others (extrapolation in beta).
    """

    strategy: BhSplitStrategy
    target: str
    mae: float


# The data limits (W-064): macro tags and TeX labels of the targets and of the curve key varied.
CEILING_TAGS = {"M": "M", "D": "D", "beta": "Beta"}
CEILING_LABELS = SYMBOLS.of("M", "D", "beta")
CEILINGS_CAPTION = (
    "Black holes: the relative error the data allow, at the 95th percentile -- the noise "
    "along a curve (median), a cubic spline through every other row of a curve (along), and a "
    "curve predicted from the two curves on each side in $\\beta$ (across); figures is "
    "$-\\log_{10}$ of the across-curve error."
)
# The ceilings figure's caption: the dataset, the curve keys varied across, the coordinate along.
CEILINGS_FIGURE = ("Black holes", "$\\beta$", "$\\rh$")


# --- pure functions ---------------------------------------------------------------------------


def charge_targets(table: pd.DataFrame) -> tuple[ChargeTarget, ...]:
    """Correlate each candidate charge target with M, beta and r_h, in CHARGE_TARGETS order."""
    return tuple(
        ChargeTarget(
            target,
            float(table[target].corr(table["M"])),
            float(table[target].corr(table["M"], method="spearman")),
            float(table[target].corr(table["beta"])),
            float(table[target].corr(table["r_h"])),
        )
        for target in CHARGE_TARGETS
    )


def beta_effect(table: pd.DataFrame, target: str) -> BetaEffect:
    """Measure how far beta moves target at fixed r_h.

    The curves share one r_h grid, so each r_h value holds one row per curve present there;
    r_h values with a single curve carry no spread and are skipped.
    """
    across_beta = table.groupby("r_h")[target].std().dropna()
    return BetaEffect(target, float(across_beta.median() / table[target].std()))


def existence_edge(table: pd.DataFrame) -> ExistenceEdge:
    """Find where each beta curve starts in r_h."""
    starts = table.groupby("beta")["r_h"].min()
    pairs = zip(starts.index.to_numpy(dtype=float), starts.to_numpy(dtype=float), strict=True)
    return ExistenceEdge(tuple(CurveStart(float(b), float(r)) for b, r in pairs))


def _score(
    table: pd.DataFrame, target: str, strategy: BhSplitStrategy, folds: int, seed: int
) -> float:
    """The 1-NN error of one strategy."""
    match strategy:
        case "random_rows":
            return kfold_mae(table, BH_SPACE, target, row_splitter(folds, seed), None)
        case "curves":
            groups = curve_ids(table, BH_SPACE)
            return kfold_mae(table, BH_SPACE, target, group_splitter(folds, seed), groups)
        case "outer_curves":
            outer = table["beta"].isin([table["beta"].min(), table["beta"].max()])
            return holdout_mae(table, BH_SPACE, target, outer.to_numpy())
        case _:
            assert_never(strategy)


def split_strategies(
    table: pd.DataFrame, target: str, folds: int, seed: int
) -> tuple[SplitScore, ...]:
    """1-NN error of target under each split strategy, in BhSplitStrategy order."""
    return tuple(
        SplitScore(strategy, target, _score(table, target, strategy, folds, seed))
        for strategy in get_args(BhSplitStrategy)
    )


SUMMARY_COLUMNS = ("r_h", "M", "D", "D_over_M")
BETA_EFFECT_TARGETS = ("M", "log10_D")
# Macro tags of the split targets: M and log10 D.
SPLIT_TAGS = {"M": "M", "log10_D": "LD"}


@dataclass(frozen=True)
class Evidence:
    """Every computed result Section 3.2 cites, computed once for the macros and the tables."""

    rows: int
    curves: int
    rows_per_curve: RowsPerCurve
    rh_step: float
    pearson_m_rh: float
    folds: int
    summaries: tuple[ColumnSummary, ...]
    targets: tuple[ChargeTarget, ...]
    beta_effects: tuple[BetaEffect, ...]
    edge: ExistenceEdge
    adjacency: Adjacency
    splits: tuple[SplitScore, ...]


def evidence(table: pd.DataFrame, folds: int, seed: int) -> Evidence:
    """Compute every Section 3.2 result from a table with the charge targets."""
    sizes = table.groupby(list(CURVE)).size()
    grid = np.unique(table["r_h"].to_numpy())
    return Evidence(
        len(table),
        len(sizes),
        RowsPerCurve(int(sizes.min()), int(sizes.max())),
        float(np.diff(grid).min()),
        float(table["M"].corr(table["r_h"])),
        folds,
        summarize(table, SUMMARY_COLUMNS),
        charge_targets(table),
        tuple(beta_effect(table, target) for target in BETA_EFFECT_TARGETS),
        existence_edge(table),
        curve_adjacency(table, BH_SPACE),
        tuple(s for t in SPLIT_TAGS for s in split_strategies(table, t, folds, seed)),
    )


def numbers(found: Evidence) -> dict[str, str]:
    """Every number Section 3.2 cites, keyed by its \\bhEda macro name."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    target = {t.target: t for t in found.targets}
    effect = {e.target: e.spread_ratio for e in found.beta_effects}
    low, high = found.edge.starts[0], found.edge.starts[-1]
    rh, m = summary["r_h", "raw"], summary["M", "raw"]
    d, dm = summary["D", "raw"], summary["D_over_M", "raw"]
    return {
        "bhEdaRows": f"{found.rows}",
        "bhEdaCurves": f"{found.curves}",
        "bhEdaRowsPerCurveMin": f"{found.rows_per_curve.minimum}",
        "bhEdaRowsPerCurveMax": f"{found.rows_per_curve.maximum}",
        "bhEdaRhStep": f"{found.rh_step:.3f}",
        "bhEdaRhMin": f"{rh.minimum:.3f}",
        "bhEdaRhMax": f"{rh.maximum:.3f}",
        "bhEdaRhOrders": f"{orders_of_magnitude(rh):.2f}",
        "bhEdaRhRawSkew": f"{rh.skew:.2f}",
        "bhEdaPearsonMRh": f"{found.pearson_m_rh:.4f}",
        "bhEdaMMin": f"{m.minimum:.3f}",
        "bhEdaMMax": f"{m.maximum:.3f}",
        "bhEdaDMin": number_tex(d.minimum),
        "bhEdaDMax": number_tex(d.maximum),
        "bhEdaDOrders": f"{orders_of_magnitude(d):.2f}",
        "bhEdaDMOrders": f"{orders_of_magnitude(dm):.2f}",
        "bhEdaSpearmanMD": f"{target['D'].spearman_m:.3f}",
        "bhEdaSpearmanMLogD": f"{target['log10_D'].spearman_m:.3f}",
        "bhEdaSpearmanMLogDM": f"{target['log10_D_over_M'].spearman_m:.3f}",
        "bhEdaPearsonRhLogD": f"{target['log10_D'].pearson_rh:.3f}",
        "bhEdaPearsonRhLogDM": f"{target['log10_D_over_M'].pearson_rh:.3f}",
        "bhEdaPearsonBetaLogD": f"{target['log10_D'].pearson_beta:.3f}",
        "bhEdaPearsonBetaLogDM": f"{target['log10_D_over_M'].pearson_beta:.3f}",
        "bhEdaBetaEffectPercentM": f"{100 * effect['M']:.2f}",
        "bhEdaBetaEffectPercentLD": f"{100 * effect['log10_D']:.1f}",
        "bhEdaEdgeBetaLow": f"{low.beta:.2f}",
        "bhEdaEdgeBetaHigh": f"{high.beta:.2f}",
        "bhEdaEdgeRhMinLow": f"{low.rh_min:.3f}",
        "bhEdaEdgeRhMinHigh": f"{high.rh_min:.3f}",
        "bhEdaSameCurvePercent": f"{100 * found.adjacency.same_curve_share:.1f}",
        "bhEdaWithinMedian": f"{found.adjacency.within_median:.4f}",
        "bhEdaAcrossMedian": f"{found.adjacency.across_median:.4f}",
        "bhEdaFolds": f"{found.folds}",
    } | {
        f"bhEdaSplit{camel(s.strategy)}{SPLIT_TAGS[s.target]}": _split_error(s)
        for s in found.splits
    }


def _split_error(score: SplitScore) -> str:
    """One split error for math mode; BH errors span orders of magnitude, so the format adapts."""
    return number_tex(score.mae)


LABELS = SYMBOLS.of("r_h", "beta", "M", "D", "D_over_M")
TARGET_LABELS = SYMBOLS.of("D", "log10_D", "D_over_M", "log10_D_over_M")
STRATEGY_LABELS: dict[BhSplitStrategy, str] = {
    "random_rows": "random rows",
    "curves": "$\\beta$ curves (GroupKFold)",
    "outer_curves": "outer $\\beta$ curves, from the others",
}


def _univariate_table(found: Evidence) -> Table:
    """Range, orders of magnitude and skew, raw against log10."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    rows = tuple(
        (
            Text(LABELS[c]),
            number(summary[c, "raw"].minimum),
            number(summary[c, "raw"].maximum),
            fixed(orders_of_magnitude(summary[c, "raw"]), 2),
            fixed(summary[c, "raw"].skew, 2),
            fixed(summary[c, "log10"].skew, 2),
        )
        for c in SUMMARY_COLUMNS
    )
    header = tuple(
        Text(h) for h in ("Variable", "min", "max", "orders", "skew", "skew of $\\log_{10}$")
    )
    caption = (
        "Black holes: range, orders of magnitude spanned (orders) and skewness of each "
        "continuous variable, raw and in $\\log_{10}$."
    )
    return Table("bh-univariate", caption, aligns("lrrrrr"), (header,), rows)


def _charge_table(found: Evidence) -> Table:
    """How each candidate charge target correlates with M, beta and r_h."""
    rows = tuple(
        (
            Text(TARGET_LABELS[t.target]),
            fixed(t.pearson_m, 3),
            fixed(t.spearman_m, 3),
            fixed(t.pearson_beta, 3),
            fixed(t.pearson_rh, 3),
        )
        for t in found.targets
    )
    header = tuple(
        Text(h)
        for h in (
            "Target",
            "Pearson $M$",
            "Spearman $M$",
            "Pearson $\\beta$",
            "Pearson $r_h$",
        )
    )
    caption = "Black holes: correlation of the candidate charge targets."
    return Table("bh-charge-correlation", caption, aligns("lrrrr"), (header,), rows)


def _split_table(found: Evidence) -> Table:
    """1-NN error per split strategy and target."""
    error = {(s.strategy, s.target): number(s.mae) for s in found.splits}
    rows = tuple(
        (
            Text(STRATEGY_LABELS[strategy]),
            error[strategy, "M"],
            error[strategy, "log10_D"],
        )
        for strategy in get_args(BhSplitStrategy)
    )
    header = tuple(Text(h) for h in ("Held out", "$M$", "$\\log_{10} D$"))
    caption = (
        f"Black holes: 1-nearest-neighbour mean absolute error under each split "
        f"({found.folds} folds; the outer curves are predicted from the others)."
    )
    return Table("bh-split-strategies", caption, aligns("lrr"), (header,), rows)


def table_object(found: Evidence, table: BhTable) -> Table:
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


def table_tex(found: Evidence, table: BhTable, layout: TableLayout = "small") -> str:
    """The booktabs table environment of one table in the paper's table layout."""
    return render(table_object(found, table), layout)


def table_objects(
    found: Evidence, names: tuple[BhTable, ...], bounds: Ceilings
) -> dict[str, Table]:
    """The section's tables by stored name: the selected ones and the ceilings table."""
    return {name: table_object(found, name) for name in names} | {
        "ceilings": ceiling_object("bh-ceilings", CEILINGS_CAPTION, bounds, CEILING_LABELS)
    }


# The section's asset folder, and the prefix that keeps its basenames unique in the flat build.
SECTION = "92_black_holes"
DEFAULT_VARIANTS = BlackHoleVariants()


def asset(figure: BhFigure) -> str:
    """The stem one figure's image and its .tex wrapper share."""
    return f"{SECTION}_fig_{figure}"


def picked_stem(figure: BhFigure, variants: BlackHoleVariants = DEFAULT_VARIANTS) -> str:
    """The stem of the file the paper includes: the picked variant's PNG under
    build/sections/<section>/figures/."""
    return f"{asset(figure)}_{getattr(variants, figure)}"


def _mass_radius_caption(variant: BhMassRadiusVariant) -> str:
    """The caption of the mass_radius figure in one variant."""
    match variant:
        case "inset":
            return (
                "Black holes: the mass $M$ against $\\rh$, one line per $\\beta$ curve; the inset "
                "zooms on $\\rh$ from 4 to 5, where the curves leave the general-relativistic "
                "relation $M = \\rh/2$ (dashed)."
            )
        case "departure":
            return (
                "Black holes: the departure $M - \\rh/2$ of the mass from the "
                "general-relativistic relation $M = \\rh/2$ (dashed zero line) against $\\rh$, "
                "one line per $\\beta$ curve."
            )
        case "both":
            return (
                "Black holes: the mass $M$ against $\\rh$ (left) and its departure $M - \\rh/2$ "
                "from the general-relativistic relation (right, dashed zero line), one line per "
                "$\\beta$ curve."
            )
        case _:
            assert_never(variant)


def caption(figure: BhFigure, variants: BlackHoleVariants = DEFAULT_VARIANTS) -> str:
    """The caption of one figure in its picked variant."""
    match figure:
        case "univariate_continuous":
            return univariate_caption("Black holes", variants.univariate_continuous)
        case "mass_radius":
            return _mass_radius_caption(variants.mass_radius)
        case "charge_target":
            match variants.charge_target:
                case "two_panels":
                    return (
                        "Black holes: $M$ against $\\log_{10}\\Dch$ (left) and "
                        "$\\log_{10}(\\Dch/M)$ (right), coloured by $\\beta$."
                    )
                case "log_d":
                    return "Black holes: $M$ against $\\log_{10}\\Dch$, coloured by $\\beta$."
                case _:
                    assert_never(variants.charge_target)
        case "existence_edge":
            match variants.existence_edge:
                case "segments":
                    return (
                        "Black holes: the $\\rh$ range of each $\\beta$ curve; dots mark where "
                        "each curve starts."
                    )
                case "edge_line":
                    return (
                        "Black holes: where each $\\beta$ curve starts, its smallest $\\rh$ "
                        "against $\\beta$."
                    )
                case "zoomed":
                    return (
                        "Black holes: the $\\rh$ range of each $\\beta$ curve near the existence "
                        "edge; dots mark where each curve starts."
                    )
                case _:
                    assert_never(variants.existence_edge)
        case _:
            assert_never(figure)


def figure_tex(figure: BhFigure, variants: BlackHoleVariants = DEFAULT_VARIANTS) -> str:
    """The figure environment of one figure in its picked variant, placed here or at the top
    of a page (!htb)."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{picked_stem(figure, variants)}}}\n"
        f"\\caption{{{caption(figure, variants)}}}\n"
        f"\\label{{fig:bh-{figure.replace('_', '-')}}}\n"
        "\\end{figure}\n"
    )


def _univariate_continuous(
    table: pd.DataFrame, style: PlotStyle, variant: UnivariateVariant
) -> Figure:
    """Histograms of the continuous variables, raw and in log10, grid-aware bins: r_h sits on
    a fine grid of thousands of values; up to grid_values, one bar per value."""
    color = anchor_color(style, "black_holes")
    return univariate_figure(table, SUMMARY_COLUMNS, LABELS, color, style.grid_values, variant)


def _curve_lines(table: pd.DataFrame, style: PlotStyle, y: str, cmap: Colormap) -> LineCollection:
    """One line per beta curve of y against r_h, coloured by beta."""
    groups = [g for _, g in table.groupby(list(CURVE))]
    lines = LineCollection(
        [np.column_stack((g["r_h"], g[y])) for g in groups],
        array=np.array([g["beta"].iloc[0] for g in groups]),
        cmap=cmap,
        linewidths=line_width(style, 0.6),
    )
    lines.set_rasterized(True)
    return lines


# The inset window near the existence edge, where the curves leave the GR relation M = r_h/2.
ZOOM_RH = (4.0, 5.0)
ZOOM_M = (2.0, 2.5)


def _gr_inset(ax: Axes, table: pd.DataFrame, style: PlotStyle, cmap: Colormap) -> None:
    """Zoom on ZOOM_RH x ZOOM_M with the GR relation M = r_h/2 (G = c = 1) dashed."""
    inset = ax.inset_axes((0.55, 0.2, 0.42, 0.4))
    inset.add_collection(_curve_lines(table, style, "M", cmap))
    rh = np.linspace(*ZOOM_RH, 2)
    width = line_width(style, 0.8)
    inset.plot(rh, rh / 2, color="black", linestyle="--", linewidth=width, label="GR")
    inset.set_xlim(*ZOOM_RH)
    inset.set_ylim(*ZOOM_M)
    inset.set_xticks(np.linspace(*ZOOM_RH, 3))
    inset.set_yticks(np.linspace(*ZOOM_M, 3))
    inset.legend(loc="upper left", fontsize="small", frameon=False)
    inset.tick_params(labelsize="small")
    ax.indicate_inset_zoom(inset, edgecolor="black")


def _mass_lines(ax: Axes, table: pd.DataFrame, style: PlotStyle, cmap: Colormap) -> LineCollection:
    """M against r_h on ax, one line per beta curve."""
    lines = _curve_lines(table, style, "M", cmap)
    ax.add_collection(lines)
    ax.autoscale()
    ax.set_xlabel(LABELS["r_h"])
    ax.set_ylabel(LABELS["M"])
    return lines


def _departure_lines(
    ax: Axes, table: pd.DataFrame, style: PlotStyle, cmap: Colormap
) -> LineCollection:
    """M - r_h/2 against r_h on ax, one line per beta curve; the GR relation is the dashed zero."""
    departed = table.assign(departure=table["M"] - table["r_h"] / 2)
    lines = _curve_lines(departed, style, "departure", cmap)
    ax.add_collection(lines)
    ax.autoscale()
    ax.axhline(0.0, color="black", linestyle="--", linewidth=line_width(style, 0.8), label="GR")
    ax.set_xlabel(LABELS["r_h"])
    ax.set_ylabel(f"{LABELS['M']} $-$ {LABELS['r_h']}$/2$")
    return lines


def _mass_radius(table: pd.DataFrame, style: PlotStyle, variant: BhMassRadiusVariant) -> Figure:
    """M against r_h, one line per beta curve: with the GR inset near the existence edge
    (inset), as the departure M - r_h/2 from GR (departure), or the two side by side (both)."""
    cmap = colormap(style.black_holes.beta_cmap)
    match variant:
        case "inset":
            fig, ax = plt.subplots(figsize=text_width_size(0.55), layout="constrained")
            lines = _mass_lines(ax, table, style, cmap)
            _gr_inset(ax, table, style, cmap)
            fig.colorbar(lines, ax=ax, label=LABELS["beta"])
        case "departure":
            fig, ax = plt.subplots(figsize=text_width_size(0.55), layout="constrained")
            lines = _departure_lines(ax, table, style, cmap)
            fig.colorbar(lines, ax=ax, label=LABELS["beta"])
        case "both":
            fig, (left, right) = plt.subplots(
                1, 2, figsize=text_width_size(0.45), layout="constrained"
            )
            _mass_lines(left, table, style, cmap)
            lines = _departure_lines(right, table, style, cmap)
            fig.colorbar(lines, ax=[left, right], label=LABELS["beta"])
        case _:
            assert_never(variant)
    return fig


def _charge_target(table: pd.DataFrame, style: PlotStyle, variant: BhChargeTargetVariant) -> Figure:
    """M against log10 D and log10 D/M (two_panels) or against log10 D alone (log_d), coloured
    by beta."""
    match variant:
        case "two_panels":
            return _charge_target_pair(table, style)
        case "log_d":
            fig, ax = plt.subplots(figsize=text_width_size(0.55), layout="constrained")
            scatter = ax.scatter(
                table["M"],
                table["log10_D"],
                c=table["beta"],
                cmap=colormap(style.black_holes.beta_cmap),
                s=mark_size(style, 0.2),
                linewidths=0,
                rasterized=True,
            )
            ax.set_xlabel(LABELS["M"])
            ax.set_ylabel(TARGET_LABELS["log10_D"])
            fig.colorbar(scatter, ax=ax, label=LABELS["beta"])
            return fig
        case _:
            assert_never(variant)


def _charge_target_pair(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """M against log10 D and log10 D/M, coloured by beta."""
    fig, axes = plt.subplots(1, 2, figsize=text_width_size(0.45), layout="constrained")
    cmap = colormap(style.black_holes.beta_cmap)
    scatters = [
        ax.scatter(
            table["M"],
            table[target],
            c=table["beta"],
            cmap=cmap,
            s=mark_size(style, 0.2),
            linewidths=0,
            rasterized=True,
        )
        for ax, target in zip(axes, ("log10_D", "log10_D_over_M"), strict=True)
    ]
    for ax, target in zip(axes, ("log10_D", "log10_D_over_M"), strict=True):
        ax.set_xlabel(LABELS["M"])
        ax.set_ylabel(TARGET_LABELS[target])
    fig.colorbar(scatters[-1], ax=list(axes), label=LABELS["beta"])
    return fig


def _existence_edge(
    table: pd.DataFrame, style: PlotStyle, variant: BhExistenceEdgeVariant
) -> Figure:
    """Each beta curve's r_h range as a segment, its start marked (segments), the same near
    the edge (zoomed), or each curve's start r_h,min against beta as one line (edge_line)."""
    match variant:
        case "segments":
            return _existence_segments(table, style)
        case "edge_line":
            fig, ax = plt.subplots(figsize=text_width_size(0.45), layout="constrained")
            found = existence_edge(table)
            ax.plot(
                [start.beta for start in found.starts],
                [start.rh_min for start in found.starts],
                color=anchor_color(style, "black_holes"),
                marker="o",
                markersize=mark_size(style, 2),
            )
            ax.set_xlabel(LABELS["beta"])
            ax.set_ylabel(f"{LABELS['r_h']}$_{{,\\min}}$")
            return fig
        case "zoomed":
            fig = _existence_segments(table, style)
            starts = [start.rh_min for start in existence_edge(table).starts]
            pad = 0.25 * (max(starts) - min(starts))
            fig.axes[0].set_xlim(min(starts) - pad, max(starts) + pad)
            return fig
        case _:
            assert_never(variant)


def _existence_segments(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Each beta curve's r_h range as a horizontal segment, its start marked."""
    fig, ax = plt.subplots(figsize=text_width_size(0.45), layout="constrained")
    ranges = table.groupby("beta")["r_h"].agg(["min", "max"])
    cmap = colormap(style.black_holes.beta_cmap)
    shade = (ranges.index - ranges.index.min()) / max(np.ptp(ranges.index), 1e-12)
    ax.hlines(
        ranges.index,
        ranges["min"],
        ranges["max"],
        colors=cmap(shade),
        linewidth=line_width(style, 1.2),
    )
    ax.scatter(ranges["min"], ranges.index, color=cmap(shade), s=mark_size(style, 6), zorder=3)
    ax.set_xlabel(LABELS["r_h"])
    ax.set_ylabel(LABELS["beta"])
    return fig


def draw(
    figure: BhFigure,
    table: pd.DataFrame,
    style: PlotStyle,
    variants: BlackHoleVariants = DEFAULT_VARIANTS,
) -> Figure:
    """Draw one BH EDA figure from a table with the charge targets, in its picked variant."""
    match figure:
        case "univariate_continuous":
            return _univariate_continuous(table, style, variants.univariate_continuous)
        case "mass_radius":
            return _mass_radius(table, style, variants.mass_radius)
        case "charge_target":
            return _charge_target(table, style, variants.charge_target)
        case "existence_edge":
            return _existence_edge(table, style, variants.existence_edge)
        case _:
            assert_never(figure)


# --- shell ------------------------------------------------------------------------------------


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The prepared table, the asset dir and the probe knobs; mk/paper.mk passes every knob."""
    parser = argparse.ArgumentParser(prog="eda_black_holes.py")
    parser.add_argument("table", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--folds", type=int, required=True, help="probe folds of the split")
    parser.add_argument("--seed", type=int, required=True, help="the split's seed")
    parser.add_argument(
        "--state",
        type=Path,
        help="the state dir: the tables and numbers are stored as JSON under <state>/<section>/",
    )
    return parser.parse_args(argv)


def _bounds(table: pd.DataFrame) -> Ceilings:
    """The data ceilings of M and D along r_h, across beta."""
    return measure(table, CURVE, "r_h", ("M", "D"), "D")


def _ceilings(bounds: Ceilings, style: PlotStyle, variant: CeilingsVariant) -> Figure:
    """The ceilings figure in one variant."""
    colors = {"beta": anchor_color(style, "black_holes")}
    return ceiling_figure(bounds, CEILING_LABELS, SYMBOLS["r_h"], colors, variant)


@dataclass(frozen=True)
class FigureData:
    """What the figures plot, stored by the main run for the figures mode: the table with its
    charge targets and the ceiling bounds."""

    table: pd.DataFrame
    bounds: Ceilings


def _figure_data_path(state: Path) -> Path:
    """Where the main run stores the figure data, machine-local interim."""
    return state / SECTION / "figures.pkl"


def _figure_arguments(argv: list[str]) -> argparse.Namespace:
    """The figures mode: the out dir and the state dir the main run stored its data in."""
    parser = argparse.ArgumentParser(prog="eda_black_holes.py")
    parser.add_argument("--figures", type=Path, required=True, help="draw every variant here")
    parser.add_argument("--state", type=Path, required=True, help="the state dir of the main run")
    return parser.parse_args(argv)


def _draw_figures(out: Path, data: FigureData, style: PlotStyle) -> None:
    """Every variant of every figure with variants as <out>/<section>_fig_<name>_<variant>.png,
    drawn from the stored data alone."""
    out.mkdir(parents=True, exist_ok=True)
    drawers: dict[str, partial[Figure]] = {}
    for name, field in BlackHoleVariants.model_fields.items():
        for variant in get_args(field.annotation):
            if name == "ceilings":
                drawers[f"{SECTION}_fig_ceilings_{variant}"] = partial(
                    _ceilings, data.bounds, style, variant
                )
                continue
            if name not in get_args(BhFigure):
                raise ValueError(f"variants field {name} names no BH figure")
            figure = cast(BhFigure, name)
            chosen = BlackHoleVariants.model_validate({name: variant})
            drawers[f"{asset(figure)}_{variant}"] = partial(draw, figure, data.table, style, chosen)
    for stem, drawer in drawers.items():
        fig = drawer()
        fig.savefig(out / f"{stem}.png", dpi=style.dpi)
        plt.close(fig)
    logger.info("done: %d BH figure variants -> %s", len(drawers), out)


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
        f"\\caption{{{caption_}}}\n\\label{{fig:bh-ceilings}}\n\\end{{figure}}\n"
    )


def main(
    argv: list[str],
    config: PaperConfig | None = None,
    section: BlackHolesSection | None = None,
) -> None:
    """Write the selected figures and tables and the numbers into the section's folder of the
    asset dir, after emptying that folder of an earlier run -- or, given --figures, every
    variant as its own PNG from the data the main run stored."""
    if "--figures" in argv:
        figure_args = _figure_arguments(argv)
        config = config or load_config()
        apply_style(config.plot, NOTATION_TEX.read_text())
        data = pickle.loads(_figure_data_path(figure_args.state).read_bytes())
        _draw_figures(figure_args.figures / SECTION / "figures", data, config.plot)
        return
    args = _arguments(argv)
    source, out = args.table, args.assets / SECTION
    assert source.is_file(), f"prepared BH table not found: {source}"
    config = config or load_config()
    apply_style(config.plot, NOTATION_TEX.read_text())
    table = with_charge_targets(pd.read_parquet(source))
    out.mkdir(parents=True, exist_ok=True)
    for stale in (p for p in out.iterdir() if p.is_file()):  # not figures/
        stale.unlink()
    section = section or load_section(BlackHolesSection, SECTION_TOML)
    for figure in section.figures:
        (out / f"{asset(figure)}.tex").write_text(figure_tex(figure, section.variants))
    found = evidence(table, args.folds, args.seed)
    bounds = _bounds(table)
    # The ceilings as uncertainties at the median mass and the median charge (W-076).
    typical = {"M": {"": float(table["M"].median())}, "D": {"": float(table["D"].median())}}
    macros = (
        numbers(found)
        | ceiling_macros("bhEda", bounds, CEILING_TAGS)
        | uncertainty_macros("bhEda", bounds, CEILING_TAGS, typical, "D")
    )
    assert args.state is not None, "--state is required"
    _store(args.state, table_objects(found, section.tables, bounds), macros)
    _figure_data_path(args.state).write_bytes(pickle.dumps(FigureData(table, bounds)))
    _write_ceilings(out, bounds, config.plot, section.variants.ceilings)
    logger.info(
        "done: BH figures %s -> %s, tables %s and numbers -> %s",
        section.figures,
        out,
        section.tables,
        args.state / SECTION,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
