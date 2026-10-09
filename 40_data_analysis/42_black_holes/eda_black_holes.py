"""Exploratory analysis of the prepared BH table behind Section 3.2 (W-018).

The BH data are 21 dense beta curves over the horizon radius r_h. The computations ask what the
NS analysis asked, and find different answers: how closely M follows r_h, how much beta moves M
and D at fixed r_h, where each curve starts (the existence edge), which charge target the data
support, and which split the curves support. The mass is raw M throughout (developer). Numbers
become \\bhEda... macros; the dataset-agnostic machinery lives in shared/eda.py.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from shared.ceilings import (
    Ceilings,
    ceiling_figure,
    ceiling_macros,
    ceiling_table,
    measure,
    uncertainty_macros,
)
from shared.config import NOTATION_TEX, BhFigure, BhTable, PaperConfig, load_config
from shared.eda import (
    Adjacency,
    ColumnSummary,
    booktabs,
    camel,
    curve_adjacency,
    curve_ids,
    grid_bins,
    group_splitter,
    holdout_mae,
    kfold_mae,
    make_curve_space,
    number_tex,
    orders_of_magnitude,
    render_macros,
    row_splitter,
    summarize,
    text_width_size,
    with_charge_targets,
)
from shared.plots import PlotStyle, anchor_color, apply_style, colormap

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

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
class ExistenceEdge:
    """Where each beta curve starts: (beta, smallest r_h), ordered by beta."""

    starts: tuple[tuple[float, float], ...]


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
    return ExistenceEdge(tuple((float(b), float(r)) for b, r in pairs))


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
    rows_per_curve: tuple[int, int]
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
        (int(sizes.min()), int(sizes.max())),
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
    (beta_low, rh_low), (beta_high, rh_high) = found.edge.starts[0], found.edge.starts[-1]
    rh, m = summary["r_h", "raw"], summary["M", "raw"]
    d, dm = summary["D", "raw"], summary["D_over_M", "raw"]
    return {
        "bhEdaRows": f"{found.rows}",
        "bhEdaCurves": f"{found.curves}",
        "bhEdaRowsPerCurveMin": f"{found.rows_per_curve[0]}",
        "bhEdaRowsPerCurveMax": f"{found.rows_per_curve[1]}",
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
        "bhEdaEdgeBetaLow": f"{beta_low:.2f}",
        "bhEdaEdgeBetaHigh": f"{beta_high:.2f}",
        "bhEdaEdgeRhMinLow": f"{rh_low:.3f}",
        "bhEdaEdgeRhMinHigh": f"{rh_high:.3f}",
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


LABELS = {"r_h": "$r_h$", "beta": "$\\beta$", "M": "$M$", "D": "$D$", "D_over_M": "$D/M$"}
TARGET_LABELS = {
    "D": "$D$",
    "log10_D": "$\\log_{10} D$",
    "D_over_M": "$D/M$",
    "log10_D_over_M": "$\\log_{10}(D/M)$",
}
STRATEGY_LABELS: dict[BhSplitStrategy, str] = {
    "random_rows": "random rows",
    "curves": "$\\beta$ curves (GroupKFold)",
    "outer_curves": "outer $\\beta$ curves, from the others",
}


def _univariate_table(found: Evidence) -> str:
    """Range, orders of magnitude and skew, raw against log10."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    rows = [
        f"{LABELS[c]} & ${number_tex(summary[c, 'raw'].minimum)}$ & "
        f"${number_tex(summary[c, 'raw'].maximum)}$ & "
        f"{orders_of_magnitude(summary[c, 'raw']):.2f} & "
        f"{summary[c, 'raw'].skew:.2f} & {summary[c, 'log10'].skew:.2f}"
        for c in SUMMARY_COLUMNS
    ]
    header = "Variable & min & max & orders & skew & skew of $\\log_{10}$"
    caption = (
        "Black holes: range, orders of magnitude spanned (orders) and skewness of each "
        "continuous variable, raw and in $\\log_{10}$."
    )
    return booktabs("bh-univariate", caption, "lrrrrr", header, rows)


def _charge_table(found: Evidence) -> str:
    """How each candidate charge target correlates with M, beta and r_h."""
    rows = [
        f"{TARGET_LABELS[t.target]} & {t.pearson_m:.3f} & {t.spearman_m:.3f} & "
        f"{t.pearson_beta:.3f} & {t.pearson_rh:.3f}"
        for t in found.targets
    ]
    header = "Target & Pearson $M$ & Spearman $M$ & Pearson $\\beta$ & Pearson $r_h$"
    caption = "Black holes: correlation of the candidate charge targets."
    return booktabs("bh-charge-correlation", caption, "lrrrr", header, rows)


def _split_table(found: Evidence) -> str:
    """1-NN error per split strategy and target."""
    error = {(s.strategy, s.target): _split_error(s) for s in found.splits}
    rows = [
        f"{STRATEGY_LABELS[strategy]} & ${error[strategy, 'M']}$ & ${error[strategy, 'log10_D']}$"
        for strategy in get_args(BhSplitStrategy)
    ]
    header = "Held out & $M$ & $\\log_{10} D$"
    caption = (
        f"Black holes: 1-nearest-neighbour mean absolute error under each split "
        f"({found.folds} folds; the outer curves are predicted from the others)."
    )
    return booktabs("bh-split-strategies", caption, "lrr", header, rows)


def table_tex(found: Evidence, table: BhTable) -> str:
    """The booktabs table environment of one table."""
    match table:
        case "univariate":
            return _univariate_table(found)
        case "charge_correlation":
            return _charge_table(found)
        case "split_strategies":
            return _split_table(found)
        case _:
            assert_never(table)


# The section's asset folder, and the prefix that keeps its basenames unique in the flat build.
SECTION = "42_black_holes"


def asset(figure: BhFigure) -> str:
    """The stem one figure's image and its .tex wrapper share."""
    return f"{SECTION}_fig_{figure}"


def caption(figure: BhFigure) -> str:
    """The caption of one figure."""
    match figure:
        case "univariate_continuous":
            return (
                "Black holes: distributions of the continuous variables, raw (left) and "
                "$\\log_{10}$ (right)."
            )
        case "mass_radius":
            return (
                "Black holes: the mass $M$ against $\\rh$, one line per $\\beta$ curve; the inset "
                "zooms on $\\rh$ from 4 to 5, where the curves leave the general-relativistic "
                "relation $M = \\rh/2$ (dashed)."
            )
        case "charge_target":
            return (
                "Black holes: $M$ against $\\log_{10}\\Dch$ (left) and $\\log_{10}(\\Dch/M)$ "
                "(right), coloured by $\\beta$."
            )
        case "existence_edge":
            return (
                "Black holes: the $\\rh$ range of each $\\beta$ curve; dots mark where each "
                "curve starts."
            )
        case _:
            assert_never(figure)


def figure_tex(figure: BhFigure) -> str:
    """The figure environment of one figure, placed here or at the top of a page (!htb)."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{asset(figure)}}}\n"
        f"\\caption{{{caption(figure)}}}\n\\label{{fig:bh-{figure.replace('_', '-')}}}\n"
        "\\end{figure}\n"
    )


# r_h sits on a fine grid of thousands of values; up to this many, one bar per value.
GRID_LIMIT = 250


def _univariate_continuous(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Histograms of the continuous variables, raw and in log10, grid-aware bins."""
    fig, axes = plt.subplots(
        len(SUMMARY_COLUMNS), 2, figsize=text_width_size(1.1), layout="constrained"
    )
    # Without edges: many thin bars would vanish under the style's white edges.
    bars = {"color": anchor_color(style, "black_holes"), "linewidth": 0}
    for row, column in enumerate(SUMMARY_COLUMNS):
        raw = table[column].to_numpy(dtype=float)
        logged = np.log10(raw)
        axes[row, 0].hist(raw, bins=grid_bins(raw, GRID_LIMIT), **bars)
        axes[row, 1].hist(logged, bins=grid_bins(logged, GRID_LIMIT), **bars)
        axes[row, 0].set_xlabel(LABELS[column])
        axes[row, 1].set_xlabel(f"$\\log_{{10}}$ {LABELS[column]}")
    return fig


def _curve_lines(table: pd.DataFrame, y: str, cmap: Colormap) -> LineCollection:
    """One line per beta curve of y against r_h, coloured by beta."""
    groups = [g for _, g in table.groupby(list(CURVE))]
    lines = LineCollection(
        [np.column_stack((g["r_h"], g[y])) for g in groups],
        array=np.array([g["beta"].iloc[0] for g in groups]),
        cmap=cmap,
        linewidths=0.6,
    )
    lines.set_rasterized(True)
    return lines


# The inset window near the existence edge, where the curves leave the GR relation M = r_h/2.
ZOOM_RH = (4.0, 5.0)
ZOOM_M = (2.0, 2.5)


def _gr_inset(ax: Axes, table: pd.DataFrame, cmap: Colormap) -> None:
    """Zoom on ZOOM_RH x ZOOM_M with the GR relation M = r_h/2 (G = c = 1) dashed."""
    inset = ax.inset_axes((0.55, 0.2, 0.42, 0.4))
    inset.add_collection(_curve_lines(table, "M", cmap))
    rh = np.linspace(*ZOOM_RH, 2)
    inset.plot(rh, rh / 2, color="black", linestyle="--", linewidth=0.8, label="GR")
    inset.set_xlim(*ZOOM_RH)
    inset.set_ylim(*ZOOM_M)
    inset.set_xticks(np.linspace(*ZOOM_RH, 3))
    inset.set_yticks(np.linspace(*ZOOM_M, 3))
    inset.legend(loc="upper left", fontsize="small", frameon=False)
    inset.tick_params(labelsize="small")
    ax.indicate_inset_zoom(inset, edgecolor="black")


def _mass_radius(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """M against r_h, one line per beta curve, with the GR inset near the existence edge."""
    fig, ax = plt.subplots(figsize=text_width_size(0.55), layout="constrained")
    cmap = colormap(style.black_holes.beta_cmap)
    lines = _curve_lines(table, "M", cmap)
    ax.add_collection(lines)
    ax.autoscale()
    ax.set_xlabel(LABELS["r_h"])
    ax.set_ylabel(LABELS["M"])
    _gr_inset(ax, table, cmap)
    fig.colorbar(lines, ax=ax, label=LABELS["beta"])
    return fig


def _charge_target(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """M against log10 D and log10 D/M, coloured by beta."""
    fig, axes = plt.subplots(1, 2, figsize=text_width_size(0.45), layout="constrained")
    cmap = colormap(style.black_holes.beta_cmap)
    scatters = [
        ax.scatter(
            table["M"],
            table[target],
            c=table["beta"],
            cmap=cmap,
            s=0.2,
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


def _existence_edge(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Each beta curve's r_h range as a horizontal segment, its start marked."""
    fig, ax = plt.subplots(figsize=text_width_size(0.45), layout="constrained")
    ranges = table.groupby("beta")["r_h"].agg(["min", "max"])
    cmap = colormap(style.black_holes.beta_cmap)
    shade = (ranges.index - ranges.index.min()) / max(np.ptp(ranges.index), 1e-12)
    ax.hlines(ranges.index, ranges["min"], ranges["max"], colors=cmap(shade), linewidth=1.2)
    ax.scatter(ranges["min"], ranges.index, color=cmap(shade), s=6, zorder=3)
    ax.set_xlabel(LABELS["r_h"])
    ax.set_ylabel(LABELS["beta"])
    return fig


def draw(figure: BhFigure, table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Draw one BH EDA figure from a table with the charge targets."""
    match figure:
        case "univariate_continuous":
            return _univariate_continuous(table, style)
        case "mass_radius":
            return _mass_radius(table, style)
        case "charge_target":
            return _charge_target(table, style)
        case "existence_edge":
            return _existence_edge(table, style)
        case _:
            assert_never(figure)


# --- shell ------------------------------------------------------------------------------------


def main(
    argv: list[str], config: PaperConfig | None = None, folds: int = 5, seed: int = 20261005
) -> None:
    """Write the selected figures and tables and the numbers into the section's folder of the
    asset dir, after emptying that folder of an earlier run."""
    assert len(argv) == 2, f"usage: eda_black_holes.py <table.parquet> <asset dir>, got {argv}"
    source, out = Path(argv[0]), Path(argv[1]) / SECTION
    assert source.is_file(), f"prepared BH table not found: {source}"
    config = config or load_config()
    apply_style(config.plot, NOTATION_TEX.read_text())
    table = with_charge_targets(pd.read_parquet(source))
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    section = config.data_analysis.black_holes
    for figure in section.figures:
        fig = draw(figure, table, config.plot)
        fig.savefig(out / f"{asset(figure)}.png", dpi=config.plot.dpi)
        plt.close(fig)
        (out / f"{asset(figure)}.tex").write_text(figure_tex(figure))
    found = evidence(table, folds, seed)
    bounds = measure(table, CURVE, "r_h", ("M", "D"), "D")
    # The ceilings as uncertainties at the median mass and the median charge (W-076).
    typical = {"M": {"": float(table["M"].median())}, "D": {"": float(table["D"].median())}}
    macros = (
        numbers(found)
        | ceiling_macros("bhEda", bounds, CEILING_TAGS)
        | uncertainty_macros("bhEda", bounds, CEILING_TAGS, typical, "D")
    )
    (out / f"{SECTION}_num.tex").write_text(render_macros(macros))
    for name in section.tables:
        (out / f"{SECTION}_tab_{name}.tex").write_text(table_tex(found, name))
    _write_ceilings(out, bounds, config.plot)
    logger.info(
        "done: BH figures %s, tables %s and numbers -> %s", section.figures, section.tables, out
    )


# The data limits (W-064): macro tags and TeX labels of the targets and of the curve key varied.
CEILING_TAGS = {"M": "M", "D": "D", "beta": "Beta"}
CEILING_LABELS = {"M": "$M$", "D": "$\\Dch$", "beta": "$\\beta$"}
# Matplotlib knows no paper macros: the figure names the charge D, as the other figures do.
CEILING_PLOT_LABELS = CEILING_LABELS | {"D": "$D$"}
CEILINGS_CAPTION = (
    "Black holes: the relative error the data allow, at the 95th percentile -- the noise "
    "along a curve (median), a cubic spline through every other row of a curve (along), and a "
    "curve predicted from the two curves on each side in $\\beta$ (across); figures is "
    "$-\\log_{10}$ of the across-curve error."
)
CEILINGS_FIGURE = (
    "Black holes: relative error of a curve predicted from the two curves on each side in "
    "$\\beta$, along $\\rh$ -- median solid, 95th percentile dashed; dotted lines mark 4 and 5 "
    "significant figures."
)


def _write_ceilings(out: Path, bounds: Ceilings, style: PlotStyle) -> None:
    """Write the ceilings table and the figure of the data limits."""
    (out / f"{SECTION}_tab_ceilings.tex").write_text(
        ceiling_table("bh-ceilings", CEILINGS_CAPTION, bounds, CEILING_LABELS)
    )
    fig = ceiling_figure(
        bounds, CEILING_PLOT_LABELS, "$r_h$", {"beta": anchor_color(style, "black_holes")}
    )
    fig.savefig(out / f"{SECTION}_fig_ceilings.png", dpi=style.dpi)
    plt.close(fig)
    (out / f"{SECTION}_fig_ceilings.tex").write_text(
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{SECTION}_fig_ceilings}}\n"
        f"\\caption{{{CEILINGS_FIGURE}}}\n\\label{{fig:bh-ceilings}}\n\\end{{figure}}\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
