"""Exploratory analysis of the prepared NS table behind Section 4.1 (W-017).

The computations are pure functions over the table, each returning a frozen record; their
numbers become \\nsEda... LaTeX macros so the text and the decisions table cite generated values.
The evidence they carry: which columns want log10, whether log10(D/M) is a better charge target
than log10 D, whether beta matters at fixed lambda, how full the (beta, lambda) grid is, and
whether rows of one curve are each other's nearest neighbours (why the split groups by curve).
The dataset-agnostic machinery lives in shared/eda.py.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from shared.config import NsFigure, NsTable, PaperConfig, load_config
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
    neighbour_distances,
    number_tex,
    orders_of_magnitude,
    render_macros,
    rim_mask,
    row_splitter,
    sci_tex,
    summarize,
    text_width_size,
    with_charge_targets,
)
from shared.plots import PlotStyle, anchor_color, apply_style, colormap

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

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
    """Every computed result Section 4.1 cites, computed once for the macros and the tables."""

    rows: int
    density_grid_values: int
    rows_per_curve: tuple[int, int]
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


def evidence(table: pd.DataFrame, folds: int, seed: int, window: float = 0.09) -> Evidence:
    """Compute every Section 4.1 result from a table with the charge targets."""
    sizes = table.groupby(list(CURVE)).size()
    return Evidence(
        len(table),
        table["rho_c"].nunique(),
        (int(sizes.min()), int(sizes.max())),
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


def numbers(found: Evidence) -> dict[str, str]:
    """Every number Section 4.1 cites, keyed by its \\nsEda macro name."""
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
        "nsEdaRowsPerCurveMin": f"{found.rows_per_curve[0]}",
        "nsEdaRowsPerCurveMax": f"{found.rows_per_curve[1]}",
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
        "nsEdaLambdaLinearPercentLogD": f"{100 * log_d.pearson_lambda ** 2:.0f}",
        "nsEdaLambdaLinearPercentLogDM": f"{100 * log_dm.pearson_lambda ** 2:.0f}",
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


TABLE_LABELS = {"rho_c": "$\\rho_c$", "M": "$M$", "D": "$D$", "D_over_M": "$D/M$"}
STRATEGY_LABELS: dict[SplitStrategy, str] = {
    "random_rows": "random rows",
    "curves": "$(\\beta, \\lambda)$ curves (GroupKFold)",
    "beta_lines": "whole $\\beta$ lines",
    "lambda_lines": "whole $\\lambda$ lines",
    "rim": "rim curves, from the interior",
}


def _univariate_table(found: Evidence) -> str:
    """Table A: range, orders of magnitude and skew, raw against log10."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    rows = [
        f"{TABLE_LABELS[c]} & ${number_tex(summary[c, 'raw'].minimum)}$ & "
        f"${number_tex(summary[c, 'raw'].maximum)}$ & "
        f"{orders_of_magnitude(summary[c, 'raw']):.1f} & "
        f"{summary[c, 'raw'].skew:.2f} & {summary[c, 'log10'].skew:.2f}"
        for c in SUMMARY_COLUMNS
    ]
    header = "Variable & min & max & orders & skew & skew of $\\log_{10}$"
    caption = (
        "Neutron stars: range, orders of magnitude spanned (orders) and skewness of each "
        "continuous variable, raw and in $\\log_{10}$."
    )
    return booktabs("ns-univariate", caption, "lrrrrr", header, rows)


def _charge_table(found: Evidence) -> str:
    """Table B: how each charge target correlates with M and the free parameters."""
    rows = [
        f"{label} & {t.pearson_m:.2f} & {t.spearman_m:.2f} & {t.pearson_lambda:.2f} & "
        f"{t.pearson_beta:.2f}"
        for t, (_, label) in zip(found.targets, CHARGE_TARGETS, strict=True)
    ]
    header = (
        "Target & Pearson $M$ & Spearman $M$ & Pearson $\\lambda$ & Pearson $\\beta$"
    )
    caption = "Neutron stars: correlation of the two candidate charge targets."
    return booktabs("ns-charge-correlation", caption, "lrrrr", header, rows)


def _split_table(found: Evidence) -> str:
    """Table C: 1-NN error per split strategy and target."""
    error = {(s.strategy, s.target): _split_error(s) for s in found.splits}
    rows = [
        f"{STRATEGY_LABELS[strategy]} & {error[strategy, 'log10_D_over_M']} & "
        f"{error[strategy, 'M']}"
        for strategy in get_args(SplitStrategy)
    ]
    header = "Held out & $\\log_{10}(D/M)$ & $M$"
    caption = (
        f"Neutron stars: 1-nearest-neighbour mean absolute error under each split "
        f"({found.folds} folds; the rim is predicted from the interior)."
    )
    return booktabs("ns-split-strategies", caption, "lrr", header, rows)


def table_tex(found: Evidence, table: NsTable) -> str:
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


TARGET_TAGS = {"M": "M", "log10_D_over_M": "DM"}
# The mass errors are an order of magnitude smaller, so they keep one more decimal.
SPLIT_DECIMALS = {"M": 4, "log10_D_over_M": 3}


def _split_error(score: SplitScore) -> str:
    """One split error with its target's decimals, as in the text and Table C alike."""
    return f"{score.mae:.{SPLIT_DECIMALS[score.target]}f}"


# The section's asset folder, and the prefix that keeps its basenames unique in the flat build.
SECTION = "41_neutron_stars"


def asset(figure: NsFigure) -> str:
    """The stem one figure's image and its .tex wrapper share."""
    return f"{SECTION}_fig_{figure}"


def caption(figure: NsFigure) -> str:
    """The caption of one figure."""
    match figure:
        case "univariate":
            return (
                "Neutron stars: distributions of each variable, raw (left) and "
                "$\\log_{10}$ (right)."
            )
        case "charge_target":
            return (
                "Neutron stars: $M$ against $\\log_{10}\\Dch$ (left) and $\\log_{10}(\\Dch/M)$ "
                "(right), coloured by $\\lambda$ (top) and $\\beta$ (bottom)."
            )
        case "mass_density":
            return (
                "Neutron stars: $M$ against $\\log_{10}\\rhoc$, one line per $(\\beta, \\lambda)$ "
                "curve coloured by $\\lambda$; dots mark each curve's last (maximum-mass) point."
            )
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
            return (
                "Neutron stars: distributions of the continuous variables, raw (left) and "
                "$\\log_{10}$ (right), one bar per value where the variable sits on a grid."
            )
        case "mass_max":
            return (
                "Neutron stars: distance of each row to its own curve's maximum mass, the "
                "pile-up window dashed (left), and each curve's maximum mass over the "
                "$(\\beta, \\lambda)$ grid (right)."
            )
        case _:
            assert_never(figure)


def figure_tex(figure: NsFigure) -> str:
    """The figure environment of one figure, placed here or at the top of a page (!htb)."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{asset(figure)}}}\n"
        f"\\caption{{{caption(figure)}}}\n\\label{{fig:ns-{figure.replace('_', '-')}}}\n"
        "\\end{figure}\n"
    )


LABELS = {
    "beta": "$\\beta$",
    "lambda": "$\\lambda$",
    "rho_c": "$\\rho_c$",
    "M": "$M$",
    "D": "$D$",
    "D_over_M": "$D/M$",
}


def _univariate(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Histograms of each variable, raw and in log10."""
    fig, axes = plt.subplots(len(LABELS), 2, figsize=text_width_size(1.6), layout="constrained")
    color = anchor_color(style, "neutron_stars")
    for row, (column, label) in enumerate(LABELS.items()):
        raw = table[column].to_numpy(dtype=float)
        axes[row, 0].hist(raw, bins=grid_bins(raw), color=color)
        axes[row, 1].hist(np.log10(raw), bins=grid_bins(np.log10(raw)), color=color)
        axes[row, 0].set_xlabel(label)
        axes[row, 1].set_xlabel(f"$\\log_{{10}}$ {label}")
    return fig


# rho_c sits on a shared grid of a few hundred values; one bar per value avoids aliasing.
DENSITY_GRID_VALUES = 250


def _univariate_continuous(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Histograms of the continuous variables, raw and in log10, grid-aware bins."""
    fig, axes = plt.subplots(
        len(SUMMARY_COLUMNS), 2, figsize=text_width_size(1.1), layout="constrained"
    )
    color = anchor_color(style, "neutron_stars")
    for row, column in enumerate(SUMMARY_COLUMNS):
        raw = table[column].to_numpy(dtype=float)
        logged = np.log10(raw)
        # Without edges: a few hundred thin bars would vanish under the style's white edges.
        bars = {"color": color, "linewidth": 0}
        axes[row, 0].hist(raw, bins=grid_bins(raw, DENSITY_GRID_VALUES), **bars)
        axes[row, 1].hist(logged, bins=grid_bins(logged, DENSITY_GRID_VALUES), **bars)
        axes[row, 0].set_xlabel(LABELS[column])
        axes[row, 1].set_xlabel(f"$\\log_{{10}}$ {LABELS[column]}")
    return fig


def _mass_max(table: pd.DataFrame, style: PlotStyle, window: float = 0.09) -> Figure:
    """Rows' distance to their curve's M_max, and each curve's M_max over the grid."""
    fig, (left, right) = plt.subplots(1, 2, figsize=text_width_size(0.45), layout="constrained")
    peak = table.groupby(list(CURVE))["M"].transform("max")
    left.hist(peak - table["M"], bins=40, color=anchor_color(style, "neutron_stars"))
    left.axvline(window, color="black", linestyle="--", linewidth=0.8)
    left.set_xlabel("$M_{\\max} - M$")
    left.set_ylabel("rows")
    maxima = table.groupby(list(CURVE))["M"].max().unstack("beta")
    mesh = right.pcolormesh(
        maxima.columns, maxima.index, maxima.to_numpy(),
        cmap=colormap(style.neutron_stars.lambda_cmap), shading="nearest",
    )
    right.set_xlabel(LABELS["beta"])
    right.set_ylabel(LABELS["lambda"])
    fig.colorbar(mesh, ax=right, label="$M_{\\max}$")
    return fig


CHARGE_TARGETS = (("log10_D", "$\\log_{10} D$"), ("log10_D_over_M", "$\\log_{10}(D/M)$"))


def _charge_row(
    fig: Figure, axes: np.ndarray, table: pd.DataFrame, column: str, cmap: Colormap
) -> None:
    """One row of the charge-target figure: both targets against M, coloured by column."""
    scatters = [
        ax.scatter(
            table["M"], table[target], c=table[column], cmap=cmap,
            s=0.2, linewidths=0, rasterized=True,
        )
        for ax, (target, _) in zip(axes, CHARGE_TARGETS, strict=True)
    ]
    for ax, (_, label) in zip(axes, CHARGE_TARGETS, strict=True):
        ax.set_ylabel(label)
    fig.colorbar(scatters[-1], ax=list(axes), label=LABELS[column])


def _charge_target(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """M against log10 D and log10 D/M, coloured by lambda and by beta."""
    fig, axes = plt.subplots(
        2, 2, figsize=text_width_size(0.9), sharex=True, layout="constrained"
    )
    _charge_row(fig, axes[0, :], table, "lambda", colormap(style.neutron_stars.lambda_cmap))
    _charge_row(fig, axes[1, :], table, "beta", colormap(style.neutron_stars.beta_cmap))
    for ax in axes[1, :]:
        ax.set_xlabel("$M$")
    return fig


def _mass_density(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """M against log10 rho_c, one line per curve coloured by lambda, last points marked."""
    fig, ax = plt.subplots(figsize=text_width_size(0.62), layout="constrained")
    cmap = colormap(style.neutron_stars.lambda_cmap)
    groups = [g for _, g in table.groupby(list(CURVE))]
    lines = LineCollection(
        [np.column_stack((np.log10(g["rho_c"]), g["M"])) for g in groups],
        array=np.array([g["lambda"].iloc[0] for g in groups]), cmap=cmap, linewidths=0.3,
    )
    lines.set_rasterized(True)
    ax.add_collection(lines)
    ends = pd.DataFrame([g.iloc[-1] for g in groups])
    ax.scatter(np.log10(ends["rho_c"]), ends["M"], c=ends["lambda"], cmap=cmap, s=1.5, zorder=3)
    ax.autoscale()
    ax.set_xlabel("$\\log_{10}\\rho_c$")
    ax.set_ylabel("$M$")
    fig.colorbar(lines, ax=ax, label=LABELS["lambda"])
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
    ax.hist(found.within, bins=edges, color=cmap(0.8), alpha=0.8, label="own curve")
    ax.hist(found.across, bins=edges, color=cmap(0.35), alpha=0.8, label="other curve")
    ax.set_xscale("log")
    ax.set_xlabel("nearest-neighbour distance (standardized)")
    ax.set_ylabel("rows")
    ax.legend()
    return fig


def draw(figure: NsFigure, table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Draw one NS EDA figure from a table with the charge targets."""
    match figure:
        case "univariate":
            return _univariate(table, style)
        case "charge_target":
            return _charge_target(table, style)
        case "mass_density":
            return _mass_density(table, style)
        case "grid_fill":
            return _grid_fill(table, style)
        case "curve_adjacency":
            return _curve_adjacency(table, style)
        case "univariate_continuous":
            return _univariate_continuous(table, style)
        case "mass_max":
            return _mass_max(table, style)
        case _:
            assert_never(figure)


# --- shell ------------------------------------------------------------------------------------


def main(
    argv: list[str], config: PaperConfig | None = None, folds: int = 5, seed: int = 20261005
) -> None:
    """Write the selected figures and tables, the numbers and the figures .tex into the section's
    folder of the asset dir, after emptying that folder of an earlier run."""
    assert len(argv) == 2, f"usage: eda_neutron_stars.py <table.parquet> <asset dir>, got {argv}"
    source, out = Path(argv[0]), Path(argv[1]) / SECTION
    assert source.is_file(), f"prepared NS table not found: {source}"
    config = config or load_config()
    apply_style(config.plot)
    table = with_charge_targets(pd.read_parquet(source))
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    section = config.data_analysis.neutron_stars
    for figure in section.figures:
        fig = draw(figure, table, config.plot)
        fig.savefig(out / f"{asset(figure)}.png", dpi=config.plot.dpi)
        plt.close(fig)
        (out / f"{asset(figure)}.tex").write_text(figure_tex(figure))
    found = evidence(table, folds, seed)
    (out / f"{SECTION}_num.tex").write_text(render_macros(numbers(found)))
    for name in section.tables:
        (out / f"{SECTION}_tab_{name}.tex").write_text(table_tex(found, name))
    logger.info(
        "done: NS figures %s, tables %s and numbers -> %s", section.figures, section.tables, out
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
