"""Exploratory analysis of the prepared NS table behind Section 4.1 (W-017).

The computations are pure functions over the table, each returning a frozen record; their
numbers become \\nsEda... LaTeX macros so the text and the decisions table cite generated values.
The evidence they carry: which columns want log10, whether log10(D/M) is a better charge target
than log10 D, whether beta matters at fixed lambda, how full the (beta, lambda) grid is, and
whether rows of one curve are each other's nearest neighbours (why the split groups by curve).
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
from scipy.stats import skew
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from shared.config import NsFigure, NsTable, PaperConfig, load_config
from shared.plots import PlotStyle, anchor_color, apply_style, colormap

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

CURVE = ("beta", "lambda")
Scale = Literal["raw", "log10"]


@dataclass(frozen=True)
class ColumnSummary:
    """One column's distribution on one scale."""

    column: str
    scale: Scale
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float
    skew: float


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


@dataclass(frozen=True)
class Adjacency:
    """Whether rows find their nearest neighbour in their own curve, in standardized inputs."""

    same_curve_share: float
    within_median: float
    across_median: float


@dataclass(frozen=True)
class NeighbourDistances:
    """Per-row nearest distances to the own curve and to another curve, where found."""

    within: np.ndarray
    across: np.ndarray


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


def with_charge_targets(table: pd.DataFrame) -> pd.DataFrame:
    """Add D_over_M, log10_D and log10_D_over_M to a table with M and D."""
    ratio = table["D"] / table["M"]
    return table.assign(
        D_over_M=ratio, log10_D=np.log10(table["D"]), log10_D_over_M=np.log10(ratio)
    )


def _summary(values: np.ndarray, column: str, scale: Scale) -> ColumnSummary:
    """Summarize one column's values (sample std and bias-corrected skew, as pandas)."""
    return ColumnSummary(
        column,
        scale,
        float(np.min(values)),
        float(np.max(values)),
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values, ddof=1)),
        float(skew(values, bias=False)),
    )


def summarize(table: pd.DataFrame, columns: tuple[str, ...]) -> tuple[ColumnSummary, ...]:
    """Summarize each column raw and in log10."""
    return tuple(
        summary
        for column in columns
        for summary in (
            _summary(table[column].to_numpy(dtype=float), column, "raw"),
            _summary(np.log10(table[column].to_numpy(dtype=float)), column, "log10"),
        )
    )


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


def _inputs(table: pd.DataFrame) -> np.ndarray:
    """Standardized (beta, lambda, log10 rho_c); a constant column stays 0."""
    raw = pd.DataFrame(
        {"beta": table["beta"], "lambda": table["lambda"], "rho_c": np.log10(table["rho_c"])}
    )
    std = raw.std(ddof=0).replace(0.0, 1.0)
    return ((raw - raw.mean()) / std).to_numpy()


def _curve_ids(table: pd.DataFrame) -> np.ndarray:
    """One integer per (beta, lambda) curve."""
    return table.groupby(list(CURVE)).ngroup().to_numpy()


def _nearest(table: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Distances to the k nearest other rows (fewer in a small table), and whether each lies in
    the row's own curve."""
    curve = _curve_ids(table)
    neighbours = NearestNeighbors(n_neighbors=min(k, len(table) - 1)).fit(_inputs(table))
    distances, indices = neighbours.kneighbors()
    return distances, curve[indices] == curve[:, None]


def _first(distances: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per row, the distance of the first neighbour the mask selects; rows with none drop out."""
    found = mask.any(axis=1)
    return distances[found, mask[found].argmax(axis=1)]


def neighbour_distances(table: pd.DataFrame, k: int = 16) -> NeighbourDistances:
    """Each row's distance to its nearest same-curve and other-curve row among its k nearest."""
    distances, same = _nearest(table, k)
    return NeighbourDistances(_first(distances, same), _first(distances, ~same))


def curve_adjacency(table: pd.DataFrame, k: int = 16) -> Adjacency:
    """Find each row's nearest neighbour in standardized (beta, lambda, log10 rho_c).

    Among the k nearest rows, the first of the same curve and the first of another give the
    within- and across-curve distances; a row with none of a kind in k contributes no distance.
    """
    distances, same = _nearest(table, k)
    return Adjacency(
        float(same[:, 0].mean()),
        float(np.median(_first(distances, same))),
        float(np.median(_first(distances, ~same))),
    )


def _mae(
    table: pd.DataFrame, target: str, splitter: KFold | GroupKFold, groups: np.ndarray | None
) -> float:
    """Mean absolute 1-NN error of target under one cross-validation splitter; NaN when there
    are fewer groups than folds."""
    if groups is not None and len(np.unique(groups)) < splitter.get_n_splits():
        return float("nan")
    model = KNeighborsRegressor(n_neighbors=1)
    predicted = cross_val_predict(
        model, _inputs(table), table[target].to_numpy(), cv=splitter, groups=groups
    )
    return float(np.abs(predicted - table[target].to_numpy()).mean())


def _rows(folds: int, seed: int) -> KFold:
    """The row-random splitter."""
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _grouped(folds: int, seed: int) -> GroupKFold:
    """The grouped splitter."""
    return GroupKFold(n_splits=folds, shuffle=True, random_state=seed)


def rim_mask(table: pd.DataFrame) -> pd.Series:
    """Mark the rows of rim curves: the grid's outer beta and lambda lines, and each beta's
    largest lambda (the boundary of an empty corner)."""
    beta, lam = table["beta"], table["lambda"]
    outer = beta.isin([beta.min(), beta.max()]) | lam.isin([lam.min(), lam.max()])
    return outer | (lam == table.groupby("beta")["lambda"].transform("max"))


def _rim_mae(table: pd.DataFrame, target: str) -> float:
    """1-NN error on the rim rows of a model fitted on the interior rows; NaN without interior."""
    rim = rim_mask(table).to_numpy()
    if rim.all():
        return float("nan")
    inputs, values = _inputs(table), table[target].to_numpy()
    model = KNeighborsRegressor(n_neighbors=1).fit(inputs[~rim], values[~rim])
    return float(np.abs(model.predict(inputs[rim]) - values[rim]).mean())


def _score(
    table: pd.DataFrame, target: str, strategy: SplitStrategy, folds: int, seed: int
) -> float:
    """The 1-NN error of one strategy."""
    match strategy:
        case "random_rows":
            return _mae(table, target, _rows(folds, seed), None)
        case "curves":
            return _mae(table, target, _grouped(folds, seed), _curve_ids(table))
        case "beta_lines":
            return _mae(table, target, _grouped(folds, seed), table["beta"].to_numpy())
        case "lambda_lines":
            return _mae(table, target, _grouped(folds, seed), table["lambda"].to_numpy())
        case "rim":
            return _rim_mae(table, target)
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


def render_macros(numbers: dict[str, str]) -> str:
    """Render one \\newcommand per number, in the given order."""
    return "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in numbers.items())


def _sci(value: float) -> str:
    """Format a number for LaTeX math mode, e.g. 8.20\\times 10^{-8}."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def number_tex(value: float) -> str:
    """Format a number for LaTeX math mode: plain with three significant digits between 0.01
    and 1000, else scientific."""
    if not 0.01 <= abs(value) < 1000:
        return _sci(value)
    decimals = max(2 - int(np.floor(np.log10(abs(value)))), 0)
    return f"{value:.{decimals}f}"


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
        curve_adjacency(table),
        tuple(s for t in SPLIT_TARGETS for s in split_strategies(table, t, folds, seed)),
    )


def _orders(summary: ColumnSummary) -> float:
    """The orders of magnitude a positive column spans."""
    return float(np.log10(summary.maximum / summary.minimum))


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
        "nsEdaRhocOrders": f"{_orders(summary['rho_c', 'raw']):.1f}",
        "nsEdaDensityGridValues": f"{found.density_grid_values}",
        "nsEdaRowsPerCurveMin": f"{found.rows_per_curve[0]}",
        "nsEdaRowsPerCurveMax": f"{found.rows_per_curve[1]}",
        "nsEdaDMin": _sci(d.minimum),
        "nsEdaDMax": f"{d.maximum:.3f}",
        "nsEdaDOrders": f"{_orders(d):.1f}",
        "nsEdaCorrMLogD": f"{log_d.spearman_m:.2f}",
        "nsEdaCorrMLogDM": f"{log_dm.spearman_m:.2f}",
        "nsEdaCorrLambdaLogD": f"{log_d.pearson_lambda:.2f}",
        "nsEdaCorrLambdaLogDM": f"{log_dm.pearson_lambda:.2f}",
        "nsEdaBetaSharePercent": f"{100 * found.beta_share:.0f}",
        "nsEdaSameCurvePercent": f"{100 * found.adjacency.same_curve_share:.1f}",
        "nsEdaWithinMedian": f"{found.adjacency.within_median:.3f}",
        "nsEdaAcrossMedian": f"{found.adjacency.across_median:.3f}",
        "nsEdaFolds": f"{found.folds}",
        "nsEdaMassMax": f"{found.mass_max:.2f}",
        "nsEdaPileUpWindow": f"{found.window:.2f}",
        "nsEdaPileUpPercent": f"{100 * found.pile_up:.0f}",
    } | {
        f"nsEdaSplit{_camel(s.strategy)}{TARGET_TAGS[s.target]}": _split_error(s)
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


def _booktabs(label: str, caption: str, spec: str, header: str, rows: list[str]) -> str:
    """One booktabs table environment."""
    body = "".join(f"{row} \\\\\n" for row in rows)
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{tab:ns-{label}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _univariate_table(found: Evidence) -> str:
    """Table A: range, orders of magnitude and skew, raw against log10."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    rows = [
        f"{TABLE_LABELS[c]} & ${number_tex(summary[c, 'raw'].minimum)}$ & "
        f"${number_tex(summary[c, 'raw'].maximum)}$ & {_orders(summary[c, 'raw']):.1f} & "
        f"{summary[c, 'raw'].skew:.2f} & {summary[c, 'log10'].skew:.2f}"
        for c in SUMMARY_COLUMNS
    ]
    header = "Variable & min & max & orders of magnitude & skew & skew of $\\log_{10}$"
    caption = (
        "Neutron stars: range and skewness of each continuous variable, raw and in $\\log_{10}$."
    )
    return _booktabs("univariate", caption, "lrrrrr", header, rows)


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
    return _booktabs("charge-correlation", caption, "lrrrr", header, rows)


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
    return _booktabs("split-strategies", caption, "lrr", header, rows)


def tables_tex(found: Evidence, tables: tuple[NsTable, ...]) -> str:
    """One booktabs table per selected table, in order."""
    return "".join(_table(found, table) for table in tables)


def _table(found: Evidence, table: NsTable) -> str:
    """Render one table."""
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


def _camel(name: str) -> str:
    """random_rows -> RandomRows, for LaTeX macro names (letters only)."""
    return "".join(part.capitalize() for part in name.split("_"))


def asset(figure: NsFigure) -> str:
    """The flat asset basename of one figure."""
    return f"41_neutron_stars_{figure}"


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


def figures_tex(figures: tuple[NsFigure, ...]) -> str:
    """One figure environment per selected figure, in order."""
    return "".join(
        "\\begin{figure}[htbp]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{asset(figure)}}}\n"
        f"\\caption{{{caption(figure)}}}\n\\label{{fig:ns-{figure.replace('_', '-')}}}\n"
        "\\end{figure}\n"
        for figure in figures
    )


LABELS = {
    "beta": "$\\beta$",
    "lambda": "$\\lambda$",
    "rho_c": "$\\rho_c$",
    "M": "$M$",
    "D": "$D$",
    "D_over_M": "$D/M$",
}


def _size(height_ratio: float) -> tuple[float, float]:
    """A text-wide figure size whose height is height_ratio times its width."""
    width = float(plt.rcParams["figure.figsize"][0])
    return (width, width * height_ratio)


GRID_VALUES = 60


def _bins(values: np.ndarray, limit: int = GRID_VALUES) -> np.ndarray | int:
    """One bin per value for a variable on a grid (at most limit distinct values), else 40.

    Equal-width bins alias against a parameter grid and draw spikes that are not in the data.
    """
    grid = np.unique(values)
    if len(grid) > limit:
        return 40
    if len(grid) == 1:
        return np.array([grid[0] - 0.5, grid[0] + 0.5])
    middles = (grid[1:] + grid[:-1]) / 2
    return np.concatenate(([2 * grid[0] - middles[0]], middles, [2 * grid[-1] - middles[-1]]))


def _univariate(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Histograms of each variable, raw and in log10."""
    fig, axes = plt.subplots(len(LABELS), 2, figsize=_size(1.6), layout="constrained")
    color = anchor_color(style, "neutron_stars")
    for row, (column, label) in enumerate(LABELS.items()):
        raw = table[column].to_numpy(dtype=float)
        axes[row, 0].hist(raw, bins=_bins(raw), color=color)
        axes[row, 1].hist(np.log10(raw), bins=_bins(np.log10(raw)), color=color)
        axes[row, 0].set_xlabel(label)
        axes[row, 1].set_xlabel(f"$\\log_{{10}}$ {label}")
    return fig


# rho_c sits on a shared grid of a few hundred values; one bar per value avoids aliasing.
DENSITY_GRID_VALUES = 250


def _univariate_continuous(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """Histograms of the continuous variables, raw and in log10, grid-aware bins."""
    fig, axes = plt.subplots(len(SUMMARY_COLUMNS), 2, figsize=_size(1.1), layout="constrained")
    color = anchor_color(style, "neutron_stars")
    for row, column in enumerate(SUMMARY_COLUMNS):
        raw = table[column].to_numpy(dtype=float)
        logged = np.log10(raw)
        # Without edges: a few hundred thin bars would vanish under the style's white edges.
        bars = {"color": color, "linewidth": 0}
        axes[row, 0].hist(raw, bins=_bins(raw, DENSITY_GRID_VALUES), **bars)
        axes[row, 1].hist(logged, bins=_bins(logged, DENSITY_GRID_VALUES), **bars)
        axes[row, 0].set_xlabel(LABELS[column])
        axes[row, 1].set_xlabel(f"$\\log_{{10}}$ {LABELS[column]}")
    return fig


def _mass_max(table: pd.DataFrame, style: PlotStyle, window: float = 0.09) -> Figure:
    """Rows' distance to their curve's M_max, and each curve's M_max over the grid."""
    fig, (left, right) = plt.subplots(1, 2, figsize=_size(0.45), layout="constrained")
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
    fig, axes = plt.subplots(2, 2, figsize=_size(0.9), sharex=True, layout="constrained")
    _charge_row(fig, axes[0, :], table, "lambda", colormap(style.neutron_stars.lambda_cmap))
    _charge_row(fig, axes[1, :], table, "beta", colormap(style.neutron_stars.beta_cmap))
    for ax in axes[1, :]:
        ax.set_xlabel("$M$")
    return fig


def _mass_density(table: pd.DataFrame, style: PlotStyle) -> Figure:
    """M against log10 rho_c, one line per curve coloured by lambda, last points marked."""
    fig, ax = plt.subplots(figsize=_size(0.62), layout="constrained")
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
    fig, ax = plt.subplots(figsize=_size(0.62), layout="constrained")
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
    fig, ax = plt.subplots(figsize=_size(0.5), layout="constrained")
    found = neighbour_distances(table)
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
    """Write the selected figures and tables, the numbers and the figures .tex into the asset
    folder, after removing this section's assets from an earlier run."""
    assert len(argv) == 2, f"usage: eda_neutron_stars.py <table.parquet> <asset dir>, got {argv}"
    source, out = Path(argv[0]), Path(argv[1])
    assert source.is_file(), f"prepared NS table not found: {source}"
    config = config or load_config()
    apply_style(config.plot)
    table = with_charge_targets(pd.read_parquet(source))
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("41_neutron_stars_*"):
        stale.unlink()
    section = config.data_analysis.neutron_stars
    for figure in section.figures:
        fig = draw(figure, table, config.plot)
        fig.savefig(out / f"{asset(figure)}.pdf", dpi=300)
        plt.close(fig)
    found = evidence(table, folds, seed)
    (out / "41_neutron_stars_numbers.tex").write_text(render_macros(numbers(found)))
    (out / "41_neutron_stars_tables.tex").write_text(tables_tex(found, section.tables))
    (out / "41_neutron_stars_figures.tex").write_text(figures_tex(section.figures))
    logger.info(
        "done: NS figures %s, tables %s and numbers -> %s", section.figures, section.tables, out
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
