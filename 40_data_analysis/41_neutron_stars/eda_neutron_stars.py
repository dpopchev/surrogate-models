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
from typing import Literal, assert_never

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from scipy.stats import skew
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from shared.config import NsFigure, PaperConfig, load_config
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


@dataclass(frozen=True)
class Leakage:
    """1-nearest-neighbour error of a target under a row-random and a curve-grouped split."""

    target: str
    mae_rows: float
    mae_groups: float


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


def _mae(table: pd.DataFrame, target: str, splitter: KFold | GroupKFold) -> float:
    """Mean absolute 1-NN error of target under one cross-validation splitter."""
    groups = _curve_ids(table) if isinstance(splitter, GroupKFold) else None
    model = KNeighborsRegressor(n_neighbors=1)
    predicted = cross_val_predict(
        model, _inputs(table), table[target].to_numpy(), cv=splitter, groups=groups
    )
    return float(np.abs(predicted - table[target].to_numpy()).mean())


def split_leakage(table: pd.DataFrame, target: str, folds: int, seed: int) -> Leakage:
    """Compare 1-NN error under row-random KFold and GroupKFold over curves."""
    rows = KFold(n_splits=folds, shuffle=True, random_state=seed)
    groups = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return Leakage(target, _mae(table, target, rows), _mae(table, target, groups))


def render_macros(numbers: dict[str, str]) -> str:
    """Render one \\newcommand per number, in the given order."""
    return "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in numbers.items())


def _sci(value: float) -> str:
    """Format a number for LaTeX math mode, e.g. 8.20\\times 10^{-8}."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def numbers(table: pd.DataFrame, folds: int, seed: int) -> dict[str, str]:
    """Every number Section 4.1 cites, keyed by its \\nsEda macro name."""
    summary = {(s.column, s.scale): s for s in summarize(table, ("rho_c", "D", "D_over_M"))}
    log_d, log_dm = charge_targets(table)
    fill = grid_fill(table)
    adjacency = curve_adjacency(table)
    leak_dm = split_leakage(table, "log10_D_over_M", folds, seed)
    leak_m = split_leakage(table, "M", folds, seed)
    rho = summary["rho_c", "log10"]
    d = summary["D", "raw"]
    return {
        "nsEdaRows": f"{len(table)}",
        "nsEdaCurves": f"{fill.curves}",
        "nsEdaGridCells": f"{fill.cells}",
        "nsEdaGridFillPercent": f"{100 * fill.fill:.1f}",
        "nsEdaRhocLogMean": f"{rho.mean:.3f}",
        "nsEdaRhocLogMedian": f"{rho.median:.3f}",
        "nsEdaRhocLogStd": f"{rho.std:.3f}",
        "nsEdaRhocRawSkew": f"{summary['rho_c', 'raw'].skew:.2f}",
        "nsEdaRhocLogSkew": f"{rho.skew:.2f}",
        "nsEdaDMin": _sci(d.minimum),
        "nsEdaDMax": f"{d.maximum:.3f}",
        "nsEdaDDecades": f"{np.log10(d.maximum / d.minimum):.1f}",
        "nsEdaCorrMLogD": f"{log_d.spearman_m:.2f}",
        "nsEdaCorrMLogDM": f"{log_dm.spearman_m:.2f}",
        "nsEdaCorrLambdaLogD": f"{log_d.pearson_lambda:.2f}",
        "nsEdaCorrLambdaLogDM": f"{log_dm.pearson_lambda:.2f}",
        "nsEdaBetaSharePercent": f"{100 * beta_share_at_fixed_lambda(table, 'log10_D_over_M'):.0f}",
        "nsEdaSameCurvePercent": f"{100 * adjacency.same_curve_share:.1f}",
        "nsEdaWithinMedian": f"{adjacency.within_median:.3f}",
        "nsEdaAcrossMedian": f"{adjacency.across_median:.3f}",
        "nsEdaFolds": f"{folds}",
        "nsEdaLeakRowsDM": f"{leak_dm.mae_rows:.3f}",
        "nsEdaLeakGroupsDM": f"{leak_dm.mae_groups:.3f}",
        "nsEdaLeakRowsM": f"{leak_m.mae_rows:.4f}",
        "nsEdaLeakGroupsM": f"{leak_m.mae_groups:.4f}",
    }


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


def _bins(values: np.ndarray) -> np.ndarray | int:
    """One bin per value for a grid parameter (at most GRID_VALUES distinct values), else 40.

    Equal-width bins alias against a parameter grid and draw spikes that are not in the data.
    """
    grid = np.unique(values)
    if len(grid) > GRID_VALUES:
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
        case _:
            assert_never(figure)


# --- shell ------------------------------------------------------------------------------------


def main(
    argv: list[str], config: PaperConfig | None = None, folds: int = 5, seed: int = 20261005
) -> None:
    """Write the selected figures, the numbers and the figures .tex into the asset folder."""
    assert len(argv) == 2, f"usage: eda_neutron_stars.py <table.parquet> <asset dir>, got {argv}"
    source, out = Path(argv[0]), Path(argv[1])
    assert source.is_file(), f"prepared NS table not found: {source}"
    config = config or load_config()
    apply_style(config.plot)
    table = with_charge_targets(pd.read_parquet(source))
    out.mkdir(parents=True, exist_ok=True)
    figures = config.data_analysis.neutron_stars.figures
    for figure in figures:
        fig = draw(figure, table, config.plot)
        fig.savefig(out / f"{asset(figure)}.pdf", dpi=300)
        plt.close(fig)
    (out / "41_neutron_stars_numbers.tex").write_text(render_macros(numbers(table, folds, seed)))
    (out / "41_neutron_stars_figures.tex").write_text(figures_tex(figures))
    logger.info("done: %d NS figures %s and their numbers -> %s", len(figures), figures, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
