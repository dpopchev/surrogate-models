"""The dataset-agnostic core of the exploratory analyses in Section 4 (W-018).

Each dataset module (eda_neutron_stars.py, eda_black_holes.py) names its inputs and its curve
key columns in a CurveSpace; the functions here summarize columns, measure how rows of one curve
neighbour each other, score 1-nearest-neighbour probes under a split, and render LaTeX numbers,
macros, tables and histogram bins for the generated assets.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

# --- vocabulary and types ---------------------------------------------------------------------

Scale = Literal["raw", "log10"]


@dataclass(frozen=True)
class CurveSpace:
    """The inputs a probe standardizes (each raw or in log10) and the columns naming a curve."""

    inputs: tuple[tuple[str, Scale], ...]
    curve: tuple[str, ...]


def make_curve_space(inputs: Mapping[str, Scale], curve: tuple[str, ...]) -> CurveSpace:
    """Build a curve space; raise ValueError when a curve column is not an input."""
    if missing := [column for column in curve if column not in inputs]:
        raise ValueError(f"curve columns {missing} are not among the inputs")
    return CurveSpace(tuple(inputs.items()), curve)


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


# --- columns ----------------------------------------------------------------------------------


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


def orders_of_magnitude(summary: ColumnSummary) -> float:
    """The orders of magnitude a positive column spans."""
    return float(np.log10(summary.maximum / summary.minimum))


# --- curves and neighbours --------------------------------------------------------------------


def standardized(table: pd.DataFrame, space: CurveSpace) -> np.ndarray:
    """The inputs of the space, each raw or in log10, standardized; a constant one stays 0."""
    raw = pd.DataFrame(
        {
            column: np.log10(table[column]) if scale == "log10" else table[column]
            for column, scale in space.inputs
        }
    )
    std = raw.std(ddof=0).replace(0.0, 1.0)
    return ((raw - raw.mean()) / std).to_numpy()


def curve_ids(table: pd.DataFrame, space: CurveSpace) -> np.ndarray:
    """One integer per curve of the space."""
    return table.groupby(list(space.curve)).ngroup().to_numpy()


def _nearest(table: pd.DataFrame, space: CurveSpace, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Distances to the k nearest other rows (fewer in a small table), and whether each lies in
    the row's own curve."""
    curve = curve_ids(table, space)
    neighbours = NearestNeighbors(n_neighbors=min(k, len(table) - 1))
    distances, indices = neighbours.fit(standardized(table, space)).kneighbors()
    return distances, curve[indices] == curve[:, None]


def _first(distances: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per row, the distance of the first neighbour the mask selects; rows with none drop out."""
    found = mask.any(axis=1)
    return distances[found, mask[found].argmax(axis=1)]


def _across(
    table: pd.DataFrame, space: CurveSpace, distances: np.ndarray, same: np.ndarray
) -> np.ndarray:
    """Every row's distance to the nearest row of another curve, in row order.

    Taken from the k nearest where one of them lies on another curve; otherwise found exactly
    by searching the other curves' rows (dense curves: all k nearest share the row's curve).
    """
    inputs, curve = standardized(table, space), curve_ids(table, space)
    across = np.full(len(curve), np.nan)
    found = (~same).any(axis=1)
    across[found] = distances[found, (~same)[found].argmax(axis=1)]
    for own in np.unique(curve[~found]):
        rows, others = (curve == own) & ~found, curve != own
        if others.any():
            nearest = NearestNeighbors(n_neighbors=1).fit(inputs[others])
            across[rows] = nearest.kneighbors(inputs[rows])[0][:, 0]
    return across[~np.isnan(across)]


def neighbour_distances(
    table: pd.DataFrame, space: CurveSpace, k: int = 16
) -> NeighbourDistances:
    """Each row's distance to its nearest same-curve row among its k nearest, and to the
    nearest row of another curve."""
    distances, same = _nearest(table, space, k)
    return NeighbourDistances(_first(distances, same), _across(table, space, distances, same))


def curve_adjacency(table: pd.DataFrame, space: CurveSpace, k: int = 16) -> Adjacency:
    """Find each row's nearest neighbour in the standardized inputs of the space.

    Among the k nearest rows, the first of the same curve gives the within-curve distance (a
    row with none in k contributes none); the across-curve distance is exact for every row.
    """
    distances, same = _nearest(table, space, k)
    return Adjacency(
        float(same[:, 0].mean()),
        float(np.median(_first(distances, same))),
        float(np.median(_across(table, space, distances, same))),
    )


def rim_mask(table: pd.DataFrame, outer: str, inner: str) -> pd.Series:
    """Mark the rows on the rim of an (outer, inner) parameter grid: the grid's smallest and
    largest values of either column, and each outer value's largest inner value (the boundary
    of an empty corner)."""
    first, second = table[outer], table[inner]
    edges = first.isin([first.min(), first.max()]) | second.isin([second.min(), second.max()])
    return edges | (second == table.groupby(outer)[inner].transform("max"))


# --- 1-nearest-neighbour probes under a split -------------------------------------------------


def row_splitter(folds: int, seed: int) -> KFold:
    """The row-random splitter."""
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def group_splitter(folds: int, seed: int) -> GroupKFold:
    """The grouped splitter."""
    return GroupKFold(n_splits=folds, shuffle=True, random_state=seed)


def kfold_mae(
    table: pd.DataFrame,
    space: CurveSpace,
    target: str,
    splitter: KFold | GroupKFold,
    groups: np.ndarray | None,
) -> float:
    """Mean absolute 1-NN error of target under one cross-validation splitter; NaN when there
    are fewer groups than folds."""
    if groups is not None and len(np.unique(groups)) < splitter.get_n_splits():
        return float("nan")
    model = KNeighborsRegressor(n_neighbors=1)
    predicted = cross_val_predict(
        model, standardized(table, space), table[target].to_numpy(), cv=splitter, groups=groups
    )
    return float(np.abs(predicted - table[target].to_numpy()).mean())


def holdout_mae(table: pd.DataFrame, space: CurveSpace, target: str, test: np.ndarray) -> float:
    """1-NN error on the test rows of a model fitted on the others; NaN when either is empty."""
    if test.all() or not test.any():
        return float("nan")
    inputs, values = standardized(table, space), table[target].to_numpy()
    model = KNeighborsRegressor(n_neighbors=1).fit(inputs[~test], values[~test])
    return float(np.abs(model.predict(inputs[test]) - values[test]).mean())


# --- LaTeX and figure rendering ---------------------------------------------------------------


def sci_tex(value: float) -> str:
    """Format a number for LaTeX math mode, e.g. 8.20\\times 10^{-8}."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def number_tex(value: float) -> str:
    """Format a number for LaTeX math mode: 0 as 0, plain with three significant digits between
    0.01 and 1000, else scientific."""
    if value == 0:
        return "0"
    if not 0.01 <= abs(value) < 1000:
        return sci_tex(value)
    decimals = max(2 - int(np.floor(np.log10(abs(value)))), 0)
    return f"{value:.{decimals}f}"


def render_macros(numbers: dict[str, str]) -> str:
    """Render one \\newcommand per number, in the given order."""
    return "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in numbers.items())


def camel(name: str) -> str:
    """random_rows -> RandomRows, for LaTeX macro names (letters only)."""
    return "".join(part.capitalize() for part in name.split("_"))


def booktabs(label: str, caption: str, spec: str, header: str, rows: list[str]) -> str:
    """One booktabs table environment, labelled tab:<label>, placed here or atop a page, in the
    small font so a table of several numeric columns fits the text width."""
    body = "".join(f"{row} \\\\\n" for row in rows)
    return (
        "\\begin{table}[!htb]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


GRID_VALUES = 60


def grid_bins(values: np.ndarray, limit: int = GRID_VALUES) -> np.ndarray | int:
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


def text_width_size(height_ratio: float) -> tuple[float, float]:
    """A figure size as wide as the text (the style's figure width), height_ratio times as high."""
    width = float(plt.rcParams["figure.figsize"][0])
    return (width, width * height_ratio)
