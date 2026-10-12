"""The dataset-agnostic core of the exploratory analyses in Section 3 (W-018).

Each dataset module (eda_neutron_stars.py, eda_black_holes.py) names its inputs and its curve
key columns in a CurveSpace; the functions here summarize columns, measure how rows of one curve
neighbour each other, score 1-nearest-neighbour probes under a split, and render LaTeX numbers,
macros, tables and histogram bins for the generated assets.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, assert_never

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import skew
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from shared.config import TableLayout, UnivariateVariant

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


@dataclass(frozen=True)
class RowsPerCurve:
    """The fewest and the most rows a curve holds."""

    minimum: int
    maximum: int


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


def neighbour_distances(table: pd.DataFrame, space: CurveSpace, k: int = 16) -> NeighbourDistances:
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


_PLAIN = re.compile(r"^\$?(-?\d+(?:\.\d+)?)\$?$")
_POWER = re.compile(r"^\$(-?\d+(?:\.\d+)?)\\times 10\^\{(-?\d+)\}\$$")
_PLUS_MINUS = re.compile(r"^\$(-?\d+(?:\.\d+)?)_\{\\pm (\d+(?:\.\d+)?)\}\$$")
_MULTICOLUMN = re.compile(r"^\\multicolumn\{(\d+)\}")
_CELL_SEPARATOR = re.compile(r"(?<!\\)&")


def _group(spec: str, at: int) -> tuple[str, int]:
    """The brace group opening at spec[at], nested braces included, and the index after it."""
    depth = 0
    for end in range(at, len(spec)):
        depth += {"{": 1, "}": -1}.get(spec[end], 0)
        if depth == 0:
            return spec[at + 1 : end], end + 1
    raise ValueError(f"unbalanced braces in tabular spec {spec!r}")


def _spec_tokens(spec: str) -> list[tuple[str, str]]:
    """The tabular spec as (kind, text) tokens: a column letter, a repeat *{n}{x} as ("*", n
    and x), a column with a width as ("p", its text), anything else (an @{} or !{} separator,
    a rule) as ("", its text)."""
    tokens, at = [], 0
    while at < len(spec):
        char = spec[at]
        if char in "@!":
            _, end = _group(spec, at + 1)
            tokens.append(("", spec[at:end]))
        elif char == "*":
            count, middle = _group(spec, at + 1)
            body, end = _group(spec, middle)
            tokens.append(("*", f"{count}:{body}"))
        elif char in "pmb":
            _, end = _group(spec, at + 1)
            tokens.append(("p", spec[at:end]))
        elif char in "lcrS":
            end = at + 1
            tokens.append((char, char))
        else:
            end = at + 1
            tokens.append(("", char))
        at = end
    return tokens


def _columns(spec: str) -> str:
    """The column letters of a tabular spec, one per column: a *{n}{x} repeat expanded, an
    @{} or !{} separator and a rule dropped, a p{}, m{} or b{} column read as p.

    >>> _columns("@{}ll*{4}{r}@{}"), _columns("lp{3cm}|r")
    ('llrrrr', 'lpr')
    >>> _columns("@{}l@{\\\\hspace{8pt}}l*{6}{@{\\\\hspace{8pt}}r}@{}")
    'llrrrrrr'
    """
    letters = []
    for kind, text in _spec_tokens(spec):
        if kind == "*":
            count, body = text.split(":", 1)
            letters.append(_columns(body) * int(count))
        elif kind:
            letters.append(kind)
    return "".join(letters)


def si_cell(cell: str) -> str:
    """A cell as siunitx reads it in an S column: a plain number, a power of ten or a value
    with its spread as numbers, anything else braced so it is set as text.

    >>> si_cell("$5.89\\\\times 10^{14}$"), si_cell("$1.96_{\\\\pm 0.02}$"), si_cell("0.57")
    ('5.89e14', '1.96 +- 0.02', '0.57')
    >>> si_cell("$1.02_{10^{3}}$")
    '{$1.02_{10^{3}}$}'
    """
    text = cell.strip()
    if found := _PLAIN.match(text):
        return found[1]
    if found := _POWER.match(text):
        return f"{found[1]}e{found[2]}"
    if found := _PLUS_MINUS.match(text):
        return f"{found[1]} +- {found[2]}"
    return f"{{{text}}}" if text else ""


def _aligned_spec(spec: str) -> str:
    """The tabular spec with each r column an S column, inside a *{n}{x} repeat too.

    >>> _aligned_spec("lrrr"), _aligned_spec("@{}ll*{4}{r}@{}")
    ('lSSS', '@{}ll*{4}{S}@{}')
    >>> _aligned_spec("l*{2}{@{\\\\hspace{8pt}}r}")
    'l*{2}{@{\\\\hspace{8pt}}S}'
    """
    parts = []
    for kind, text in _spec_tokens(spec):
        if kind == "*":
            count, body = text.split(":", 1)
            parts.append(f"*{{{count}}}{{{_aligned_spec(body)}}}")
        else:
            parts.append("S" if kind == "r" else text)
    return "".join(parts)


def _aligned_line(line: str, columns: str, body: bool) -> str:
    """One header or body line for S columns: a body cell of an S column read as a number, a
    header cell braced as text; a multicolumn cell spans its columns untouched. Cells split on
    every unescaped &, so an empty cell between two ampersands keeps its column."""
    cells, at = [], 0
    for cell in (part.strip() for part in _CELL_SEPARATOR.split(line)):
        if spanned := _MULTICOLUMN.match(cell):
            cells.append(cell)
            at += int(spanned[1])
            continue
        if at < len(columns) and columns[at] == "S":
            cells.append(si_cell(cell) if body else (f"{{{cell}}}" if cell else ""))
        else:
            cells.append(cell)
        at += 1
    return " & ".join(cells)


_SI_NUMBER = re.compile(r"^(-?)(\d+)(?:\.(\d+))?(?:e(-?)(\d+))?(?: \+- (\d+(?:\.\d+)?))?$")


def _table_format(cells: list[str]) -> str:
    """The siunitx table-format reserving the widest sign, integer, decimal, uncertainty and
    exponent parts of an S column's si_cell forms; braced text cells and blanks are skipped,
    and a column without a number gets none.

    >>> _table_format(["5.89e14", "3.82e-8", "{--}"])
    '1.2e-2'
    >>> _table_format(["1.96 +- 0.02", "0.4 +- 0.13"])
    '1.2(2)'
    >>> _table_format(["{--}", ""])
    ''
    """
    found = [m for m in map(_SI_NUMBER.match, cells) if m]
    if not found:
        return ""
    sign = "-" if any(m[1] for m in found) else ""
    integer = max(len(m[2]) for m in found)
    decimals = max(len(m[3] or "") for m in found)
    shape = f"{sign}{integer}" + (f".{decimals}" if decimals else "")
    spreads = [round(float(m[6]) * 10**decimals) for m in found if m[6]]
    if spreads:
        shape += f"({max(len(str(s)) for s in spreads)})"
    exponents = [m for m in found if m[5]]
    if exponents:
        exponent_sign = "-" if any(m[4] for m in exponents) else ""
        shape += f"e{exponent_sign}{max(len(m[5]) for m in exponents)}"
    return shape


def _column_cells(rows: list[str], columns: str) -> list[list[str]]:
    """Each column's cells over the body rows, a multicolumn cell skipped."""
    found: list[list[str]] = [[] for _ in columns]
    for row in rows:
        at = 0
        for cell in (part.strip() for part in _CELL_SEPARATOR.split(row)):
            if spanned := _MULTICOLUMN.match(cell):
                at += int(spanned[1])
                continue
            if at < len(columns):
                found[at].append(cell)
            at += 1
    return found


def _formatted_spec(spec: str, formats: list[str]) -> str:
    """The spec with every repeat expanded and the k-th S column given formats[k] as its
    table-format (a plain S for an empty one).

    >>> _formatted_spec("l*{2}{@{}S}", ["1.2", ""])
    'l@{}S[table-format=1.2]@{}S'
    """
    parts, pending = [], iter(formats)

    def expand(text: str) -> None:
        for kind, token in _spec_tokens(text):
            if kind == "*":
                count, body = token.split(":", 1)
                for _ in range(int(count)):
                    expand(body)
            elif kind == "S":
                shape = next(pending)
                parts.append(f"S[table-format={shape}]" if shape else "S")
            else:
                parts.append(token)

    expand(spec)
    return "".join(parts)


def booktabs(
    label: str,
    caption: str,
    spec: str,
    header: str,
    rows: list[str],
    layout: TableLayout = "small",
) -> str:
    """One booktabs table environment, labelled tab:<label>, placed here or atop a page, in the
    small font so a table of several numeric columns fits the text width; aligned, each right
    column becomes an siunitx S column aligning its numbers on the decimal marker."""
    match layout:
        case "small":
            pass
        case "aligned":
            spec = _aligned_spec(spec)
            columns = _columns(spec)
            header = " \\\\\n".join(
                _aligned_line(line, columns, body=False) for line in header.split(" \\\\\n")
            )
            rows = [_aligned_line(row, columns, body=True) for row in rows]
            cells = zip(_column_cells(rows, columns), columns, strict=True)
            spec = _formatted_spec(spec, [_table_format(c) for c, kind in cells if kind == "S"])
        case _:
            assert_never(layout)
    body = "".join(f"{row} \\\\\n" for row in rows)
    return (
        "\\begin{table}[!htb]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def grid_bins(values: np.ndarray, limit: int) -> np.ndarray | int:
    """One bin per value for a variable on a grid (at most limit distinct values, a plot
    style's grid_values or univariate_grid_values), else 40.

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


def univariate_caption(subject: str, variant: UnivariateVariant, tail: str = "") -> str:
    """The caption of the univariate figure of one dataset (subject) in one variant."""
    match variant:
        case "histograms":
            layout = "raw (left) and $\\log_{10}$ (right)"
        case "log_counts":
            layout = "raw (left) and $\\log_{10}$ (right), the row counts on a log scale"
        case "transposed":
            layout = "raw (top) and $\\log_{10}$ (bottom), a column per variable"
        case _:
            assert_never(variant)
    return f"{subject}: distributions of the continuous variables, {layout}{tail}."


def univariate_figure(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    labels: Mapping[str, str],
    color: tuple[float, float, float],
    grid_values: int,
    variant: UnivariateVariant,
) -> Figure:
    """Histograms of the continuous columns, raw and in log10, one bar per value up to
    grid_values distinct values: a row per column (histograms), the same with log-y row counts
    (log_counts), or the raw and the log10 as two rows at half the height (transposed)."""
    match variant:
        case "histograms" | "log_counts":
            fig, axes = plt.subplots(
                len(columns), 2, figsize=text_width_size(1.1), layout="constrained"
            )
            pairs = [(axes[row, 0], axes[row, 1]) for row in range(len(columns))]
        case "transposed":
            fig, axes = plt.subplots(
                2, len(columns), figsize=text_width_size(0.55), layout="constrained"
            )
            pairs = [(axes[0, col], axes[1, col]) for col in range(len(columns))]
        case _:
            assert_never(variant)
    # Without edges: a few hundred thin bars would vanish under the style's white edges.
    bars = {"color": color, "linewidth": 0}
    for (raw_axes, log_axes), column in zip(pairs, columns, strict=True):
        raw = table[column].to_numpy(dtype=float)
        logged = np.log10(raw)
        raw_axes.hist(raw, bins=grid_bins(raw, grid_values), **bars)
        log_axes.hist(logged, bins=grid_bins(logged, grid_values), **bars)
        raw_axes.set_xlabel(labels[column])
        log_axes.set_xlabel(f"$\\log_{{10}}$ {labels[column]}")
        if variant == "log_counts":
            for axes_ in (raw_axes, log_axes):
                axes_.set_yscale("log")
                axes_.set_ylabel("rows")
    return fig
