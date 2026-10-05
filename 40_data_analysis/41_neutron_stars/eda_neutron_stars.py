"""Exploratory analysis of the prepared NS table behind Section 4.1 (W-017).

The computations are pure functions over the table, each returning a frozen record; their
numbers become \\nsEda... LaTeX macros so the text and the decisions table cite generated values.
The evidence they carry: which columns want log10, whether log10(D/M) is a better charge target
than log10 D, whether beta matters at fixed lambda, how full the (beta, lambda) grid is, and
whether rows of one curve are each other's nearest neighbours (why the split groups by curve).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

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


def curve_adjacency(table: pd.DataFrame, k: int = 16) -> Adjacency:
    """Find each row's nearest neighbour in standardized (beta, lambda, log10 rho_c).

    Among the k nearest rows, the first of the same curve and the first of another give the
    within- and across-curve distances; a row with none of a kind in k contributes no distance.
    """
    curve = _curve_ids(table)
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(_inputs(table)).kneighbors()
    same = curve[indices] == curve[:, None]
    first = np.where(same.any(axis=1), same.argmax(axis=1), -1)
    other = np.where((~same).any(axis=1), (~same).argmax(axis=1), -1)
    rows = np.arange(len(curve))
    return Adjacency(
        float(same[:, 0].mean()),
        float(np.median(distances[rows[first >= 0], first[first >= 0]])),
        float(np.median(distances[rows[other >= 0], other[other >= 0]])),
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
