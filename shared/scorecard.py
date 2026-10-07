"""One scorecard for every candidate surrogate (W-065).

A candidate is scored by its relative error on the rows it predicts -- the error in M for the
mass, the error in D rebuilt with the true M for the charge -- summarized per zone of rows
(interior test curves, the extrapolation curves, ...) and per decade of D, as significant
figures, -log10 of the relative error; beside it the ripple of its residual across the grid of
curve parameters, which a smooth surrogate keeps near zero.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

import numpy as np

from shared.ceilings import Spread, spread
from shared.design import Target

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Scorecard:
    """The spread of the relative errors per zone and per decade of D."""

    zones: dict[str, Spread]
    decades: dict[int, Spread]


# --- pure functions ---------------------------------------------------------------------------


def relative_errors(y_true: np.ndarray, y_pred: np.ndarray, target: Target) -> np.ndarray:
    """Per row, the relative error of M, or of D rebuilt with the true M from Y = log10(D/M)
    (the mass cancels: D_pred / D_true = 10^(Y_pred - Y_true))."""
    match target:
        case "mass":
            return np.abs((y_pred - y_true) / y_true)
        case "charge":
            return np.abs(10.0 ** (y_pred - y_true) - 1.0)
        case _:
            assert_never(target)


def significant_figures(error: float) -> float:
    """The significant figures a relative error keeps, -log10 of it."""
    return float(-np.log10(error))


def scorecard(
    errors: np.ndarray, zones: Mapping[str, np.ndarray], charge: np.ndarray | None
) -> Scorecard:
    """The spread of the errors in each zone (a boolean mask of rows) and, when the true
    charge D of the rows is given, per decade of D."""
    decades: dict[int, Spread] = {}
    if charge is not None:
        decade = np.floor(np.log10(charge)).astype(int)
        decades = {int(d): spread(errors[decade == d]) for d in np.unique(decade)}
    return Scorecard({name: spread(errors[mask]) for name, mask in zones.items()}, decades)


def ripple(residual: np.ndarray, across: np.ndarray, fixed: np.ndarray) -> float:
    """The root mean square of the second difference of the residual across the grid values
    `across`, taken within each group of rows sharing `fixed` (the other inputs); zero for a
    residual linear in `across`."""
    second: list[np.ndarray] = []
    for group in np.unique(fixed):
        rows = np.flatnonzero(fixed == group)
        ordered = residual[rows[np.argsort(across[rows])]]
        if len(ordered) >= 3:
            second.append(np.diff(ordered, 2))
    if not second:
        return float("nan")
    return float(np.sqrt(np.mean(np.concatenate(second) ** 2)))
