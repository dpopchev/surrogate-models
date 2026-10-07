"""The relative error the data themselves allow a surrogate (W-064).

A curve is the rows of a table sharing its curve keys, ordered along one coordinate (NS: beta,
lambda along log10 rho_c; BH: beta along r_h). Three bounds, each a relative error on a
positive target: the noise of the target along a curve; how well a spline through every other
row of a curve predicts the rows between (the along-curve ceiling); and how well the curves next
to a held-out curve predict it (the across-curve ceiling), the bound a surrogate meets on
held-out curves, since the curves are sampled far more densely along than across.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Spread:
    """The median, 95th percentile and maximum of a set of relative errors."""

    median: float
    p95: float
    max: float


# --- pure functions ---------------------------------------------------------------------------


def spread(errors: np.ndarray) -> Spread:
    """The spread of the finite errors."""
    finite = errors[np.isfinite(errors)]
    return Spread(float(np.median(finite)), float(np.quantile(finite, 0.95)), float(finite.max()))


def noise_proxy(table: pd.DataFrame, keys: Sequence[str], along: str, target: str) -> Spread:
    """Per curve, the median relative fourth difference of the target along the curve over
    sqrt(70); spread over the curves. Zero for a target whose logarithm is cubic along it."""
    per_curve = [
        float(np.median(np.abs(np.diff(np.log(curve[target].to_numpy(float)), 4))))
        for curve in _curves(table, keys, along)
        if len(curve) > 4
    ]
    return spread(np.array(per_curve) / np.sqrt(70.0))


def _curves(table: pd.DataFrame, keys: Sequence[str], along: str) -> list[pd.DataFrame]:
    """The curves of a table in key order, each sorted along its coordinate."""
    ordered = table.sort_values([*keys, along])
    return [curve for _, curve in ordered.groupby(list(keys))]


def along_curve_errors(
    table: pd.DataFrame, keys: Sequence[str], along: str, target: str
) -> np.ndarray:
    """The relative errors on the odd rows of each curve of a cubic spline in log target
    through its even rows."""
    errors = []
    for curve in _curves(table, keys, along):
        x, y = curve[along].to_numpy(float), curve[target].to_numpy(float)
        if len(x) < 8:
            continue
        spline = CubicSpline(x[::2], np.log(y[::2]))
        odd = slice(1, len(x) - 1, 2)
        errors.append(np.abs(np.expm1(spline(x[odd]) - np.log(y[odd]))))
    return np.concatenate(errors)


def across_curve_errors(
    table: pd.DataFrame,
    fixed: Sequence[str],
    across: str,
    along: str,
    target: str,
    neighbours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave each interior curve out of its group of curves sharing the fixed keys; predict
    its rows from the `neighbours` curves on each side along `across` (linear for one, cubic
    for two), each neighbour interpolated along `along` first. Returns the relative errors and
    the true target on the scored rows."""
    errors: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    groups = [table] if not fixed else [group for _, group in table.groupby(list(fixed))]
    for group in groups:
        curves = _curves(group, [across], along)
        for i in range(neighbours, len(curves) - neighbours):
            window = curves[i - neighbours : i + neighbours + 1]
            scored = _held_out(window, neighbours, across, along, target)
            if scored is not None:
                errors.append(scored[0])
                truths.append(scored[1])
    if not errors:
        return np.array([]), np.array([])
    return np.concatenate(errors), np.concatenate(truths)


def _held_out(
    window: list[pd.DataFrame], held: int, across: str, along: str, target: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """The relative errors and true target of the curve at index held of the window, predicted
    from the other curves of the window; None when their ranges along it leave no rows."""
    curve, others = window[held], window[:held] + window[held + 1 :]
    low = max(other[along].min() for other in others)
    high = min(other[along].max() for other in others)
    rows = curve[(curve[along] >= low) & (curve[along] <= high)]
    if len(rows) < 3:
        return None
    x, truth = rows[along].to_numpy(float), rows[target].to_numpy(float)
    at = np.array([other[across].iloc[0] for other in others], dtype=float)
    logs = np.array(
        [
            PchipInterpolator(other[along].to_numpy(float), np.log(other[target].to_numpy(float)))(
                x
            )
            for other in others
        ]
    )
    position = float(curve[across].iloc[0])
    if len(others) == 2:
        weight = (position - at[0]) / (at[1] - at[0])
        predicted = logs[0] * (1.0 - weight) + logs[1] * weight
    else:
        predicted = CubicSpline(at, logs, axis=0)(position)
    return np.abs(np.expm1(predicted - np.log(truth))), truth


def per_decade(target: np.ndarray, errors: np.ndarray) -> dict[int, float]:
    """The median error per decade of the target, keyed by floor(log10 target)."""
    decades = np.floor(np.log10(target)).astype(int)
    return {int(d): float(np.median(errors[decades == d])) for d in np.unique(decades)}
