"""The relative error the data themselves allow a surrogate (W-064).

A curve is the rows of a table sharing its curve keys, ordered along one coordinate (NS: beta,
lambda along log10 rho_c; BH: beta along r_h). Three bounds, each a relative error on a
positive target: the noise of the target along a curve; how well a spline through every other
row of a curve predicts the rows between (the along-curve ceiling); and how well the curves next
to a held-out curve predict it (the across-curve ceiling), the bound a surrogate meets on
held-out curves, since the curves are sampled far more densely along than across.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline, PchipInterpolator

from shared.eda import booktabs, sci_tex

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Spread:
    """The median, 95th percentile, maximum and mean of a set of relative errors."""

    median: float
    p95: float
    max: float
    mean: float


@dataclass(frozen=True)
class Scored:
    """The rows of held-out curves: their relative errors, true target and position along."""

    errors: np.ndarray
    truth: np.ndarray
    position: np.ndarray


@dataclass(frozen=True)
class Profile:
    """The median and 95th percentile of the errors in equal-count bins along the curves."""

    centers: tuple[float, ...]
    median: tuple[float, ...]
    p95: tuple[float, ...]


@dataclass(frozen=True)
class Ceilings:
    """The bounds of one dataset: per target the noise and the along-curve ceiling; per target
    and curve key varied the across-curve ceiling (cubic, two curves on each side) and its
    profile along the curves; per curve key varied the decade target's ceiling by decade."""

    noise: dict[str, Spread]
    along: dict[str, Spread]
    across: dict[tuple[str, str], Spread]
    profiles: dict[tuple[str, str], Profile]
    decades: dict[str, dict[int, Spread]]


PROFILE_BINS = 20


# --- pure functions ---------------------------------------------------------------------------


def measure(
    table: pd.DataFrame,
    keys: Sequence[str],
    along: str,
    targets: Sequence[str],
    decade_target: str,
) -> Ceilings:
    """The ceilings of a table whose curves are keyed by keys and run along `along`; each key
    in turn is varied across curves while the others stay fixed."""
    across: dict[tuple[str, str], Spread] = {}
    profiles: dict[tuple[str, str], Profile] = {}
    decades: dict[str, dict[int, Spread]] = {}
    for target in targets:
        for key in keys:
            fixed = [k for k in keys if k != key]
            scored = across_curve_errors(table, fixed, key, along, target, neighbours=2)
            across[target, key] = spread(scored.errors)
            profiles[target, key] = profile(scored.position, scored.errors, PROFILE_BINS)
            if target == decade_target:
                decade = np.floor(np.log10(scored.truth)).astype(int)
                decades[key] = {
                    int(d): spread(scored.errors[decade == d]) for d in np.unique(decade)
                }
    return Ceilings(
        noise={t: noise_proxy(table, keys, along, t) for t in targets},
        along={t: spread(along_curve_errors(table, keys, along, t)) for t in targets},
        across=across,
        profiles=profiles,
        decades=decades,
    )


def ceiling_macros(prefix: str, found: Ceilings, tags: Mapping[str, str]) -> dict[str, str]:
    """The \\<prefix>Ceil... macros of the ceilings; tags name each target and curve key."""
    macros = {
        f"{prefix}Ceil{tags[target]}Across{tags[key]}": _sci(found_spread.p95)
        for (target, key), found_spread in found.across.items()
    }
    for target in {target for target, _ in found.across}:
        macros[f"{prefix}Ceil{tags[target]}Figures"] = f"{reachable_figures(found, target):.1f}"
    for target, noise in found.noise.items():
        macros[f"{prefix}Ceil{tags[target]}Noise"] = _sci(noise.median)
    for target, along in found.along.items():
        macros[f"{prefix}Ceil{tags[target]}Along"] = _sci(along.p95)
    return macros


def ceiling_table(label: str, caption: str, found: Ceilings, labels: Mapping[str, str]) -> str:
    """A table of the ceilings: per target the median noise, the along-curve p95, the
    across-curve p95 per curve key and the significant figures they allow."""
    keys = list(dict.fromkeys(key for _, key in found.across))
    rows = [
        " & ".join(
            [
                labels[target],
                f"${_sci(found.noise[target].median)}$",
                f"${_sci(found.along[target].p95)}$",
                *(f"${_sci(found.across[target, key].p95)}$" for key in keys),
                f"{reachable_figures(found, target):.1f}",
            ]
        )
        for target in found.noise
    ]
    across = " & ".join(f"across {labels[key]}" for key in keys)
    header = f"Target & noise & along & {across} & figures"
    return booktabs(label, caption, "l" + "r" * (len(keys) + 3), header, rows)


def decade_table(label: str, caption: str, found: Ceilings, labels: Mapping[str, str]) -> str:
    """A table of the decade target's across-curve ceiling: per decade the median per curve
    key varied and the significant figures the worse of them allows."""
    keys = list(found.decades)
    decades = sorted({d for per_key in found.decades.values() for d in per_key})
    rows = []
    for decade in decades:
        medians = [found.decades[key].get(decade) for key in keys]
        cells = [f"${_sci(s.median)}$" if s else "--" for s in medians]
        worst = max(s.median for s in medians if s)
        rows.append(" & ".join([f"$10^{{{decade}}}$", *cells, f"{-np.log10(worst):.1f}"]))
    across = " & ".join(f"across {labels[key]}" for key in keys)
    header = f"Decade & {across} & figures"
    return booktabs(label, caption, "l" + "r" * (len(keys) + 1), header, rows)


def ceiling_figure(
    found: Ceilings, labels: Mapping[str, str], along_label: str, colors: Mapping[str, Any]
) -> Figure:
    """One panel per target: the across-curve relative error along the curves, the median
    solid and the 95th percentile dashed per curve key varied, beside 4 and 5 figures."""
    targets = list(found.noise)
    fig, axes = plt.subplots(1, len(targets), sharey=True, squeeze=False)
    for ax, target in zip(axes[0], targets, strict=True):
        for (t, key), line in found.profiles.items():
            if t != target:
                continue
            centers = np.array(line.centers)
            ax.plot(
                centers,
                np.maximum(line.median, FLOOR_ON_PLOT),
                color=colors[key],
                label=f"across {labels[key]}",
            )
            ax.plot(centers, np.maximum(line.p95, FLOOR_ON_PLOT), color=colors[key], ls="--")
        for figures in (4, 5):
            ax.axhline(10.0**-figures, color="0.6", ls=":", lw=0.8)
            ax.annotate(
                f"{figures} figures",
                (1.0, 10.0**-figures),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize="x-small",
                color="0.4",
            )
        ax.set_yscale("log")
        ax.set_xlabel(along_label)
        ax.set_title(labels[target])
    axes[0][0].set_ylabel("relative error")
    axes[0][0].legend(fontsize="small")
    fig.tight_layout()
    return fig


# Errors at machine precision are drawn here, so a log axis can show an exact interpolation.
FLOOR_ON_PLOT = 1e-12


def _sci(value: float) -> str:
    """A bound in scientific notation, or a dash when the data left it undefined."""
    return sci_tex(value) if np.isfinite(value) else "--"


def reachable_figures(found: Ceilings, target: str) -> float:
    """The significant figures, -log10 of the relative error, that the worst direction across
    curves allows the target at its 95th percentile."""
    worst = max(s.p95 for (t, _), s in found.across.items() if t == target)
    return float(-np.log10(worst))


def profile(position: np.ndarray, errors: np.ndarray, bins: int) -> Profile:
    """The errors' median and 95th percentile in `bins` equal-count bins of position."""
    order = np.argsort(position)
    chunks = [chunk for chunk in np.array_split(order, bins) if len(chunk)]
    return Profile(
        tuple(float(np.median(position[c])) for c in chunks),
        tuple(float(np.median(errors[c])) for c in chunks),
        tuple(float(np.quantile(errors[c], 0.95)) for c in chunks),
    )


def spread(errors: np.ndarray) -> Spread:
    """The spread of the finite errors."""
    finite = errors[np.isfinite(errors)]
    if not len(finite):
        return Spread(np.nan, np.nan, np.nan, np.nan)
    return Spread(
        float(np.median(finite)),
        float(np.quantile(finite, 0.95)),
        float(finite.max()),
        float(finite.mean()),
    )


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
    return np.concatenate(errors) if errors else np.array([])


def across_curve_errors(
    table: pd.DataFrame,
    fixed: Sequence[str],
    across: str,
    along: str,
    target: str,
    neighbours: int,
) -> Scored:
    """Leave each interior curve out of its group of curves sharing the fixed keys; predict
    its rows from the `neighbours` curves on each side along `across` (linear for one, cubic
    for two), each neighbour interpolated along `along` first."""
    parts: list[Scored] = []
    groups = [table] if not fixed else [group for _, group in table.groupby(list(fixed))]
    for group in groups:
        curves = _curves(group, [across], along)
        for i in range(neighbours, len(curves) - neighbours):
            window = curves[i - neighbours : i + neighbours + 1]
            scored = _held_out(window, neighbours, across, along, target)
            if scored is not None:
                parts.append(scored)
    if not parts:
        empty = np.array([])
        return Scored(empty, empty, empty)
    return Scored(
        np.concatenate([p.errors for p in parts]),
        np.concatenate([p.truth for p in parts]),
        np.concatenate([p.position for p in parts]),
    )


def _held_out(
    window: list[pd.DataFrame], held: int, across: str, along: str, target: str
) -> Scored | None:
    """The scored rows of the curve at index held of the window, predicted from the other
    curves of the window; None when their ranges along it leave too few rows."""
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
    return Scored(np.abs(np.expm1(predicted - np.log(truth))), truth, x)
