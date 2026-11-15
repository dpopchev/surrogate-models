"""The physical domain of each map: where stable solutions exist (W-153).

A black hole exists only for r_h at or above r_h,min(beta), the edge of each sampled beta curve,
and up to the end of the curve. A neutron-star curve of fixed (beta, lambda) is stable from its
smallest rho_c up to its last row, where the prepared table is cut at the maximum mass. Between
sampled curves an edge is interpolated linearly in the curve keys; outside the sampled keys
nothing is interpolated. A query outside is refused with the reason and the nearest valid input.
A domain with a margin (W-173), counted in steps of the sampled coupling grid, answers a physically
valid query beyond the sampled couplings by at most the margin with a warning; there the physical
edge is the linear continuation of the edge of the two outermost sampled curves.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

Reason = Literal["coupling range", "existence edge", "stable branch", "sampled range"]

# A tolerance so a row on an interpolated edge is not refused by rounding.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class BlackHoleDomain:
    """Per sampled beta (ascending) the smallest and the largest r_h of its curve."""

    betas: tuple[float, ...]
    r_min: tuple[float, ...]
    r_max: tuple[float, ...]
    margin: float = 0.0


@dataclass(frozen=True)
class NeutronStarDomain:
    """Per sampled (beta, lambda) the smallest rho_c and the rho_c of the last row of its curve."""

    keys: tuple[tuple[float, float], ...]
    rho_min: tuple[float, ...]
    rho_max: tuple[float, ...]
    margin: float = 0.0


Domain = BlackHoleDomain | NeutronStarDomain


@dataclass(frozen=True)
class Inside:
    """The query lies where stable solutions exist."""


@dataclass(frozen=True)
class Outside:
    """The query lies beyond the edge named by the reason; nearest is the closest valid input."""

    reason: Reason
    nearest: Mapping[str, float]


@dataclass(frozen=True)
class Warned:
    """The query lies beyond the sampled couplings, inside the margin: steps of the coupling grid
    beyond them, nearest the closest sampled input."""

    reason: Literal["coupling range"]
    steps: float
    nearest: Mapping[str, float]


Verdict = Inside | Outside | Warned


# --- smart constructors -----------------------------------------------------------------------


def _require(table: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError naming the missing columns, or the empty table."""
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"the prepared table lacks the columns {missing}")
    if table.empty:
        raise ValueError("the prepared table has no rows")


def make_black_hole_domain(table: pd.DataFrame) -> BlackHoleDomain:
    """Read the domain from the columns beta and r_h of a prepared table."""
    _require(table, ["beta", "r_h"])
    edges = table.groupby("beta")["r_h"].agg(["min", "max"]).sort_index()
    return BlackHoleDomain(
        betas=tuple(float(b) for b in edges.index),
        r_min=tuple(float(v) for v in edges["min"]),
        r_max=tuple(float(v) for v in edges["max"]),
    )


def make_neutron_star_domain(table: pd.DataFrame) -> NeutronStarDomain:
    """Read the domain from the columns beta, lambda and rho_c; the last row of each curve is
    its stable edge, since the prepared table is cut at the maximum mass."""
    _require(table, ["beta", "lambda", "rho_c"])
    keys: list[tuple[float, float]] = []
    low: list[float] = []
    high: list[float] = []
    for (beta, lam), curve in table.groupby(["beta", "lambda"], sort=True):
        keys.append((float(beta), float(lam)))  # type: ignore[arg-type]
        low.append(float(curve["rho_c"].min()))
        high.append(float(curve["rho_c"].iloc[-1]))
    return NeutronStarDomain(keys=tuple(keys), rho_min=tuple(low), rho_max=tuple(high))


# --- verdicts ---------------------------------------------------------------------------------


def check(domain: Domain, x: Mapping[str, float]) -> Verdict:
    """The verdict on one input row: its couplings and its curve coordinate."""
    match domain:
        case BlackHoleDomain():
            return _check_black_hole(domain, x)
        case NeutronStarDomain():
            return _check_neutron_star(domain, x)
        case _:
            assert_never(domain)


def _spacing(values: np.ndarray) -> float:
    """The median spacing of the sampled values of one coupling; 0 when fewer than two."""
    distinct = np.unique(values)
    return float(np.median(np.diff(distinct))) if len(distinct) > 1 else 0.0


def _steps(excess: float, spacing: float) -> float:
    """How many grid steps an excess is; infinite where the grid has no step."""
    if excess == 0.0:
        return 0.0
    return excess / spacing if spacing > 0.0 else float("inf")


def _check_black_hole(domain: BlackHoleDomain, x: Mapping[str, float]) -> Verdict:
    beta, r_h = x["beta"], x["r_h"]
    beta_in = float(np.clip(beta, domain.betas[0], domain.betas[-1]))
    low = float(np.interp(beta_in, domain.betas, domain.r_min))
    high = float(np.interp(beta_in, domain.betas, domain.r_max))
    nearest = {"beta": beta_in, "r_h": float(np.clip(r_h, low, high))}
    steps = _steps(abs(beta - beta_in), _spacing(np.array(domain.betas)))
    if beta_in != beta:
        if steps > domain.margin + TOLERANCE:
            return Outside("coupling range", nearest)
        low += _edge_slope(domain, beta > beta_in) * (beta - beta_in)
    if r_h < low - TOLERANCE:
        return Outside("existence edge", nearest)
    if r_h > high + TOLERANCE:
        return Outside("sampled range", nearest)
    return Warned("coupling range", steps, nearest) if beta_in != beta else Inside()


def _edge_slope(domain: BlackHoleDomain, upper: bool) -> float:
    """The slope of the existence edge between the two outermost curves on one side."""
    if len(domain.betas) < 2:
        return 0.0
    i, j = (-2, -1) if upper else (0, 1)
    return (domain.r_min[j] - domain.r_min[i]) / (domain.betas[j] - domain.betas[i])


def _check_neutron_star(domain: NeutronStarDomain, x: Mapping[str, float]) -> Verdict:
    beta, lam, rho = x["beta"], x["lambda"], x["rho_c"]
    keys = np.array(domain.keys)
    beta_in = float(np.clip(beta, keys[:, 0].min(), keys[:, 0].max()))
    lam_in = float(np.clip(lam, keys[:, 1].min(), keys[:, 1].max()))
    low, high = stable_edge(domain, beta_in, lam_in)
    if np.isnan(low):
        # Inside the bounding box of the keys but off their hull: snap to the nearest curve.
        i = int(np.argmin(np.hypot(keys[:, 0] - beta_in, keys[:, 1] - lam_in)))
        beta_in, lam_in = float(keys[i, 0]), float(keys[i, 1])
        low, high = domain.rho_min[i], domain.rho_max[i]
    nearest = {"beta": beta_in, "lambda": lam_in, "rho_c": float(np.clip(rho, low, high))}
    if (beta_in, lam_in) != (beta, lam):
        if _in_box(keys, beta, lam):
            return Outside("sampled range", nearest)
        return _extrapolated_neutron_star(domain, keys, (beta, lam), rho, nearest)
    if rho < low - TOLERANCE:
        return Outside("sampled range", nearest)
    if rho > high + TOLERANCE:
        return Outside("stable branch", nearest)
    return Inside()


def _extrapolated_neutron_star(
    domain: NeutronStarDomain,
    keys: np.ndarray,
    couplings: tuple[float, float],
    rho: float,
    nearest: dict[str, float],
) -> Verdict:
    """The verdict beyond the box of the sampled keys: the stable edge continues as the plane
    through the sampled edges, the margin counts the steps of the coarser coupling."""
    excess = np.array([couplings[0] - nearest["beta"], couplings[1] - nearest["lambda"]])
    steps = max(
        _steps(abs(excess[0]), _spacing(keys[:, 0])), _steps(abs(excess[1]), _spacing(keys[:, 1]))
    )
    if steps > domain.margin + TOLERANCE:
        return Outside("coupling range", nearest)
    here = stable_edge(domain, nearest["beta"], nearest["lambda"])
    low = here[0] + float(_plane_slope(keys, np.array(domain.rho_min)) @ excess)
    high = here[1] + float(_plane_slope(keys, np.array(domain.rho_max)) @ excess)
    if rho < low - TOLERANCE:
        return Outside("sampled range", nearest)
    if rho > high + TOLERANCE:
        return Outside("stable branch", nearest)
    return Warned("coupling range", steps, nearest)


def _plane_slope(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The gradient of the least-squares plane of an edge over the sampled couplings."""
    design = np.column_stack([keys, np.ones(len(keys))])
    return np.asarray(np.linalg.lstsq(design, values, rcond=None)[0][:2])


def _in_box(keys: np.ndarray, beta: float, lam: float) -> bool:
    """Whether the couplings lie inside the bounding box of the sampled keys."""
    return bool(
        keys[:, 0].min() <= beta <= keys[:, 0].max() and keys[:, 1].min() <= lam <= keys[:, 1].max()
    )


def stable_edge(domain: NeutronStarDomain, beta: float, lam: float) -> tuple[float, float]:
    """The interpolated smallest and last rho_c at one coupling pair; nan off the hull."""
    if len(domain.keys) == 1:
        same = (beta, lam) == domain.keys[0]
        return (domain.rho_min[0], domain.rho_max[0]) if same else (np.nan, np.nan)
    keys = np.array(domain.keys)
    low = LinearNDInterpolator(keys, np.array(domain.rho_min))(beta, lam)
    high = LinearNDInterpolator(keys, np.array(domain.rho_max))(beta, lam)
    return float(low), float(high)
