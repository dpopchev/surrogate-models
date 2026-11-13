"""The physical domain of each map: where stable solutions exist (W-153).

A black hole exists only for r_h at or above r_h,min(beta), the edge of each sampled beta curve,
and up to the end of the curve. A neutron-star curve of fixed (beta, lambda) is stable from its
smallest rho_c up to its last row, where the prepared table is cut at the maximum mass. Between
sampled curves an edge is interpolated linearly in the curve keys; outside the sampled keys
nothing is interpolated. A query outside is refused with the reason and the nearest valid input.
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


@dataclass(frozen=True)
class NeutronStarDomain:
    """Per sampled (beta, lambda) the smallest rho_c and the rho_c of the last row of its curve."""

    keys: tuple[tuple[float, float], ...]
    rho_min: tuple[float, ...]
    rho_max: tuple[float, ...]


Domain = BlackHoleDomain | NeutronStarDomain


@dataclass(frozen=True)
class Inside:
    """The query lies where stable solutions exist."""


@dataclass(frozen=True)
class Outside:
    """The query lies beyond the edge named by the reason; nearest is the closest valid input."""

    reason: Reason
    nearest: Mapping[str, float]


Verdict = Inside | Outside


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


def _check_black_hole(domain: BlackHoleDomain, x: Mapping[str, float]) -> Verdict:
    beta, r_h = x["beta"], x["r_h"]
    beta_in = float(np.clip(beta, domain.betas[0], domain.betas[-1]))
    low = float(np.interp(beta_in, domain.betas, domain.r_min))
    high = float(np.interp(beta_in, domain.betas, domain.r_max))
    nearest = {"beta": beta_in, "r_h": float(np.clip(r_h, low, high))}
    if beta_in != beta:
        return Outside("coupling range", nearest)
    if r_h < low - TOLERANCE:
        return Outside("existence edge", nearest)
    if r_h > high + TOLERANCE:
        return Outside("sampled range", nearest)
    return Inside()


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
        in_box = _in_box(keys, beta, lam)
        return Outside("sampled range" if in_box else "coupling range", nearest)
    if rho < low - TOLERANCE:
        return Outside("sampled range", nearest)
    if rho > high + TOLERANCE:
        return Outside("stable branch", nearest)
    return Inside()


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
