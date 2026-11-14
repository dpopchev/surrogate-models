"""The zones of held-out rows -- interior, edge, physical edge -- and the figures each keeps.

Pure functions on numpy arrays; nothing is fitted.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from paperlib.ceilings import spread
from paperlib.scorecard import significant_figures

Side = Literal["start", "end"]


@dataclass(frozen=True)
class ZoneFigures:
    """The significant figures kept at the 95th percentile and at the worst row of a zone."""

    p95: float
    worst: float


def curve_position(groups: np.ndarray, along: np.ndarray) -> np.ndarray:
    """Per row its rank along its own curve over the curve's row count less one; 0 for a
    one-row curve."""
    position = np.zeros(len(groups))
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if len(rows) > 1:
            ranks = np.argsort(np.argsort(along[rows], kind="stable"), kind="stable")
            position[rows] = ranks / (len(rows) - 1)
    return position


def physical_edge(position: np.ndarray, side: Side, fraction: float) -> np.ndarray:
    """The rows within `fraction` of the start (the black-hole existence edge) or of the end
    (the neutron-star stable edge) of their curve."""
    if side == "start":
        return position <= fraction
    return position >= 1.0 - fraction


def zone_masks(
    ablation: np.ndarray, position: np.ndarray, side: Side, fraction: float
) -> dict[str, np.ndarray]:
    """The interior (ablation False), edge (ablation True) and physical-edge row masks."""
    return {
        "interior": ~ablation,
        "edge": ablation.copy(),
        "physical": physical_edge(position, side, fraction),
    }


def zone_figures(
    errors: np.ndarray, masks: Mapping[str, np.ndarray]
) -> dict[str, ZoneFigures | None]:
    """Per zone the figures of its p95 and of its worst row; None for an empty zone."""
    figures: dict[str, ZoneFigures | None] = {}
    for name, mask in masks.items():
        if not mask.any():
            figures[name] = None
            continue
        found = spread(errors[mask])
        figures[name] = ZoneFigures(significant_figures(found.p95), significant_figures(found.max))
    return figures
