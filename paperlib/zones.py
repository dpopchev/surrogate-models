"""The zones of held-out rows -- interior, edge, physical edge -- and the figures each keeps.

Pure functions on numpy arrays; only edge_held_out fits, through the fitter it is given.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from paperlib.ceilings import spread
from paperlib.design import Design, TargetForm
from paperlib.harness import Fitter, fit_on
from paperlib.scorecard import relative_errors, significant_figures

Side = Literal["start", "end"]


@dataclass(frozen=True)
class ZoneFigures:
    """The significant figures kept at the 95th percentile and at the worst row of a zone."""

    p95: float
    worst: float


@dataclass(frozen=True)
class EdgeHeldOut:
    """The figures of the edge and interior zones of a fit made without the edge, and the
    seconds its fit and its prediction took."""

    zones: Mapping[str, ZoneFigures | None]
    fit_seconds: float
    predict_seconds: float


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


def slope_violations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    along: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, tuple[int, int]]:
    """Per zone the pair (violations, steps) over the steps of each curve in the order of `along`.

    A step is two neighbouring rows of one curve, both in the zone; one whose true difference is
    zero is not counted; a violation is a counted step whose predicted difference has another
    sign than the true one, zero included.
    """
    found = {name: (0, 0) for name in masks}
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(along[rows], kind="stable")]
        true_step = np.diff(y_true[rows])
        pred_step = np.diff(y_pred[rows])
        counted = true_step != 0.0
        wrong = counted & (np.sign(pred_step) != np.sign(true_step))
        for name, mask in masks.items():
            both = mask[rows][:-1] & mask[rows][1:]
            violations, steps = found[name]
            found[name] = (
                violations + int((wrong & both).sum()),
                steps + int((counted & both).sum()),
            )
    return found


def margin_steps(
    edge_p95: float, edge_worst: float, interior_p95: float, within: float = 1.0
) -> int:
    """One step of the coupling grid when the edge p95 figures are at least the interior p95
    figures less `within` and the worst edge row keeps positive figures, else zero."""
    return int(edge_p95 >= interior_p95 - within and edge_worst > 0.0)


def edge_held_out(
    design: Design,
    target: TargetForm,
    fitter: Fitter,
    seed: int,
    valid_fraction: float,
    clock: Callable[[], float],
) -> EdgeHeldOut:
    """One fit made without the edge and the test curves, scored on the edge curves outside
    the test set and on the interior test curves."""
    edge = design.ablation & ~design.test
    if not edge.any():
        raise ValueError("no edge curve to hold out")
    started = clock()
    predict = fit_on(design, ~design.ablation & ~design.test, fitter, seed, valid_fraction)
    fitted = clock()
    scored = edge | (~design.ablation & design.test)
    predictions = np.zeros(len(design.y))
    predictions[scored] = predict(design.X[scored])
    predicted = clock()
    errors = np.zeros(len(design.y))
    errors[scored] = relative_errors(design.y[scored], predictions[scored], target)
    masks = {"edge": edge, "interior": ~design.ablation & design.test}
    return EdgeHeldOut(zone_figures(errors, masks), fitted - started, predicted - fitted)
