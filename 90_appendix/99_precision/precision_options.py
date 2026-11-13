"""The precision options of the tuning and precision appendix (W-093): what each one is and
which family takes it.

An option is a stage laid on the tuned fitter of a candidate. The residual stage fits a second
model on the first one's residuals; float64 and L-BFGS are for the networks, none of which is a
saved winner yet, so their stages are written when one is.
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from paperlib.candidates import Candidate, Knobs, tuned_fitter
from paperlib.eda import CurveSpace
from paperlib.families import make_rbf
from paperlib.harness import FitRows, Fitter, Predictor
from paperlib.surrogate import Training, sklearn_fitter

logger = logging.getLogger(__name__)

Option = Literal["tuned", "float64", "lbfgs", "residual"]

NETWORKS = ("MLP", "ResNet")


def residual_fitter(first: Fitter, second: Fitter) -> Fitter:
    """A fitter that fits first, fits second on what first leaves and predicts the sum."""

    def fit(rows: FitRows, seed: int) -> Predictor:
        base = first(rows, seed)
        left = FitRows(
            rows.x_fit,
            rows.y_fit - base(rows.x_fit),
            rows.x_valid,
            rows.y_valid - base(rows.x_valid),
        )
        correction = second(left, seed)
        return lambda x: np.asarray(base(x)) + np.asarray(correction(x))

    return fit


def applicable(candidate: Candidate) -> dict[Option, bool]:
    """Which options the candidate's family takes: tuned and residual always; float64 and
    L-BFGS only where a gradient stage runs, the networks, still unwritten."""
    if candidate.family in NETWORKS:
        logger.warning("[WARN] not implemented yet: float64 and L-BFGS for networks")
    return {"tuned": True, "float64": False, "lbfgs": False, "residual": True}


def option_fitter(
    option: Option,
    candidate: Candidate,
    space: CurveSpace,
    knobs: Knobs,
    training: Training,
    params: Mapping[str, Any],
    residual_neighbours: int,
) -> Fitter:
    """The candidate's fitter under the option; ValueError for an option its family refuses."""
    if not applicable(candidate)[option]:
        raise ValueError(f"option {option} does not apply to {candidate.family}")
    tuned = tuned_fitter(candidate, space, knobs, training, params)
    if option == "tuned":
        return tuned
    return residual_fitter(tuned, sklearn_fitter(lambda: make_rbf(residual_neighbours)))
