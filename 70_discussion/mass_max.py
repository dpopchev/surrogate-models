"""The maximum neutron-star mass the surrogate predicts for each held-out curve (W-156).

The stable edge in rho_c of a held-out (beta, lambda) is interpolated from the training curves
alone (paperlib/domain.py); the mass surrogate is evaluated there and set against the last row of
the held-out curve, the maximum mass of the data. Keys outside the training keys are skipped and
counted. Nothing is fitted.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from paperlib.design import NEUTRON_STARS
from paperlib.domain import make_neutron_star_domain
from paperlib.harness import Predictor

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class CurveMassMax:
    """One held-out curve: its keys, the predicted and the data maximum mass (M_sun) and the
    relative error of the prediction."""

    beta: float
    lam: float
    predicted: float
    data: float
    error: float


@dataclass(frozen=True)
class MassMaxes:
    """The held-out curves with a prediction, and how many were skipped for lying outside the
    training keys."""

    curves: tuple[CurveMassMax, ...]
    skipped: int


# --- pure functions ---------------------------------------------------------------------------


def predicted_mass_max(table: pd.DataFrame, split: pd.DataFrame, predict: Predictor) -> MassMaxes:
    """The maximum mass of each held-out curve, read from the surrogate at the interpolated edge.

    `table` is the prepared neutron-star table (beta, lambda, rho_c, M; each curve cut at its
    maximum mass), `split` the frozen split file; `predict` takes the design inputs in the order
    of paperlib.design.NEUTRON_STARS (log10 rho_c, beta, lambda).
    """
    space = NEUTRON_STARS.space
    curve = list(space.curve)
    own = split.loc[split["dataset"] == NEUTRON_STARS.dataset, [*curve, "label"]]
    labelled = table.merge(own, on=curve, how="left", validate="many_to_one")
    training = labelled[labelled["label"].notna() & (labelled["label"] != "test")]
    domain = make_neutron_star_domain(training)
    edge_of = LinearNDInterpolator(np.array(domain.keys), np.array(domain.rho_max))
    found: list[CurveMassMax] = []
    skipped = 0
    for (beta, lam), rows in labelled[labelled["label"] == "test"].groupby(curve, sort=True):
        edge = float(edge_of(float(beta), float(lam)))  # type: ignore[arg-type]
        if np.isnan(edge):
            skipped += 1
            continue
        x = np.array([[np.log10(edge), beta, lam]], dtype=np.float32)
        predicted = float(predict(x)[0])
        data = float(rows["M"].iloc[-1])
        found.append(
            CurveMassMax(float(beta), float(lam), predicted, data, abs(predicted - data) / data)  # type: ignore[arg-type]
        )
    return MassMaxes(curves=tuple(found), skipped=skipped)
