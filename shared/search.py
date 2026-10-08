"""The equal-budget search over a family's hyperparameters (W-068).

Every candidate gets the same number of Optuna trials. A trial is scored on the design's frozen
folds alone -- each fold held out in turn, the fit on the other training curves -- as the mean
p95 significant figures of its folds; the test curves are never fitted nor predicted, so the
search cannot tune on them. A space is data: Optuna distributions by parameter name; a family
is a callable that turns one trial's parameters into a harness fitter.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, assert_never, get_args

import numpy as np
import optuna
import torch
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from threadpoolctl import threadpool_limits

from shared.ceilings import spread
from shared.design import Design, TargetForm
from shared.harness import Fitter, split_valid
from shared.scorecard import relative_errors, significant_figures
from shared.surrogate import Activation, Loss

# --- vocabulary and types ---------------------------------------------------------------------

Space = Mapping[str, BaseDistribution]
# The unit of prediction: one row, or a whole curve along its normalized coordinate.
Unit = Literal["pointwise", "curve-wise"]
# family(params) -> the harness fitter of one trial's parameters
Family = Callable[[dict[str, Any]], Fitter]


_NETWORK: Space = {
    "width": IntDistribution(32, 512, log=True),
    "depth": IntDistribution(2, 8),
    "activation": CategoricalDistribution(get_args(Activation)),
    "loss": CategoricalDistribution(get_args(Loss)),
    "lr": FloatDistribution(1e-4, 1e-2, log=True),
}

# Each family's own hyperparameters, centred on the screen's settings (k 8, 100 neighbours,
# the paper.toml network); the ranges are a first guess, widened when a best trial sits on an
# edge.
SPACES: Mapping[str, Space] = {
    "k-NN": {
        "k": IntDistribution(2, 64, log=True),
        "weights": CategoricalDistribution(("uniform", "distance")),
    },
    "local RBF": {"neighbours": IntDistribution(50, 400, log=True)},
    "GPR": {"kernel": CategoricalDistribution(("rbf", "matern52", "rational_quadratic"))},
    "XGBoost": {
        "max_depth": IntDistribution(3, 12),
        "n_estimators": IntDistribution(100, 2000, log=True),
        "learning_rate": FloatDistribution(0.01, 0.3, log=True),
    },
    "MLP": _NETWORK,
    "ResNet": _NETWORK,
}

# The curve-wise unit's cubic spline knots along s, ends included (the screen ran 8).
KNOTS = IntDistribution(6, 24)


@dataclass(frozen=True)
class Searched:
    """A finished search: the best trial's parameters, its mean fold figures, and the number
    of trials run."""

    params: dict[str, Any]
    figures: float
    trials: int


# --- pure functions ---------------------------------------------------------------------------


def space_of(family: str, unit: Unit) -> Space:
    """The space a candidate -- a family in one unit of prediction -- is searched over: a
    curve-wise one also searches its spline's knots."""
    match unit:
        case "pointwise":
            return SPACES[family]
        case "curve-wise":
            return {**SPACES[family], "knots": KNOTS}
        case _:
            assert_never(unit)


def search(
    family: Family,
    space: Space,
    budget: int,
    design: Design,
    target: TargetForm,
    seed: int,
    valid_fraction: float,
    study: optuna.Study | None = None,
) -> Searched:
    """Run `budget` trials of the family over its space on the design's frozen folds, in the
    given study (a journal one, for the dashboard) or an in-memory one sampled from the seed."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = study or optuna.create_study(direction="maximize", sampler=sampler)
    for _ in range(budget):
        trial = study.ask(dict(space))
        fitter = family(trial.params)
        study.tell(trial, fold_figures(design, target, fitter, seed, valid_fraction))
    return Searched(params=study.best_params, figures=study.best_value, trials=budget)


def fold_figures(
    design: Design, target: TargetForm, fitter: Fitter, seed: int, valid_fraction: float
) -> float:
    """The mean p95 significant figures of the fitter over the design's frozen folds: each
    fold predicted by a fit on the other training curves, less their validation curves. Every
    numerical library runs on one thread, as in the harness (W-037: one thread is fastest)."""
    figures = []
    with threadpool_limits(limits=1):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for fold in sorted(set(design.fold[design.fold >= 0].tolist())):
                held = design.fold == fold
                rows = np.flatnonzero(~design.test & ~held)
                valid = split_valid(design.groups[rows], valid_fraction, seed)
                fit_rows, valid_rows = rows[~valid], rows[valid]
                x_valid, y_valid = design.X[valid_rows], design.y[valid_rows]
                predict = fitter(design.X[fit_rows], design.y[fit_rows], x_valid, y_valid, seed)
                errors = relative_errors(design.y[held], predict(design.X[held]), target)
                figures.append(significant_figures(spread(errors).p95))
        finally:
            torch.set_num_threads(threads)
    return float(np.mean(figures))
