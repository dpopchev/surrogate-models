"""The equal-budget search over a family's hyperparameters (W-068).

Every candidate gets the same number of Optuna trials. A trial is scored on the design's frozen
folds alone -- each fold held out in turn, the fit on the other training curves -- as the mean
p95 significant figures of its folds; the test curves are never fitted nor predicted, so the
search cannot tune on them. A space is data: Optuna distributions by parameter name; a family
is a callable that turns one trial's parameters into a harness fitter.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, assert_never, get_args

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState

from paperlib.candidates import Unit
from paperlib.ceilings import spread
from paperlib.design import Design, TargetForm
from paperlib.harness import Fitter, fit_on, one_thread
from paperlib.runs import Setting
from paperlib.scorecard import relative_errors, significant_figures
from paperlib.surrogate import Activation, Loss

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

Space = Mapping[str, BaseDistribution]
# trial_fitter(params) -> the harness fitter of one trial's parameters
TrialFitter = Callable[[dict[str, Any]], Fitter]


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
# A black-hole curve holds about 5700 rows against a neutron-star curve's 6 at the fewest, and
# its best knots sat at the top of KNOTS (W-068), so its range runs wider.
BLACK_HOLE_KNOTS = IntDistribution(6, 384, log=True)


@dataclass(frozen=True)
class Searched:
    """A finished search: the best trial's parameters, its mean fold figures, the number of
    trials run and how many of them failed (a raising fit, or a nan score)."""

    params: Mapping[str, Setting]
    figures: float
    trials: int
    failed: int
    cut: bool = False


@dataclass(frozen=True)
class NoCompletedTrial:
    """A search none of whose trials completed: the number run and how many failed (the rest
    were pruned or left unfinished)."""

    trials: int
    failed: int


SearchResult = Searched | NoCompletedTrial


# --- pure functions ---------------------------------------------------------------------------


def space_of(family: str, unit: Unit, dataset: str) -> Space:
    """The space a candidate -- a family in one unit of prediction -- is searched over on one
    dataset: a curve-wise one also searches its spline's knots, over the wider black-hole range
    on black_holes."""
    match unit:
        case "pointwise":
            return SPACES[family]
        case "curve-wise":
            knots = BLACK_HOLE_KNOTS if dataset == "black_holes" else KNOTS
            return {**SPACES[family], "knots": knots}
        case _:
            assert_never(unit)


def search(
    family: TrialFitter,
    space: Space,
    budget: int,
    design: Design,
    target: TargetForm,
    seed: int,
    valid_fraction: float,
    study: optuna.Study | None = None,
    seconds: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SearchResult:
    """Run `budget` trials of the family over its space on the design's frozen folds, in the
    given study (a journal one, for the dashboard) or an in-memory one sampled from the seed.
    The ask / tell loop is the trial's boundary: a fit that raises fails only its trial (told
    FAIL, logged), as a nan score does; the search runs its whole budget. NoCompletedTrial when
    the study holds no completed trial. With `seconds`, the loop ends before asking a trial
    once the clock has passed it since the start (the first trial always runs): the trials
    reported are those run, and a Searched says it was cut."""
    if study is None:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
    start = clock()
    numbers: list[int] = []
    cut = False
    for index in range(budget):
        if seconds is not None and index > 0 and clock() - start > seconds:
            cut = True
            break
        trial = study.ask(dict(space))
        numbers.append(trial.number)
        try:
            fitter = family(trial.params)
            study.tell(trial, fold_figures(design, target, fitter, seed, valid_fraction))
        except Exception as error:
            study.tell(trial, state=TrialState.FAIL)
            logger.warning("trial %d failed: %s", trial.number, error)
    states = {trial.number: trial.state for trial in study.get_trials(deepcopy=False)}
    failed = sum(states[number] == TrialState.FAIL for number in numbers)
    if TrialState.COMPLETE not in states.values():
        return NoCompletedTrial(trials=len(numbers), failed=failed)
    return Searched(
        params=study.best_params,
        figures=study.best_value,
        trials=len(numbers),
        failed=failed,
        cut=cut,
    )


def evaluate(
    family: TrialFitter,
    points: Sequence[dict[str, Any]],
    design: Design,
    target: TargetForm,
    seeds: Sequence[int],
    valid_fraction: float,
) -> SearchResult:
    """Score a fixed list of points in place of a search: each point's figures are the mean of
    its fold figures over the seeds. A point whose fit raises fails alone (logged); the best
    point is a Searched with the points run as its trials, NoCompletedTrial when all failed."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for index, point in enumerate(points):
        try:
            fitter = family(point)
            figures = [fold_figures(design, target, fitter, s, valid_fraction) for s in seeds]
        except Exception as error:
            logger.warning("point %d failed: %s", index, error)
            continue
        scored.append((float(np.mean(figures)), point))
    failed = len(points) - len(scored)
    if not scored:
        return NoCompletedTrial(trials=len(points), failed=failed)
    best_figures, best_point = max(scored, key=lambda pair: pair[0])
    return Searched(params=best_point, figures=best_figures, trials=len(points), failed=failed)


def fold_figures(
    design: Design, target: TargetForm, fitter: Fitter, seed: int, valid_fraction: float
) -> float:
    """The mean p95 significant figures of the fitter over the design's frozen folds: each
    fold predicted by a fit on the other training curves, less their validation curves. Every
    numerical library runs on one thread, as in the harness (W-037: one thread is fastest)."""
    figures = []
    with one_thread():
        for fold in sorted(set(design.fold[design.fold >= 0].tolist())):
            held = design.fold == fold
            predict = fit_on(design, ~design.test & ~held, fitter, seed, valid_fraction)
            errors = relative_errors(design.y[held], predict(design.X[held]), target)
            figures.append(significant_figures(spread(errors).p95))
    return float(np.mean(figures))
