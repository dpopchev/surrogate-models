"""The fair harness every candidate surrogate is scored through (W-065).

A candidate is a fitter: given the rows to fit and the validation rows, and a seed, it returns a
predictor. The harness owns everything else, so two scorecards differ only in the candidate:
the frozen folds and test curves of the design, one split of each training set into fit and
validation curves (the same for every candidate; validation rows are never fit rows), one
thread for every numerical library, the seeds, and the timing of the fit and of a one-row and
an emcee-sized batch prediction.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from shared.ceilings import Spread, spread
from shared.design import Design, Target
from shared.scorecard import Scorecard, relative_errors, scorecard

# --- vocabulary and types ---------------------------------------------------------------------

Predictor = Callable[[np.ndarray], np.ndarray]
# fitter(X_fit, y_fit, X_valid, y_valid, seed) -> predictor
Fitter = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], Predictor]

# Rows an MCMC step asks the surrogate for at once (an emcee ensemble of walkers).
BATCH_ROWS = 64


@dataclass(frozen=True)
class Timing:
    """Wall-clock seconds of the final fit, of a one-row and of a batch prediction."""

    fit: float
    predict_one: float
    predict_batch: float


@dataclass(frozen=True)
class Run:
    """One seed's result: the scorecard on the test curves, the error spread on each frozen
    fold, and the timing of the final fit."""

    seed: int
    test: Scorecard
    folds: tuple[Spread, ...]
    timing: Timing


# --- pure functions ---------------------------------------------------------------------------


def split_valid(groups: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A mask of the rows whose curves form the validation share of these rows' curves."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    _, valid_rows = next(splitter.split(groups, groups=groups))
    valid = np.zeros(len(groups), dtype=bool)
    valid[valid_rows] = True
    return valid


def harness(
    design: Design,
    target: Target,
    fitter: Fitter,
    seeds: Sequence[int],
    valid_fraction: float,
    charge: np.ndarray | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[Run, ...]:
    """Score a fitter on the design for each seed; charge is the true D of the design's rows,
    for the per-decade scorecard of the charge target."""
    tick = clock or time.perf_counter
    runs = []
    with threadpool_limits(limits=1):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for seed in seeds:
                runs.append(_run(design, target, fitter, seed, valid_fraction, charge, tick))
        finally:
            torch.set_num_threads(threads)
    return tuple(runs)


def _fit(
    design: Design, train: np.ndarray, fitter: Fitter, seed: int, fraction: float
) -> Predictor:
    """Fit on the training rows less their validation curves, validated on those curves."""
    rows = np.flatnonzero(train)
    valid = split_valid(design.groups[rows], fraction, seed)
    fit_rows, valid_rows = rows[~valid], rows[valid]
    return fitter(
        design.X[fit_rows], design.y[fit_rows], design.X[valid_rows], design.y[valid_rows], seed
    )


def _run(
    design: Design,
    target: Target,
    fitter: Fitter,
    seed: int,
    fraction: float,
    charge: np.ndarray | None,
    tick: Callable[[], float],
) -> Run:
    """One seed: every frozen fold, then the final fit on all non-test curves."""
    folds = []
    for fold in sorted(set(design.fold[design.fold >= 0].tolist())):
        held = design.fold == fold
        predict = _fit(design, ~design.test & ~held, fitter, seed, fraction)
        folds.append(spread(relative_errors(design.y[held], predict(design.X[held]), target)))
    started = tick()
    predict = _fit(design, ~design.test, fitter, seed, fraction)
    fitted = tick()
    x_test = design.X[design.test]
    predict(x_test[:1])
    one = tick()
    predict(np.resize(x_test, (BATCH_ROWS, x_test.shape[1])))
    batch = tick()
    errors = relative_errors(design.y[design.test], predict(x_test), target)
    zones = {
        "interior": ~design.ablation[design.test],
        "extrapolation": design.ablation[design.test],
    }
    found = scorecard(errors, zones, None if charge is None else charge[design.test])
    return Run(seed, found, tuple(folds), Timing(fitted - started, one - fitted, batch - one))
