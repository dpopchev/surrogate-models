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
from scipy.interpolate import CubicSpline
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from shared.ceilings import Spread, spread
from shared.design import Design, TargetForm
from shared.scorecard import Scorecard, relative_errors, ripple, scorecard

# --- vocabulary and types ---------------------------------------------------------------------

Predictor = Callable[[np.ndarray], np.ndarray]
# fitter(X_fit, y_fit, X_valid, y_valid, seed) -> predictor
Fitter = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], Predictor]

# Rows an MCMC step asks the surrogate for at once (an emcee ensemble of walkers).
BATCH_ROWS = 64

# Points of the along-curve grid the test curves are aligned on for the ripple (T-170).
RIPPLE_GRID = 32


@dataclass(frozen=True)
class Timing:
    """Wall-clock seconds of the final fit, of a one-row and of a batch prediction."""

    fit: float
    predict_one: float
    predict_batch: float


@dataclass(frozen=True)
class Run:
    """One seed's result: the scorecard on the test curves, the error spread on each frozen
    fold, the timing of the final fit, its predictions on the test rows, and the ripple of
    its residual on the test curves across each curve parameter (T-170)."""

    seed: int
    test: Scorecard
    folds: tuple[Spread, ...]
    timing: Timing
    predictions: np.ndarray
    ripple: tuple[float, ...] = ()


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
    target: TargetForm,
    fitter: Fitter,
    seeds: Sequence[int],
    valid_fraction: float,
    charge: np.ndarray | None = None,
    clock: Callable[[], float] | None = None,
    record: Callable[[Run], None] = lambda run: None,
) -> tuple[Run, ...]:
    """Score a fitter on the design for each seed; charge is the true D of the design's rows,
    for the per-decade scorecard of the charge target; record is handed each run as it ends
    (the run ledger, W-077)."""
    tick = clock or time.perf_counter
    runs = []
    with threadpool_limits(limits=1):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for seed in seeds:
                run = _run(design, target, fitter, seed, valid_fraction, charge, tick)
                record(run)
                runs.append(run)
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
    target: TargetForm,
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
    predictions = predict(x_test)
    errors = relative_errors(design.y[design.test], predictions, target)
    zones = {
        "test": np.ones(int(design.test.sum()), dtype=bool),
        "interior": ~design.ablation[design.test],
        "extrapolation": design.ablation[design.test],
    }
    found = scorecard(errors, zones, None if charge is None else charge[design.test])
    timing = Timing(fitted - started, one - fitted, batch - one)
    found_ripple = ripples(x_test, design.y[design.test], predict)
    return Run(seed, found, tuple(folds), timing, predictions, found_ripple)


def ripples(x: np.ndarray, y: np.ndarray, predict: Predictor) -> tuple[float, ...]:
    """Per curve parameter (the columns of x after the along-curve input in column 0), the
    ripple of the predictor's residual across that parameter on a grid shared by the curves
    differing only in it."""
    return tuple(_ripple_across(x, y, predict, column) for column in range(1, x.shape[1]))


def _ripple_across(x: np.ndarray, y: np.ndarray, predict: Predictor, column: int) -> float:
    """The ripple across one curve parameter: the curves sharing every other parameter are
    sampled on one grid over the overlap of their along-curve inputs, the truth by a cubic
    spline of each curve, and the residual's second differences taken across the parameter at
    each grid point; nan when no three such curves overlap."""
    others = np.delete(x[:, 1:], column - 1, axis=1)
    _, family = np.unique(others, axis=0, return_inverse=True)
    residuals, across, fixed = [], [], []
    for group in np.unique(family):
        rows = family == group
        values = np.unique(x[rows, column])
        curves = [rows & (x[:, column] == value) for value in values]
        low = max(float(x[curve, 0].min()) for curve in curves)
        high = min(float(x[curve, 0].max()) for curve in curves)
        if len(curves) < 3 or low >= high:
            continue
        grid = np.linspace(low, high, RIPPLE_GRID)
        for value, curve in zip(values, curves, strict=True):
            order = np.argsort(x[curve, 0])
            truth = CubicSpline(x[curve, 0][order], y[curve][order])(grid)
            on_grid = np.repeat(x[curve][:1], RIPPLE_GRID, axis=0)
            on_grid[:, 0] = grid
            residuals.append(predict(on_grid) - truth)
            across.append(np.full(RIPPLE_GRID, value))
            fixed.append(group * RIPPLE_GRID + np.arange(RIPPLE_GRID))
    if not residuals:
        return float("nan")
    return ripple(np.concatenate(residuals), np.concatenate(across), np.concatenate(fixed))
