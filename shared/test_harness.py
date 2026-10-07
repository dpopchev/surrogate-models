"""Facts about the fair harness, on a tiny synthetic design and toy fitters."""

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_info

from shared.design import Design
from shared.harness import harness, split_valid

GROUPS = np.repeat(np.arange(10), 3)  # ten curves of three rows

# Eight curves of five rows, y = 1 + 0.1 p + x: curves 6 and 7 are test (7 is extrapolation);
# the six training curves fall into two frozen folds.
P = np.repeat(np.arange(8.0), 5)
X_ALONG = np.tile(np.linspace(0.0, 1.0, 5), 8)
TOY = Design(
    X=np.column_stack([X_ALONG, P]),
    y=1.0 + 0.1 * P + X_ALONG,
    groups=P.astype(int),
    test=P >= 6,
    fold=np.where(P >= 6, -1, P.astype(int) % 2),
    ablation=P == 7,
)


def recording(seen: list[np.ndarray]):
    """A fitter that records the X of its validation rows and predicts the mean of y_fit."""

    def fit(x_fit, y_fit, x_valid, y_valid, seed):
        seen.append(x_valid)
        return lambda x: np.full(len(x), y_fit.mean())

    return fit


def test_validation_rows_are_whole_curves() -> None:
    valid = split_valid(GROUPS, fraction=0.2, seed=0)
    assert set(GROUPS[valid]).isdisjoint(GROUPS[~valid])


def test_every_fitter_gets_the_same_validation_rows() -> None:
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    harness(TOY, "mass", recording(first), seeds=[0], valid_fraction=0.34)
    harness(TOY, "mass", recording(second), seeds=[0], valid_fraction=0.34)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def scaled(factor: float):
    """A fitter that predicts the true y (1 + 0.1 p + x) times factor."""

    def fit(x_fit, y_fit, x_valid, y_valid, seed):
        return lambda x: factor * (1.0 + 0.1 * x[:, 1] + x[:, 0])

    return fit


def test_a_fitter_never_sees_a_test_curve() -> None:
    fitted: list[np.ndarray] = []

    def fit(x_fit, y_fit, x_valid, y_valid, seed):
        fitted.append(np.concatenate([x_fit[:, 1], x_valid[:, 1]]))
        return lambda x: np.zeros(len(x))

    harness(TOY, "mass", fit, seeds=[0], valid_fraction=0.34)
    assert max(float(p.max()) for p in fitted) < 6.0


def test_one_run_per_seed() -> None:
    assert len(harness(TOY, "mass", scaled(1.0), seeds=[0, 1, 2], valid_fraction=0.34)) == 3


def test_each_frozen_fold_is_scored() -> None:
    (run,) = harness(TOY, "mass", scaled(1.0), seeds=[0], valid_fraction=0.34)
    assert len(run.folds) == 2


def test_the_extrapolation_zone_holds_the_ablation_test_curve() -> None:
    (run,) = harness(TOY, "mass", scaled(1.1), seeds=[0], valid_fraction=0.34)
    assert run.test.zones["extrapolation"].median == pytest.approx(0.1)


def test_the_test_zone_holds_every_test_curve() -> None:
    (run,) = harness(TOY, "mass", scaled(1.1), seeds=[0], valid_fraction=0.34)
    assert run.test.zones["test"].mean == pytest.approx(0.1)


def test_a_run_keeps_its_predictions_on_the_test_rows() -> None:
    (run,) = harness(TOY, "mass", scaled(1.0), seeds=[0], valid_fraction=0.34)
    assert run.predictions.tolist() == pytest.approx(TOY.y[TOY.test].tolist())


def test_the_fit_is_timed_with_the_injected_clock() -> None:
    readings = iter([10.0, 12.5, 12.6, 13.0])
    (run,) = harness(
        TOY, "mass", scaled(1.0), seeds=[0], valid_fraction=0.34, clock=lambda: next(readings)
    )
    assert run.timing.fit == pytest.approx(2.5)


def test_every_library_runs_on_one_thread_during_a_fit() -> None:
    threads: list[int] = []

    def fit(x_fit, y_fit, x_valid, y_valid, seed):
        threads.append(torch.get_num_threads())
        threads.extend(pool["num_threads"] for pool in threadpool_info())
        return lambda x: np.zeros(len(x))

    harness(TOY, "mass", fit, seeds=[0], valid_fraction=0.34)
    assert set(threads) == {1}
