"""Facts about the precision options of Section 5.3, on tiny synthetic rows."""

import logging

import numpy as np
import pytest
from precision_options import applicable, option_fitter, residual_fitter
from sklearn.neighbors import KNeighborsRegressor

from paperlib.candidates import FAMILIES, Candidate, Knobs
from paperlib.eda import make_curve_space
from paperlib.harness import FitRows, Fitter, Predictor
from paperlib.surrogate import Training, sklearn_fitter

X_FIT = np.array([[0.0], [1.0], [2.0], [3.0]])
Y_FIT = np.array([1.0, 3.0, 5.0, 7.0])
X_VALID = np.array([[4.0], [5.0]])
Y_VALID = np.array([9.0, 11.0])
ROWS = FitRows(X_FIT, Y_FIT, X_VALID, Y_VALID)

KNOBS = Knobs(k=2, neighbours=3, knots=4, per_curve=2)
TRAINING = Training(
    width=4,
    depth=1,
    activation="relu",
    loss="mse",
    lr=1e-3,
    max_epochs=1,
    batch_size=2,
    patience=1,
    valid_fraction=0.5,
    seed=0,
)
SPACE = make_curve_space({"a": "raw", "b": "raw"}, ("a",))


def constant(value: float) -> Fitter:
    """A fake fitter that predicts one constant."""

    def fitter(rows: FitRows, seed: int) -> Predictor:
        return lambda x: np.full(len(x), value)

    return fitter


def recording() -> tuple[list[FitRows], Fitter]:
    """A fake fitter that records the rows it is fitted on and predicts 0.5."""
    seen: list[FitRows] = []

    def fitter(rows: FitRows, seed: int) -> Predictor:
        seen.append(rows)
        return lambda x: np.full(len(x), 0.5)

    return seen, fitter


@pytest.mark.model
def test_the_residual_stage_predicts_the_sum() -> None:
    _, second = recording()
    predictor = residual_fitter(constant(2.0), second)(ROWS, 0)
    assert np.allclose(predictor(X_VALID), [2.5, 2.5])


@pytest.mark.model
def test_the_second_stage_is_fitted_on_y_minus_the_first_prediction() -> None:
    seen, second = recording()
    residual_fitter(constant(2.0), second)(ROWS, 0)
    (rows,) = seen
    assert np.allclose(
        np.concatenate([rows.y_fit, rows.y_valid]), np.concatenate([Y_FIT, Y_VALID]) - 2.0
    )


@pytest.mark.model
def test_the_second_stage_sees_the_same_inputs() -> None:
    seen, second = recording()
    residual_fitter(constant(2.0), second)(ROWS, 0)
    (rows,) = seen
    assert np.array_equal(
        np.concatenate([rows.x_fit, rows.x_valid]), np.concatenate([X_FIT, X_VALID])
    )


@pytest.mark.model
def test_the_inputs_are_left_untouched() -> None:
    _, second = recording()
    y_before = ROWS.y_fit.copy()
    residual_fitter(constant(2.0), second)(ROWS, 0)
    assert np.array_equal(ROWS.y_fit, y_before)


@pytest.mark.model
def test_a_curve_wise_local_rbf_takes_tuned_and_residual_only() -> None:
    assert applicable(Candidate("local RBF", "curve-wise")) == {
        "tuned": True,
        "float64": False,
        "lbfgs": False,
        "residual": True,
    }


@pytest.mark.model
def test_every_family_takes_tuned_and_residual() -> None:
    taken = [
        (table["tuned"], table["residual"])
        for family in FAMILIES
        for table in [applicable(Candidate(family, "pointwise"))]
    ]
    assert taken == [(True, True)] * len(FAMILIES)


@pytest.mark.model
def test_a_network_warns_that_its_stages_are_not_written(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        applicable(Candidate("MLP", "pointwise"))
    assert "[WARN] not implemented yet: float64 and L-BFGS for networks" in caplog.text


@pytest.mark.model
def test_option_fitter_refuses_lbfgs_for_a_local_rbf() -> None:
    with pytest.raises(ValueError, match=r"lbfgs.*local RBF"):
        option_fitter("lbfgs", Candidate("local RBF", "curve-wise"), SPACE, KNOBS, TRAINING, {}, 3)


@pytest.mark.model
def test_the_tuned_option_predicts_a_row_per_input() -> None:
    fitter = option_fitter("tuned", Candidate("k-NN", "pointwise"), SPACE, KNOBS, TRAINING, {}, 3)
    assert fitter(ROWS, 0)(X_VALID).shape[0] == len(X_VALID)


@pytest.mark.model
def test_the_residual_option_predicts_a_row_per_input() -> None:
    candidate = Candidate("k-NN", "pointwise")
    fitter = option_fitter("residual", candidate, SPACE, KNOBS, TRAINING, {}, 2)
    assert fitter(ROWS, 0)(X_VALID).shape[0] == len(X_VALID)


@pytest.mark.model
def test_the_residual_stage_does_not_raise_the_training_error_of_a_toy_knn() -> None:
    first = sklearn_fitter(lambda: KNeighborsRegressor(n_neighbors=3))
    second = sklearn_fitter(lambda: KNeighborsRegressor(n_neighbors=1))
    plain = first(ROWS, 0)(X_FIT)
    staged = residual_fitter(first, second)(ROWS, 0)(X_FIT)
    assert np.abs(staged - Y_FIT).max() <= np.abs(plain - Y_FIT).max() + 1e-12
