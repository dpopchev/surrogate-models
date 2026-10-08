"""Facts about the model families, on tiny synthetic data."""

import numpy as np
import pytest

from shared.families import make_gpr, make_knn, make_rbf, make_xgboost

# A 6 x 6 grid of rows and a target linear in both inputs.
GRID = np.array([[a, b] for a in range(6) for b in range(6)], dtype=float)


def linear(rows: np.ndarray) -> np.ndarray:
    """A target linear in both inputs."""
    return 2.0 * rows[:, 0] - 3.0 * rows[:, 1] + 1.0


def test_the_local_rbf_reproduces_a_linear_target_between_the_rows() -> None:
    between = np.array([[2.5, 2.5], [1.5, 3.5]])
    predicted = make_rbf(neighbours=12).fit(GRID, linear(GRID)).predict(between)
    assert predicted.tolist() == pytest.approx(linear(between).tolist())


# Rows on two parallel lines in three inputs: every neighbourhood lies in one plane, so the
# thin-plate's linear part [1, x1, x2, x3] has no full rank there (W-086).
TWO_LINES = np.array([[a, b, 0.0] for b in (0.0, 1.0) for a in range(6)], dtype=float)


def test_the_local_rbf_predicts_from_a_planar_neighbourhood() -> None:
    model = make_rbf(neighbours=12).fit(TWO_LINES, TWO_LINES[:, 0] + TWO_LINES[:, 1])
    assert np.isfinite(model.predict(np.array([[2.5, 0.5, 0.0]]))).all()


# A 4 x 4 x 4 block of rows, whose neighbourhoods span the inputs, and the two lines far away.
BLOCK = np.array([[a, b, c] for a in range(4) for b in range(4) for c in range(4)], dtype=float)
BLOCK_AND_LINES = np.vstack([BLOCK, TWO_LINES + np.array([100.0, 0.0, 0.0])])


def linear3(rows: np.ndarray) -> np.ndarray:
    """A target linear in three inputs."""
    return 2.0 * rows[:, 0] - 3.0 * rows[:, 1] + rows[:, 2] + 1.0


def test_a_spanning_query_keeps_the_thin_plate_beside_a_planar_one() -> None:
    model = make_rbf(neighbours=12).fit(BLOCK_AND_LINES, linear3(BLOCK_AND_LINES))
    queries = np.array([[1.5, 1.5, 1.5], [102.5, 0.5, 0.0]])
    assert model.predict(queries)[0] == pytest.approx(linear3(queries)[0])


def test_the_gpr_reproduces_a_target_it_was_fitted_on() -> None:
    model = make_gpr().fit(GRID, linear(GRID))
    assert model.predict(GRID[7:8]).tolist() == pytest.approx(linear(GRID[7:8]).tolist(), abs=1e-3)


def test_xgboost_predicts_one_value_per_row() -> None:
    model = make_xgboost().fit(GRID, linear(GRID))
    assert model.predict(GRID[:3]).shape == (3,)


def test_knn_with_one_neighbour_returns_the_target_of_the_row_it_is_given() -> None:
    x, y = np.array([[0.0], [1.0], [2.0]]), np.array([10.0, 20.0, 30.0])
    assert make_knn(1).fit(x, y).predict(np.array([[1.0]])).tolist() == [20.0]
