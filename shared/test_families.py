"""Facts about the model families, on tiny synthetic data."""

import numpy as np
import pytest

from shared.families import make_knn, make_rbf

# A 6 x 6 grid of rows and a target linear in both inputs.
GRID = np.array([[a, b] for a in range(6) for b in range(6)], dtype=float)


def linear(rows: np.ndarray) -> np.ndarray:
    """A target linear in both inputs."""
    return 2.0 * rows[:, 0] - 3.0 * rows[:, 1] + 1.0


def test_the_local_rbf_reproduces_a_linear_target_between_the_rows() -> None:
    between = np.array([[2.5, 2.5], [1.5, 3.5]])
    predicted = make_rbf(neighbours=12).fit(GRID, linear(GRID)).predict(between)
    assert predicted.tolist() == pytest.approx(linear(between).tolist())


def test_knn_with_one_neighbour_returns_the_target_of_the_row_it_is_given() -> None:
    x, y = np.array([[0.0], [1.0], [2.0]]), np.array([10.0, 20.0, 30.0])
    assert make_knn(1).fit(x, y).predict(np.array([[1.0]])).tolist() == [20.0]
