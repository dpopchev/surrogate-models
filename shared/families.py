"""The model families a screen scores through the harness (W-066, W-067).

Each factory returns a scikit-learn estimator that shared/surrogate.py's sklearn_fitter fits on
the harness's fit rows. The inputs are standardized on those rows, so a distance means the same
in every direction whatever the units.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_knn(k: int) -> Pipeline:
    """k nearest neighbours weighted by inverse distance, on standardized inputs."""
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    return Pipeline([("scale", StandardScaler()), ("knn", knn)])


class LocalRBF(RegressorMixin, BaseEstimator):
    """A thin-plate-spline RBF interpolant built from each point's nearest neighbours only: a
    global RBF on about 1e5 rows costs the cube of the rows."""

    def __init__(self, neighbours: int = 50) -> None:
        self.neighbours = neighbours

    def fit(self, x: np.ndarray, y: np.ndarray) -> LocalRBF:
        """Keep the interpolant of the rows, and its fallback for a planar neighbourhood."""
        rows, target = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.interpolant_ = RBFInterpolator(
            rows, target, neighbors=self.neighbours, kernel="thin_plate_spline"
        )
        # The thin-plate's linear part needs neighbours that span the inputs; a constant part,
        # as the linear kernel at degree 0 has, never loses rank (W-086).
        self.fallback_ = RBFInterpolator(
            rows, target, neighbors=self.neighbours, kernel="linear", degree=0
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The interpolant at each row, the fallback where its neighbourhood is planar."""
        return self._predict(np.asarray(x, dtype=np.float64))

    def _predict(self, rows: np.ndarray) -> np.ndarray:
        """The interpolant at the rows; a batch it refuses is halved until the rows it still
        refuses are single ones, which the fallback predicts."""
        try:
            return self.interpolant_(rows)
        except np.linalg.LinAlgError:
            if len(rows) == 1:
                return self.fallback_(rows)
            half = len(rows) // 2
            return np.concatenate([self._predict(rows[:half]), self._predict(rows[half:])])


def make_rbf(neighbours: int) -> Pipeline:
    """The local RBF on standardized inputs."""
    return Pipeline([("scale", StandardScaler()), ("rbf", LocalRBF(neighbours))])
