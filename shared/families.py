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
        """Keep the interpolant of the rows."""
        self.interpolant_ = RBFInterpolator(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            neighbors=self.neighbours,
            kernel="thin_plate_spline",
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The interpolant at each row."""
        return self.interpolant_(np.asarray(x, dtype=np.float64))


def make_rbf(neighbours: int) -> Pipeline:
    """The local RBF on standardized inputs."""
    return Pipeline([("scale", StandardScaler()), ("rbf", LocalRBF(neighbours))])
