"""Facts about the curve-wise fitter, on synthetic curves that are cubic along their coordinate."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from shared.curvewise import curvewise_fitter


def curves(keys: list[float], rows: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Rows of one curve per key p: the coordinate a runs from p to p + 2 (s from 0 to 1) and
    the target (1 + p) s^3 + p s + 1 is cubic in s; the columns are (a, p)."""
    s = np.linspace(0.0, 1.0, rows)
    x = np.vstack([np.column_stack([p + 2.0 * s, np.full_like(s, p)]) for p in keys])
    p, s_all = x[:, 1], (x[:, 0] - x[:, 1]) / 2.0
    return x, (1.0 + p) * s_all**3 + p * s_all + 1.0


def test_an_exact_base_reproduces_the_rows_it_was_fitted_on() -> None:
    x, y = curves([0.0, 1.0, 2.0, 3.0])
    fit = curvewise_fitter(lambda: KNeighborsRegressor(n_neighbors=1), keys=[1], along=0, knots=4)
    predict = fit(x, y, x[:0], y[:0], 0)
    assert predict(x).tolist() == pytest.approx(y.tolist(), abs=1e-9)


def test_a_curve_shorter_than_its_coefficients_still_lets_the_others_be_reproduced() -> None:
    # Four knots give six coefficients; the curve p = 4 has four rows only (W-087).
    long_x, long_y = curves([0.0, 1.0, 2.0, 3.0])
    short_x, short_y = curves([4.0], rows=4)
    x, y = np.vstack([long_x, short_x]), np.concatenate([long_y, short_y])
    fit = curvewise_fitter(lambda: KNeighborsRegressor(n_neighbors=1), keys=[1], along=0, knots=4)
    predict = fit(x, y, x[:0], y[:0], 0)
    assert predict(long_x).tolist() == pytest.approx(long_y.tolist(), abs=1e-9)


def test_a_curve_unseen_in_the_fit_rows_is_predicted_from_its_keys() -> None:
    # The spline coefficients and the range are linear in p here, so a linear base is exact.
    x, y = curves([0.0, 1.0, 3.0, 4.0])
    unseen_x, unseen_y = curves([2.0])
    fit = curvewise_fitter(LinearRegression, keys=[1], along=0, knots=4)
    predict = fit(x, y, x[:0], y[:0], 0)
    assert predict(unseen_x).tolist() == pytest.approx(unseen_y.tolist(), abs=1e-9)
