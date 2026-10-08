"""The curve-wise representation as a harness fitter (W-066).

The data are dense along each curve and sparse across curves (W-064), so instead of mapping
every row to its target, a curve is fitted once along a normalized coordinate s in [0, 1] -- a
least-squares cubic B-spline on fixed knots -- and a base model learns, from the curve keys
alone, the spline coefficients and the curve's range along its coordinate. A row is predicted by
rebuilding its curve from its keys and evaluating it at the row's s. The range is predicted, not
read from the rows to predict: a held-out curve's range ends where its data were cut (NS: at
M_max), which the target decides.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.interpolate import BSpline

from shared.harness import Fitter


def curvewise_fitter(
    make_base: Callable[[], Any], keys: Sequence[int], along: int, knots: int
) -> Fitter:
    """A harness fitter of the curve-wise representation: keys are the columns of the curve
    keys, along the column of the coordinate along a curve, knots the spline's knots in s
    (ends included); make_base builds a multi-output scikit-learn regressor."""
    keys = list(keys)
    t = np.concatenate([np.zeros(3), np.linspace(0.0, 1.0, knots), np.ones(3)])

    def fit(
        x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, seed: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        curve_keys, rows_of = np.unique(x_fit[:, keys], axis=0, return_inverse=True)
        outputs = []
        for curve in range(len(curve_keys)):
            mine = rows_of == curve
            a, y = x_fit[mine, along], y_fit[mine]
            order = np.argsort(a)
            start, end = float(a.min()), float(a.max())
            s = (a[order] - start) / (end - start)
            # Least squares on the B-spline basis; a curve with fewer rows than coefficients
            # gets the minimum-norm solution, so every curve yields the full vector (W-087).
            basis = BSpline.design_matrix(s, t, 3).toarray()
            coefficients = np.linalg.lstsq(basis, y[order], rcond=None)[0]
            outputs.append([*coefficients, start, end])
        base = make_base().fit(curve_keys, np.array(outputs))

        def predict(x: np.ndarray) -> np.ndarray:
            wanted, rows = np.unique(x[:, keys], axis=0, return_inverse=True)
            predicted = base.predict(wanted)
            y = np.empty(len(x))
            for curve, out in enumerate(predicted):
                mine = rows == curve
                start, end = out[-2], out[-1]
                s = (x[mine, along] - start) / (end - start)
                y[mine] = BSpline(t, out[:-2], 3)(s)
            return y

        return predict

    return fit
