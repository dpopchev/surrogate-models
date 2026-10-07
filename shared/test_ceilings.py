"""Facts about the data ceilings, on tiny synthetic curves."""

import numpy as np
import pandas as pd
import pytest

from shared.ceilings import (
    across_curve_errors,
    along_curve_errors,
    noise_proxy,
    per_decade,
    spread,
)

# Three curves keyed by p along x: log y is cubic in x, so a cubic spline along a curve is exact.
X = np.linspace(0.0, 1.0, 12)
SMOOTH = pd.DataFrame(
    {
        "p": np.repeat([1.0, 2.0, 3.0], len(X)),
        "x": np.tile(X, 3),
        "y": np.exp(np.tile(1.0 + X - 0.5 * X**2 + 0.2 * X**3, 3)),
    }
)


def family(log_in_p) -> pd.DataFrame:
    """Seven curves p = 1..7 on the same x grid; log y = x + log_in_p(p)."""
    p = np.repeat(np.arange(1.0, 8.0), len(X))
    return pd.DataFrame({"p": p, "x": np.tile(X, 7), "y": np.exp(np.tile(X, 7) + log_in_p(p))})


def test_spread_gives_the_median_of_the_errors() -> None:
    assert spread(np.array([0.1, 0.2, 0.3, 0.4])).median == pytest.approx(0.25)


def test_noise_proxy_is_zero_for_a_log_cubic_target() -> None:
    assert noise_proxy(SMOOTH, ["p"], "x", "y").max == pytest.approx(0.0, abs=1e-12)


def test_noise_proxy_finds_relative_noise_of_its_own_order() -> None:
    jitter = np.random.default_rng(0).normal(0.0, 1e-3, len(SMOOTH))
    noisy = SMOOTH.assign(y=SMOOTH["y"] * (1.0 + jitter))
    assert 3e-4 < noise_proxy(noisy, ["p"], "x", "y").median < 3e-3


def test_a_cubic_spline_along_a_log_cubic_curve_is_exact() -> None:
    assert along_curve_errors(SMOOTH, ["p"], "x", "y").max() == pytest.approx(0.0, abs=1e-12)


def test_cubic_interpolation_across_curves_cubic_in_p_is_exact() -> None:
    cubic = family(lambda p: 0.1 * p - 0.02 * p**2 + 0.003 * p**3)
    errors, _ = across_curve_errors(cubic, [], "p", "x", "y", neighbours=2)
    assert errors.max() == pytest.approx(0.0, abs=1e-9)


def test_linear_interpolation_across_misses_the_curvature_in_p() -> None:
    # log y = 0.1 p^2: the mean of the neighbours at p - 1 and p + 1 is 0.1 too high.
    errors, _ = across_curve_errors(family(lambda p: 0.1 * p**2), [], "p", "x", "y", neighbours=1)
    assert errors.min() == pytest.approx(np.expm1(0.1), rel=1e-6)


def test_curves_are_held_out_within_each_group_of_fixed_keys() -> None:
    # Two groups q = 1, 2 of three curves each: one interior curve per group, 12 rows each.
    three = family(lambda p: 0.1 * p)
    three = three[three["p"] <= 3.0]
    grouped = pd.concat([three.assign(q=1.0), three.assign(q=2.0)])
    errors, _ = across_curve_errors(grouped, ["q"], "p", "x", "y", neighbours=1)
    assert len(errors) == 2 * len(X)


def test_across_curve_errors_return_the_true_target_of_the_scored_rows() -> None:
    linear = family(lambda p: 0.1 * p)
    _, truth = across_curve_errors(linear[linear["p"] <= 3.0], [], "p", "x", "y", neighbours=1)
    expected = linear.loc[linear["p"] == 2.0, "y"].to_numpy()
    assert truth.tolist() == pytest.approx(expected.tolist())


def test_per_decade_takes_the_median_error_in_each_decade() -> None:
    target, errors = np.array([1e-3, 2e-3, 3e-3, 0.5]), np.array([0.1, 0.3, 0.2, 0.4])
    assert per_decade(target, errors) == {-3: pytest.approx(0.2), -1: pytest.approx(0.4)}
