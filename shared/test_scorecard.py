"""Facts about the scorecard, on hand-made arrays."""

import numpy as np
import pytest

from shared.scorecard import relative_errors, ripple, scorecard, significant_figures


def test_the_mass_error_is_relative_to_the_true_mass() -> None:
    errors = relative_errors(np.array([2.0, 4.0]), np.array([2.2, 3.6]), "mass")
    assert errors.tolist() == pytest.approx([0.1, 0.1])


def test_the_charge_error_is_relative_to_d_rebuilt_with_the_true_mass() -> None:
    y_true = np.array([-1.0, -6.0])
    errors = relative_errors(y_true, y_true + np.log10(1.1), "charge")
    assert errors.tolist() == pytest.approx([0.1, 0.1])


def test_a_relative_error_of_one_in_a_thousand_keeps_three_figures() -> None:
    assert significant_figures(1e-3) == pytest.approx(3.0)


ERRORS = np.array([0.1, 0.2, 0.3, 0.4])
ZONES = {
    "interior": np.array([True, True, False, False]),
    "rim": np.array([False, False, True, True]),
}


def test_the_scorecard_spreads_the_errors_of_each_zone() -> None:
    assert scorecard(ERRORS, ZONES, charge=None).zones["rim"].median == pytest.approx(0.35)


def test_the_scorecard_spreads_the_errors_per_decade_of_the_charge() -> None:
    charge = np.array([2e-7, 5e-7, 3e-2, 4e-2])
    assert scorecard(ERRORS, ZONES, charge).decades[-7].median == pytest.approx(0.15)


# Five grid lines p = 1..5 at two fixed points x = 0, 1 (rows interleaved).
ACROSS = np.tile(np.arange(1.0, 6.0), 2)
FIXED = np.repeat([0.0, 1.0], 5)


def test_ripple_is_zero_for_a_residual_linear_across_the_grid() -> None:
    assert ripple(0.3 + 0.1 * ACROSS, ACROSS, FIXED) == pytest.approx(0.0)


def test_ripple_measures_a_residual_alternating_across_the_grid() -> None:
    # 0, 1, 0, 1, 0 per fixed point: second differences 2, -2, 2 -> RMS 2.
    alternating = (ACROSS % 2 == 0).astype(float)
    assert ripple(alternating, ACROSS, FIXED) == pytest.approx(2.0)
