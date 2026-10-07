"""Facts about the data ceilings, on tiny synthetic curves."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from shared.ceilings import (
    Ceilings,
    Spread,
    across_curve_errors,
    along_curve_errors,
    ceiling_figure,
    ceiling_macros,
    ceiling_table,
    decade_table,
    measure,
    noise_proxy,
    profile,
    spread,
    uncertainty_macros,
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


def test_spread_gives_the_mean_of_the_errors() -> None:
    assert spread(np.array([0.1, 0.2, 0.3, 1.0])).mean == pytest.approx(0.4)


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
    scored = across_curve_errors(cubic, [], "p", "x", "y", neighbours=2)
    assert scored.errors.max() == pytest.approx(0.0, abs=1e-9)


def test_linear_interpolation_across_misses_the_curvature_in_p() -> None:
    # log y = 0.1 p^2: the mean of the neighbours at p - 1 and p + 1 is 0.1 too high.
    scored = across_curve_errors(family(lambda p: 0.1 * p**2), [], "p", "x", "y", neighbours=1)
    assert scored.errors.min() == pytest.approx(np.expm1(0.1), rel=1e-6)


def test_curves_are_held_out_within_each_group_of_fixed_keys() -> None:
    # Two groups q = 1, 2 of three curves each: one interior curve per group, 12 rows each.
    three = family(lambda p: 0.1 * p)
    three = three[three["p"] <= 3.0]
    grouped = pd.concat([three.assign(q=1.0), three.assign(q=2.0)])
    scored = across_curve_errors(grouped, ["q"], "p", "x", "y", neighbours=1)
    assert len(scored.errors) == 2 * len(X)


def test_across_curve_errors_return_the_true_target_of_the_scored_rows() -> None:
    linear = family(lambda p: 0.1 * p)
    scored = across_curve_errors(linear[linear["p"] <= 3.0], [], "p", "x", "y", neighbours=1)
    expected = linear.loc[linear["p"] == 2.0, "y"].to_numpy()
    assert scored.truth.tolist() == pytest.approx(expected.tolist())


def test_across_curve_errors_return_the_position_of_the_scored_rows() -> None:
    linear = family(lambda p: 0.1 * p)
    scored = across_curve_errors(linear[linear["p"] <= 3.0], [], "p", "x", "y", neighbours=1)
    assert scored.position.tolist() == pytest.approx(X.tolist())


def test_a_profile_bins_the_errors_by_position_into_equal_counts() -> None:
    position, errors = np.arange(8.0), np.array([1.0, 3.0, 5.0, 7.0, 2.0, 2.0, 4.0, 4.0])
    assert profile(position, errors, bins=4).median == (2.0, 6.0, 2.0, 4.0)


def grid_family() -> pd.DataFrame:
    """A 7 x 7 grid of curves (p, q) on the x grid; log y = x + 0.1 p + 0.05 q, z = 2 y."""
    p, q = np.meshgrid(np.arange(1.0, 8.0), np.arange(1.0, 8.0))
    rows = [
        {"p": a, "q": b, "x": x, "y": float(np.exp(x + 0.1 * a + 0.05 * b))}
        for a, b in zip(p.ravel(), q.ravel(), strict=True)
        for x in X
    ]
    return pd.DataFrame(rows).assign(z=lambda t: 2.0 * t["y"])


@pytest.fixture(scope="module")
def found() -> Ceilings:
    return measure(grid_family(), ["p", "q"], "x", ["y", "z"], decade_target="y")


class TestMeasure:
    def test_varies_each_curve_key_across_curves_for_each_target(self, found) -> None:
        assert sorted(found.across) == [("y", "p"), ("y", "q"), ("z", "p"), ("z", "q")]

    def test_profiles_the_error_along_the_curves_for_each_target_and_key(self, found) -> None:
        assert sorted(found.profiles) == [("y", "p"), ("y", "q"), ("z", "p"), ("z", "q")]


TAGS = {"y": "Y", "z": "Z", "p": "P", "q": "Q"}
KNOWN = Ceilings(
    noise={"y": Spread(1e-8, 2e-8, 3e-8, 1e-8)},
    along={"y": Spread(1e-7, 2e-6, 3e-5, 1e-6)},
    across={
        ("y", "p"): Spread(1e-6, 1e-4, 1e-3, 1e-5),
        ("y", "q"): Spread(1e-6, 1e-3, 1e-2, 1e-4),
    },
    profiles={},
    decades={"p": {-3: Spread(4e-4, 1e-3, 2e-3, 5e-4)}, "q": {-3: Spread(2e-4, 5e-4, 1e-3, 3e-4)}},
)


class TestCeilingMacros:
    def test_name_the_across_curve_p95_per_target_and_key(self) -> None:
        assert ceiling_macros("ex", KNOWN, TAGS)["exCeilYAcrossQ"] == "1.00\\times 10^{-3}"

    def test_name_the_figures_the_worst_direction_across_allows(self) -> None:
        # p95 across: 1e-4 along p, 1e-3 along q -> the worse, 1e-3, keeps 3.0 figures.
        assert ceiling_macros("ex", KNOWN, TAGS)["exCeilYFigures"] == "3.0"

    def test_name_the_median_noise_per_target(self) -> None:
        assert ceiling_macros("ex", KNOWN, TAGS)["exCeilYNoise"] == "1.00\\times 10^{-8}"

    def test_name_the_along_curve_p95_per_target(self) -> None:
        assert ceiling_macros("ex", KNOWN, TAGS)["exCeilYAlong"] == "2.00\\times 10^{-6}"


class TestUncertaintyMacros:
    # p95 across: 1e-4 along p, 1e-3 along q -> the worse, 1e-3, is the relative uncertainty.
    def test_the_sigma_is_the_worst_p95_times_the_typical_value(self) -> None:
        macros = uncertainty_macros("ex", KNOWN, TAGS, {"y": {"": 2.0}}, decade_target="y")
        assert macros["exCeilYSigma"] == "2.00\\times 10^{-3}"

    def test_the_decade_target_s_sigma_in_dex_is_the_worst_p95_over_ln_10(self) -> None:
        macros = uncertainty_macros("ex", KNOWN, TAGS, {"y": {"": 2.0}}, decade_target="y")
        assert macros["exCeilYSigmaDex"] == "4.34\\times 10^{-4}"

    def test_names_the_typical_value_each_uncertainty_is_taken_at(self) -> None:
        macros = uncertainty_macros("ex", KNOWN, TAGS, {"y": {"Max": 2.19}}, decade_target="z")
        assert macros["exCeilYAtMax"] == "2.19"


LABELS = {"y": "$Y$", "z": "$Z$", "p": "$p$", "q": "$q$"}


def test_the_ceiling_table_gives_one_row_per_target() -> None:
    row = (
        "$Y$ & $1.00\\times 10^{-8}$ & $2.00\\times 10^{-6}$ & $1.00\\times 10^{-4}$ & "
        "$1.00\\times 10^{-3}$ & 3.0"
    )
    assert row in ceiling_table("ex-ceilings", "Ceilings.", KNOWN, LABELS)


def test_the_decade_table_gives_the_median_per_direction_and_the_figures_of_the_worse() -> None:
    # Medians 4e-4 (across p) and 2e-4 (across q): the worse keeps -log10(4e-4) = 3.4 figures.
    row = "$10^{-3}$ & $4.00\\times 10^{-4}$ & $2.00\\times 10^{-4}$ & 3.4"
    assert row in decade_table("ex-decades", "Per decade.", KNOWN, LABELS)


class TestCeilingFigure:
    @pytest.fixture(scope="class")
    @staticmethod
    def figure(found):
        fig = ceiling_figure(found, LABELS, "$x$", {"p": "C0", "q": "C1"})
        yield fig
        plt.close(fig)

    def test_draws_one_panel_per_target(self, figure) -> None:
        assert len(figure.axes) == 2

    def test_puts_the_relative_error_on_a_log_axis(self, figure) -> None:
        assert figure.axes[0].get_yscale() == "log"


TOO_FEW = family(lambda p: 0.1 * p).query("p <= 2.0 and x <= 0.3")  # 2 curves of 4 rows


class TestTooFewCurves:
    def test_measure_leaves_the_bounds_undefined(self) -> None:
        found = measure(TOO_FEW, ["p"], "x", ["y"], decade_target="y")
        assert np.isnan(found.across["y", "p"].p95)

    def test_the_macros_show_an_undefined_bound_as_a_dash(self) -> None:
        found = measure(TOO_FEW, ["p"], "x", ["y"], decade_target="y")
        assert ceiling_macros("ex", found, TAGS)["exCeilYAcrossP"] == "--"


def test_measure_keys_the_decade_ceilings_by_floor_log10_of_the_target(found) -> None:
    # y = exp(x + 0.1 p + 0.05 q) lies in [1.2, 7.8] on the held-out rows: decade 0 alone.
    assert sorted(found.decades["p"]) == [0]
