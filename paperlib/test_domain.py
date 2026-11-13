"""Facts about the physical domain of each map, on tiny synthetic curves."""

import numpy as np
import pandas as pd
import pytest

from paperlib.domain import (
    Inside,
    Outside,
    Verdict,
    check,
    make_black_hole_domain,
    make_neutron_star_domain,
    stable_edge,
)

# Two beta curves; the edge r_h,min is 1.0 at beta 0 and 2.0 at beta 1, the curve ends at 5.0.
BLACK_HOLES = pd.DataFrame(
    {
        "beta": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "r_h": [1.0, 3.0, 5.0, 2.0, 3.0, 5.0],
    }
)

# A 2 x 2 grid of (beta, lambda) curves; each is cut at its last row (the stable edge).
NEUTRON_STARS = pd.DataFrame(
    {
        "beta": [0.0] * 3 + [0.0] * 3 + [1.0] * 3 + [1.0] * 3,
        "lambda": [0.0] * 3 + [1.0] * 3 + [0.0] * 3 + [1.0] * 3,
        "rho_c": [1.0, 2.0, 4.0] * 4,
    }
)


def reason_of(verdict: Verdict) -> str:
    """The reason of a refusal; an Inside verdict has none."""
    return verdict.reason if isinstance(verdict, Outside) else "inside"


def nearest_of(verdict: Verdict) -> dict[str, float]:
    """The nearest valid input of a refusal; fails the test on an Inside verdict."""
    assert isinstance(verdict, Outside), "expected a refusal"
    return dict(verdict.nearest)


@pytest.mark.model
def test_query_below_the_existence_edge_is_outside() -> None:
    verdict = check(make_black_hole_domain(BLACK_HOLES), {"beta": 1.0, "r_h": 1.5})
    assert reason_of(verdict) == "existence edge"


@pytest.mark.model
def test_query_on_a_sampled_row_is_inside() -> None:
    assert check(make_black_hole_domain(BLACK_HOLES), {"beta": 0.0, "r_h": 3.0}) == Inside()


@pytest.mark.model
def test_edge_between_sampled_betas_is_interpolated() -> None:
    verdict = check(make_black_hole_domain(BLACK_HOLES), {"beta": 0.5, "r_h": 1.2})
    assert verdict == Outside("existence edge", {"beta": 0.5, "r_h": 1.5})


@pytest.mark.model
def test_beta_outside_the_sampled_range_is_refused_as_coupling_range() -> None:
    verdict = check(make_black_hole_domain(BLACK_HOLES), {"beta": 2.0, "r_h": 3.0})
    assert verdict == Outside("coupling range", {"beta": 1.0, "r_h": 3.0})


@pytest.mark.model
def test_nearest_of_a_refused_query_is_inside() -> None:
    domain = make_black_hole_domain(BLACK_HOLES)
    verdict = check(domain, {"beta": 1.0, "r_h": 1.5})
    assert check(domain, nearest_of(verdict)) == Inside()


@pytest.mark.model
def test_neutron_star_query_above_the_stable_edge_is_outside() -> None:
    verdict = check(
        make_neutron_star_domain(NEUTRON_STARS), {"beta": 0.5, "lambda": 0.5, "rho_c": 4.5}
    )
    assert verdict == Outside("stable branch", {"beta": 0.5, "lambda": 0.5, "rho_c": 4.0})


@pytest.mark.model
def test_neutron_star_query_inside_the_curve_is_inside() -> None:
    domain = make_neutron_star_domain(NEUTRON_STARS)
    assert check(domain, {"beta": 0.5, "lambda": 0.5, "rho_c": 3.0}) == Inside()


@pytest.mark.model
def test_stable_edge_is_the_mean_of_two_neighbouring_edges() -> None:
    # Three non-collinear curves; the last rho_c of each is 4, 6 and 8, the first 1, 2 and 3.
    table = pd.DataFrame(
        {
            "beta": [0.0] * 2 + [1.0] * 2 + [0.0] * 2,
            "lambda": [0.0] * 2 + [0.0] * 2 + [1.0] * 2,
            "rho_c": [1.0, 4.0, 2.0, 6.0, 3.0, 8.0],
        }
    )
    domain = make_neutron_star_domain(table)
    low, high = stable_edge(domain, 0.5, 0.5)  # midway between (1, 0) and (0, 1)
    off_low, off_high = stable_edge(domain, 1.0, 1.0)  # in the box, off the hull
    assert (low, high, *np.isnan([off_low, off_high])) == pytest.approx((2.5, 7.0, True, True))
