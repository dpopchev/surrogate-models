"""Facts about the BH exploratory computations, on tiny synthetic curves."""

import pandas as pd
import pytest
from eda_black_holes import (
    beta_effect,
    charge_targets,
    evidence,
    existence_edge,
    mass_correction,
    numbers,
    split_strategies,
    with_mass_correction,
)

from shared.eda import number_tex, with_charge_targets


def curves(start_step: float = 0.5) -> pd.DataFrame:
    """Three beta curves on an r_h grid of step 0.5 up to 3.0, each starting later in r_h;
    M = r_h/2 + 0.01 beta, D = 0.1 beta / r_h."""
    rows = [
        {"beta": b, "r_h": r, "M": r / 2 + 0.01 * b, "D": 0.1 * b / r}
        for b in (1.0, 2.0, 3.0)
        for r in [1.0 + start_step * (b - 1) + 0.5 * i for i in range(5)]
        if r <= 3.0
    ]
    return pd.DataFrame(rows)


TABLE = with_mass_correction(with_charge_targets(curves()))


def test_mass_correction_is_m_minus_half_the_horizon_radius() -> None:
    assert TABLE["M_correction"].iloc[0] == pytest.approx(0.01)


def test_existence_edge_lists_where_each_curve_starts() -> None:
    assert [r for _, r in existence_edge(TABLE).starts] == [1.0, 1.5, 2.0]


def test_a_target_free_of_beta_has_no_beta_effect() -> None:
    assert beta_effect(TABLE.assign(y=TABLE["r_h"]), "y").spread_ratio == pytest.approx(0.0)


def test_a_target_driven_by_beta_has_a_beta_effect() -> None:
    assert beta_effect(TABLE.assign(y=TABLE["beta"]), "y").spread_ratio > 0.5


def test_the_mass_correction_follows_beta_here() -> None:
    assert mass_correction(TABLE).pearson_beta == pytest.approx(1.0)


def test_charge_targets_are_compared_in_order() -> None:
    assert [c.target for c in charge_targets(TABLE)] == [
        "D",
        "log10_D",
        "D_over_M",
        "log10_D_over_M",
    ]


def test_d_falls_as_r_h_grows_here() -> None:
    assert charge_targets(TABLE)[0].pearson_rh < 0


SCORES = evidence(TABLE, folds=3, seed=0).splits


class TestNumbers:
    @pytest.fixture
    def found(self):
        return numbers(evidence(TABLE, folds=3, seed=0))

    def test_existence_edge_low_end(self, found) -> None:
        assert found["bhEdaEdgeRhMinLow"] == "1.000"

    def test_existence_edge_high_end(self, found) -> None:
        assert found["bhEdaEdgeRhMinHigh"] == "2.000"

    def test_curves_are_counted(self, found) -> None:
        assert found["bhEdaCurves"] == "3"

    def test_the_r_h_grid_step(self, found) -> None:
        assert found["bhEdaRhStep"] == "0.500"

    def test_the_beta_effect_on_m_is_in_percent(self, found) -> None:
        assert found["bhEdaBetaEffectPercentM"].replace(".", "").isdigit()

    def test_split_errors_use_the_adaptive_number_format(self, found) -> None:
        score = next(s for s in SCORES if (s.strategy, s.target) == ("curves", "M_correction"))
        assert found["bhEdaSplitCurvesMC"] == number_tex(score.mae)

    def test_split_errors_are_named_per_strategy_and_target(self, found) -> None:
        assert {"bhEdaSplitOuterCurvesMC", "bhEdaSplitCurvesLD"} <= found.keys()


def test_split_strategies_score_every_strategy_in_order() -> None:
    scores = split_strategies(TABLE, "M", folds=3, seed=0)
    assert [s.strategy for s in scores] == ["random_rows", "curves", "outer_curves"]
