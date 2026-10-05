"""Facts about the BH exploratory computations, on tiny synthetic curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from eda_black_holes import (
    beta_effect,
    charge_targets,
    draw,
    evidence,
    existence_edge,
    figure_tex,
    main,
    mass_correction,
    numbers,
    split_strategies,
    table_tex,
    with_mass_correction,
)

from shared.config import PaperConfig
from shared.eda import number_tex, with_charge_targets
from shared.plots import PlotStyle


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


STYLE = PlotStyle(usetex=False)


class TestFigures:
    @pytest.fixture(autouse=True)
    def _close(self):
        yield
        plt.close("all")

    def test_univariate_continuous_has_four_columns_raw_and_log10(self) -> None:
        assert len(draw("univariate_continuous", TABLE, STYLE).axes) == 8

    def test_univariate_continuous_draws_bars_without_edges(self) -> None:
        figure = draw("univariate_continuous", TABLE, STYLE)
        assert figure.axes[0].patches[0].get_linewidth() == 0

    def test_mass_correction_has_one_panel_and_its_colorbar(self) -> None:
        assert len(draw("mass_correction", TABLE, STYLE).axes) == 2

    def test_charge_target_has_two_panels_and_a_colorbar(self) -> None:
        assert len(draw("charge_target", TABLE, STYLE).axes) == 3

    def test_existence_edge_has_one_panel(self) -> None:
        assert len(draw("existence_edge", TABLE, STYLE).axes) == 1


FOUND = evidence(TABLE, folds=3, seed=0)


def test_figure_tex_labels_its_figure() -> None:
    assert "\\label{fig:bh-existence-edge}" in figure_tex("existence_edge")


def test_table_tex_labels_its_table() -> None:
    assert "\\label{tab:bh-split-strategies}" in table_tex(FOUND, "split_strategies")


def test_univariate_table_has_a_row_for_the_horizon_radius() -> None:
    assert "$r_h$ &" in table_tex(FOUND, "univariate")


def test_charge_table_has_a_row_per_candidate_target() -> None:
    assert table_tex(FOUND, "charge_correlation").count("\\\\\n") == 5


def test_main_writes_each_selected_asset_and_drops_stale_ones(tmp_path: Path) -> None:
    table = tmp_path / "bh.parquet"
    curves().to_parquet(table)
    (tmp_path / "42_black_holes_univariate_continuous.pdf").write_text("deselected")
    selection = {"figures": ["existence_edge"], "tables": ["split_strategies"]}
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "data_analysis": {"black_holes": selection}}
    )
    main([str(table), str(tmp_path)], config=config, folds=3)
    assert sorted(p.name for p in tmp_path.glob("42_black_holes_*")) == [
        "42_black_holes_existence_edge.pdf",
        "42_black_holes_fig_existence_edge.tex",
        "42_black_holes_numbers.tex",
        "42_black_holes_tab_split_strategies.tex",
    ]


def test_split_strategies_score_every_strategy_in_order() -> None:
    scores = split_strategies(TABLE, "M", folds=3, seed=0)
    assert [s.strategy for s in scores] == ["random_rows", "curves", "outer_curves"]
