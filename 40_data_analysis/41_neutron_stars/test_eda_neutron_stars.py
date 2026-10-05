"""Facts about the NS exploratory computations, on tiny synthetic curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from eda_neutron_stars import (
    beta_share_at_fixed_lambda,
    charge_targets,
    draw,
    evidence,
    figure_tex,
    grid_fill,
    main,
    numbers,
    pile_up_share,
    split_strategies,
    table_tex,
)

from shared.config import PaperConfig
from shared.eda import with_charge_targets
from shared.plots import PlotStyle


def curves(target: dict[tuple[float, float], list[float]]) -> pd.DataFrame:
    """Curves keyed by (beta, lambda): rho_c = 10, 100, ...; M = 1, 1.1, ...; D = target."""
    rows = [
        {"beta": b, "lambda": lam, "rho_c": 10.0 ** (i + 1), "M": 1.0 + 0.1 * i, "D": value}
        for (b, lam), values in target.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


TABLE = curves({(1.0, 1.0): [0.1, 0.2], (2.0, 1.0): [0.3, 0.4], (1.0, 2.0): [0.5, 0.6]})


def test_charge_targets_compare_log10_d_and_log10_d_over_m() -> None:
    table = with_charge_targets(TABLE.assign(M=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]))
    assert [c.target for c in charge_targets(table)] == ["log10_D", "log10_D_over_M"]


def test_charge_target_correlates_with_m() -> None:
    table = with_charge_targets(TABLE.assign(M=[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]))
    table = table.assign(log10_D=[1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    assert charge_targets(table)[0].pearson_m == pytest.approx(1.0)


def test_beta_explains_all_when_the_target_depends_only_on_beta() -> None:
    table = curves({(1.0, 1.0): [0.1, 0.1], (2.0, 1.0): [0.9, 0.9]}).assign(y=[1.0, 1.0, 5.0, 5.0])
    assert beta_share_at_fixed_lambda(table, "y") == pytest.approx(1.0)


def test_beta_explains_nothing_when_the_target_follows_rho_c_only() -> None:
    table = curves({(1.0, 1.0): [0.1, 0.1], (2.0, 1.0): [0.9, 0.9]}).assign(y=[1.0, 5.0, 1.0, 5.0])
    assert beta_share_at_fixed_lambda(table, "y") == pytest.approx(0.0)


def test_grid_fill_counts_curves_over_the_product_grid() -> None:
    assert grid_fill(TABLE).fill == pytest.approx(0.75)


def test_a_row_random_split_scores_better_than_a_grouped_one() -> None:
    offsets = {(1.0, 1.0): 1.0, (2.0, 1.0): 9.0, (3.0, 1.0): 2.0, (4.0, 1.0): 8.0}
    table = curves({key: [off] * 6 for key, off in offsets.items()}).rename(columns={"D": "y"})
    scores = {s.strategy: s.mae for s in split_strategies(table, "y", folds=2, seed=0)}
    assert scores["random_rows"] < scores["curves"]


DENSE = curves({(b, lam): [0.1, 0.2, 0.3, 0.4] for b in (1.0, 50.0) for lam in (1.0, 2.0)})


def test_numbers_name_the_grid_fill_in_percent() -> None:
    found = numbers(evidence(with_charge_targets(DENSE), folds=2, seed=0))
    assert found["nsEdaGridFillPercent"] == "100.0"


def test_figure_tex_includes_its_figure() -> None:
    assert "\\includegraphics[width=\\textwidth]{41_neutron_stars_grid_fill}" in figure_tex(
        "grid_fill"
    )


def grid(betas: range, lambdas: range, missing: tuple[tuple[int, int], ...] = ()) -> pd.DataFrame:
    """A beta x lambda grid of 4-row curves whose D follows beta, lambda and the row."""
    return curves(
        {
            (float(b), float(lam)): [0.1 * b + 0.01 * lam + 0.001 * i for i in range(4)]
            for b in betas
            for lam in lambdas
            if (b, lam) not in missing
        }
    )


def test_split_strategies_score_every_strategy_in_order() -> None:
    scores = split_strategies(grid(range(1, 4), range(1, 4)), "D", folds=3, seed=0)
    assert [s.strategy for s in scores] == [
        "random_rows",
        "curves",
        "beta_lines",
        "lambda_lines",
        "rim",
    ]


def test_numbers_name_the_rim_score_of_the_charge_target() -> None:
    table = with_charge_targets(grid(range(1, 4), range(1, 4)))
    assert "nsEdaSplitRimDM" in numbers(evidence(table, folds=3, seed=0))


def test_numbers_give_the_charge_range_in_orders_of_magnitude() -> None:
    assert "nsEdaDOrders" in numbers(evidence(with_charge_targets(DENSE), folds=2, seed=0))


class TestGridNumbers:
    @pytest.fixture
    def found(self):
        table = with_charge_targets(DENSE.iloc[:-1])
        return numbers(evidence(table, folds=2, seed=0))

    def test_density_grid_values_count_the_distinct_central_densities(self, found) -> None:
        assert found["nsEdaDensityGridValues"] == "4"

    def test_rows_per_curve_maximum(self, found) -> None:
        assert found["nsEdaRowsPerCurveMax"] == "4"

    def test_rows_per_curve_minimum(self, found) -> None:
        assert found["nsEdaRowsPerCurveMin"] == "3"

    def test_density_orders_of_magnitude(self, found) -> None:
        assert found["nsEdaRhocOrders"] == "3.0"


def test_pile_up_share_counts_rows_within_the_window_of_their_curve_maximum() -> None:
    assert pile_up_share(DENSE, window=0.15) == pytest.approx(0.5)


GRID = evidence(with_charge_targets(grid(range(1, 4), range(1, 4))), folds=3, seed=0)


class TestChargeNumbers:
    @pytest.fixture
    def found(self):
        return numbers(GRID)

    def test_lambda_linear_share_is_the_squared_pearson_in_percent(self, found) -> None:
        expected = f"{100 * GRID.targets[1].pearson_lambda ** 2:.0f}"
        assert found["nsEdaLambdaLinearPercentLogDM"] == expected

    def test_pearson_with_m_is_named_for_both_targets(self, found) -> None:
        assert {"nsEdaPearsonMLogD", "nsEdaPearsonMLogDM"} <= found.keys()

    def test_beta_correlation_is_named_for_both_targets(self, found) -> None:
        assert {"nsEdaCorrBetaLogD", "nsEdaCorrBetaLogDM"} <= found.keys()


def test_charge_split_macros_use_the_tables_three_decimals() -> None:
    assert len(numbers(GRID)["nsEdaSplitCurvesDM"].split(".")[1]) == 3


def test_table_tex_labels_its_table() -> None:
    assert "\\label{tab:ns-split-strategies}" in table_tex(GRID, "split_strategies")


def test_univariate_table_has_a_row_for_the_central_density() -> None:
    assert "$\\rho_c$ &" in table_tex(GRID, "univariate")


STYLE = PlotStyle(usetex=False)


class TestFigures:
    @pytest.fixture(autouse=True)
    def _close(self):
        yield
        plt.close("all")

    def test_univariate_has_a_raw_and_a_log10_panel_per_column(self) -> None:
        assert len(draw("univariate", with_charge_targets(DENSE), STYLE).axes) == 12

    def test_univariate_gives_a_grid_parameter_one_bar_per_value(self) -> None:
        figure = draw("univariate", with_charge_targets(DENSE), STYLE)
        assert len(figure.axes[0].patches) == 2

    def test_charge_target_has_four_panels_and_two_colorbars(self) -> None:
        assert len(draw("charge_target", with_charge_targets(DENSE), STYLE).axes) == 6

    def test_mass_density_has_one_panel_and_its_colorbar(self) -> None:
        assert len(draw("mass_density", with_charge_targets(DENSE), STYLE).axes) == 2

    def test_grid_fill_has_one_panel_and_its_colorbar(self) -> None:
        assert len(draw("grid_fill", with_charge_targets(DENSE), STYLE).axes) == 2

    def test_curve_adjacency_has_one_panel(self) -> None:
        assert len(draw("curve_adjacency", with_charge_targets(DENSE), STYLE).axes) == 1

    def test_univariate_continuous_has_four_columns_raw_and_log10(self) -> None:
        assert len(draw("univariate_continuous", with_charge_targets(DENSE), STYLE).axes) == 8

    def test_univariate_continuous_gives_the_density_grid_one_bar_per_value(self) -> None:
        figure = draw("univariate_continuous", with_charge_targets(DENSE), STYLE)
        assert len(figure.axes[0].patches) == 4

    def test_univariate_continuous_draws_bars_without_edges(self) -> None:
        figure = draw("univariate_continuous", with_charge_targets(DENSE), STYLE)
        assert figure.axes[0].patches[0].get_linewidth() == 0

    def test_mass_max_has_two_panels_and_a_colorbar(self) -> None:
        assert len(draw("mass_max", with_charge_targets(DENSE), STYLE).axes) == 3


def test_main_writes_each_selected_figure_and_drops_stale_ones(tmp_path: Path) -> None:
    table = tmp_path / "ns.parquet"
    DENSE.to_parquet(table)
    (tmp_path / "41_neutron_stars_univariate.pdf").write_text("deselected since the last run")
    selection = {"figures": ["grid_fill"], "tables": ["split_strategies"]}
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "data_analysis": {"neutron_stars": selection}}
    )
    main([str(table), str(tmp_path)], config=config, folds=2)
    assert sorted(p.name for p in tmp_path.glob("41_neutron_stars_*")) == [
        "41_neutron_stars_fig_grid_fill.tex",
        "41_neutron_stars_grid_fill.pdf",
        "41_neutron_stars_numbers.tex",
        "41_neutron_stars_tab_split_strategies.tex",
    ]
