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
    curve_adjacency,
    draw,
    figures_tex,
    grid_fill,
    main,
    neighbour_distances,
    numbers,
    render_macros,
    split_leakage,
    summarize,
    with_charge_targets,
)

from shared.config import PaperConfig
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


def test_charge_ratio_is_d_over_m() -> None:
    table = pd.DataFrame({"M": [2.0], "D": [0.5]})
    assert with_charge_targets(table)["D_over_M"].tolist() == [0.25]


def test_log10_charge_ratio() -> None:
    table = pd.DataFrame({"M": [2.0], "D": [0.2]})
    assert with_charge_targets(table)["log10_D_over_M"].tolist() == pytest.approx([-1.0])


class TestSummarize:
    @pytest.fixture
    def summaries(self):
        return {(s.column, s.scale): s for s in summarize(TABLE, ("rho_c",))}

    def test_raw_maximum(self, summaries) -> None:
        assert summaries["rho_c", "raw"].maximum == 100.0

    def test_log10_mean(self, summaries) -> None:
        assert summaries["rho_c", "log10"].mean == pytest.approx(1.5)

    def test_one_summary_per_column_and_scale(self, summaries) -> None:
        assert len(summaries) == 2


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


def test_rows_of_dense_curves_neighbour_their_own_curve() -> None:
    dense = {(b, 1.0): [0.1] * 6 for b in (1.0, 50.0, 100.0)}
    assert curve_adjacency(curves(dense), k=3).same_curve_share == pytest.approx(1.0)


def test_a_row_random_split_scores_better_than_a_grouped_one() -> None:
    offsets = {(1.0, 1.0): 1.0, (2.0, 1.0): 9.0, (3.0, 1.0): 2.0, (4.0, 1.0): 8.0}
    table = curves({key: [off] * 6 for key, off in offsets.items()}).rename(columns={"D": "y"})
    leakage = split_leakage(table, "y", folds=2, seed=0)
    assert leakage.mae_rows < leakage.mae_groups


def test_macros_render_one_newcommand_per_number() -> None:
    assert render_macros({"nsEdaX": "1.5"}) == "\\newcommand{\\nsEdaX}{1.5}\n"


DENSE = curves({(b, lam): [0.1, 0.2, 0.3, 0.4] for b in (1.0, 50.0) for lam in (1.0, 2.0)})


def test_every_row_of_a_dense_curve_has_a_within_curve_distance() -> None:
    assert len(neighbour_distances(DENSE, k=3).within) == 16


def test_numbers_name_the_grid_fill_in_percent() -> None:
    assert numbers(with_charge_targets(DENSE), folds=2, seed=0)["nsEdaGridFillPercent"] == "100.0"


def test_figures_tex_includes_each_selected_figure() -> None:
    assert "\\includegraphics[width=\\textwidth]{41_neutron_stars_grid_fill}" in figures_tex(
        ("grid_fill",)
    )


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


def test_main_writes_each_selected_figure(tmp_path: Path) -> None:
    table = tmp_path / "ns.parquet"
    DENSE.to_parquet(table)
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "data_analysis": {"neutron_stars": {"figures": ["grid_fill"]}}}
    )
    main([str(table), str(tmp_path)], config=config, folds=2)
    assert sorted(p.name for p in tmp_path.glob("41_neutron_stars_*")) == [
        "41_neutron_stars_figures.tex",
        "41_neutron_stars_grid_fill.pdf",
        "41_neutron_stars_numbers.tex",
    ]
