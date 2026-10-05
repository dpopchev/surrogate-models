"""Facts about the NS exploratory computations, on tiny synthetic curves."""

import pandas as pd
import pytest
from eda_neutron_stars import (
    beta_share_at_fixed_lambda,
    charge_targets,
    curve_adjacency,
    grid_fill,
    render_macros,
    split_leakage,
    summarize,
    with_charge_targets,
)


def curves(target: dict[tuple[float, float], list[float]]) -> pd.DataFrame:
    """Curves keyed by (beta, lambda), each with rho_c = 10, 100, 1000, ...; M = 1, D = target."""
    rows = [
        {"beta": b, "lambda": lam, "rho_c": 10.0 ** (i + 1), "M": 1.0, "D": value}
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
