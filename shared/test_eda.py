"""Facts about the dataset-agnostic EDA core, on tiny synthetic curves."""

import numpy as np
import pandas as pd
import pytest

from shared.eda import (
    booktabs,
    curve_adjacency,
    grid_bins,
    group_splitter,
    holdout_mae,
    kfold_mae,
    make_curve_space,
    neighbour_distances,
    number_tex,
    render_macros,
    rim_mask,
    row_splitter,
    summarize,
    with_charge_targets,
)

# Curves keyed by a free parameter p, sampled along x = 10, 100, 1000, ...
SPACE = make_curve_space({"p": "raw", "x": "log10"}, ("p",))


def curves(target: dict[float, list[float]]) -> pd.DataFrame:
    """One curve per p, its y values along x = 10, 100, ..."""
    rows = [
        {"p": p, "x": 10.0 ** (i + 1), "y": value}
        for p, values in target.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_charge_ratio_is_d_over_m() -> None:
    table = pd.DataFrame({"M": [2.0], "D": [0.5]})
    assert with_charge_targets(table)["D_over_M"].tolist() == [0.25]


def test_log10_charge_ratio() -> None:
    table = pd.DataFrame({"M": [2.0], "D": [0.2]})
    assert with_charge_targets(table)["log10_D_over_M"].tolist() == pytest.approx([-1.0])


class TestSummarize:
    @pytest.fixture
    def summaries(self):
        table = curves({1.0: [0.1, 0.2], 2.0: [0.3, 0.4]})
        return {(s.column, s.scale): s for s in summarize(table, ("x",))}

    def test_raw_maximum(self, summaries) -> None:
        assert summaries["x", "raw"].maximum == 100.0

    def test_log10_mean(self, summaries) -> None:
        assert summaries["x", "log10"].mean == pytest.approx(1.5)

    def test_one_summary_per_column_and_scale(self, summaries) -> None:
        assert len(summaries) == 2


def test_a_curve_column_outside_the_inputs_is_rejected() -> None:
    with pytest.raises(ValueError, match="curve columns"):
        make_curve_space({"x": "raw"}, ("p",))


def test_rows_of_dense_curves_neighbour_their_own_curve() -> None:
    table = curves({p: [0.1] * 6 for p in (1.0, 50.0, 100.0)})
    assert curve_adjacency(table, SPACE, k=3).same_curve_share == pytest.approx(1.0)


def test_every_row_of_a_dense_curve_has_a_within_curve_distance() -> None:
    table = curves({p: [0.1] * 4 for p in (1.0, 50.0)})
    assert len(neighbour_distances(table, SPACE, k=3).within) == 8


# Two dense curves: within k = 2 every neighbour is on the own curve.
TWO = curves({p: [0.1] * 6 for p in (1.0, 2.0)})


def test_every_row_gets_an_other_curve_distance_beyond_its_k_nearest() -> None:
    assert len(neighbour_distances(TWO, SPACE, k=2).across) == 12


def test_the_other_curve_distance_is_exact_beyond_its_k_nearest() -> None:
    # p standardizes to -1 and +1, so the other curve at the same x lies 2 apart.
    assert curve_adjacency(TWO, SPACE, k=2).across_median == pytest.approx(2.0)


OFFSETS = curves({p: [off] * 6 for p, off in {1.0: 1.0, 2.0: 9.0, 3.0: 2.0, 4.0: 8.0}.items()})


def test_a_row_random_split_scores_better_than_a_grouped_one() -> None:
    rows = kfold_mae(OFFSETS, SPACE, "y", row_splitter(2, 0), None)
    groups = kfold_mae(OFFSETS, SPACE, "y", group_splitter(2, 0), OFFSETS["p"].to_numpy())
    assert rows < groups


def test_fewer_groups_than_folds_scores_nan() -> None:
    one_group = np.zeros(len(OFFSETS))
    assert np.isnan(kfold_mae(OFFSETS, SPACE, "y", group_splitter(2, 0), one_group))


def test_a_holdout_predicts_the_test_rows_from_the_nearest_other_curve() -> None:
    test = (OFFSETS["p"] == 4.0).to_numpy()  # nearest other curve: p = 3, y = 2 against y = 8
    assert holdout_mae(OFFSETS, SPACE, "y", test) == pytest.approx(6.0)


def test_a_holdout_of_every_row_scores_nan() -> None:
    assert np.isnan(holdout_mae(OFFSETS, SPACE, "y", np.ones(len(OFFSETS), dtype=bool)))


def grid(missing: tuple[tuple[int, int], ...] = (), size: int = 3) -> pd.DataFrame:
    """One row per cell of a size x size (a, b) grid, without the missing cells."""
    cells = [
        (a, b) for a in range(1, size + 1) for b in range(1, size + 1) if (a, b) not in missing
    ]
    return pd.DataFrame(cells, columns=["a", "b"])


def test_only_the_centre_of_a_full_3x3_grid_is_off_the_rim() -> None:
    assert int((~rim_mask(grid(), "a", "b")).sum()) == 1


def test_the_boundary_of_an_empty_corner_is_on_the_rim() -> None:
    table = grid(missing=((4, 4), (3, 4)), size=4)
    on_rim = rim_mask(table, "a", "b")
    assert bool(on_rim[(table["a"] == 3) & (table["b"] == 3)].all())


def test_a_generated_table_is_set_in_the_small_font() -> None:
    assert "\\centering\n\\small\n" in booktabs("x", "A table.", "lr", "a & b", ["1 & 2"])


def test_macros_render_one_newcommand_per_number() -> None:
    assert render_macros({"nsEdaX": "1.5"}) == "\\newcommand{\\nsEdaX}{1.5}\n"


def test_a_moderate_number_prints_plainly() -> None:
    assert number_tex(0.4712) == "0.471"


def test_a_moderate_number_keeps_three_significant_digits() -> None:
    assert number_tex(0.47) == "0.470"


def test_zero_prints_as_zero() -> None:
    assert number_tex(0.0) == "0"


def test_a_tiny_number_prints_in_scientific_notation() -> None:
    assert number_tex(8.203e-8) == "8.20\\times 10^{-8}"


def test_grid_bins_give_one_bin_per_grid_value() -> None:
    edges = np.asarray(grid_bins(np.array([1.0, 1.0, 2.0, 3.0])))
    assert len(edges) - 1 == 3


def test_grid_bins_fall_back_to_forty_beyond_the_limit() -> None:
    assert grid_bins(np.arange(10.0), limit=5) == 40
