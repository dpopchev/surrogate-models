"""Facts about the Section 4.3 numbers, on tiny synthetic split and tables."""

from pathlib import Path

import pandas as pd
import pytest
from preprocessing_numbers import floor_share, main, numbers, split_summary

from shared.config import PaperConfig

# Five NS curves: two test curves, folds of size 2 and 1, one flagged for the ablation.
NS, BH = "neutron_stars", "black_holes"
SPLIT = pd.DataFrame(
    {
        "dataset": [NS] * 5 + [BH] * 2,
        "label": ["test", "test", "fold0", "fold0", "fold1", "test", "fold0"],
        "ablation": [True, False, False, False, False, False, True],
    }
)
CHARGES = pd.DataFrame({"D": [1e-7, 1e-3, 1e-2, 2e-6]})


class TestSplitSummary:
    @pytest.fixture
    def summary(self):
        return split_summary(SPLIT, NS)

    def test_counts_the_curves(self, summary) -> None:
        assert summary.curves == 5

    def test_counts_the_test_curves(self, summary) -> None:
        assert summary.test == 2

    def test_the_smallest_fold(self, summary) -> None:
        assert summary.fold_min == 1

    def test_the_largest_fold(self, summary) -> None:
        assert summary.fold_max == 2

    def test_counts_the_folds(self, summary) -> None:
        assert summary.folds == 2

    def test_counts_the_ablation_curves(self, summary) -> None:
        assert summary.ablation == 1


def test_floor_share_counts_rows_below_the_floor() -> None:
    assert floor_share(CHARGES, NS, eps=1e-5).below == 2


class TestNumbers:
    @pytest.fixture
    def found(self):
        splits = (split_summary(SPLIT, NS), split_summary(SPLIT, BH))
        floors = (floor_share(CHARGES, NS, 1e-5), floor_share(CHARGES, BH, 1e-5))
        return numbers(splits, floors, eps=1e-5, seed=7)

    def test_names_the_test_curves_per_dataset(self, found) -> None:
        assert found["prepNsTestCurves"] == "2"

    def test_gives_the_floor_in_scientific_notation(self, found) -> None:
        assert found["prepChargeFloor"] == "1.00\\times 10^{-5}"

    def test_gives_the_rows_below_the_floor_in_percent(self, found) -> None:
        assert found["prepNsFloorPercent"] == "50.0"

    def test_names_the_seed(self, found) -> None:
        assert found["prepSeed"] == "7"


def test_main_writes_the_numbers(tmp_path: Path) -> None:
    split, ns, bh = tmp_path / "s.parquet", tmp_path / "ns.parquet", tmp_path / "bh.parquet"
    SPLIT.to_parquet(split)
    CHARGES.to_parquet(ns)
    CHARGES.to_parquet(bh)
    out = tmp_path / "43_preprocessing_numbers.tex"
    main([str(split), str(ns), str(bh), str(out), "--seed", "7"], config=PaperConfig())
    assert "\\newcommand{\\prepBhTestCurves}{1}" in out.read_text()
