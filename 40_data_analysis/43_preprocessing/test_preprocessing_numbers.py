"""Facts about the Section 3.3 numbers, on tiny synthetic split and tables."""

from pathlib import Path

import pandas as pd
import pytest
from preprocessing_numbers import main, numbers, split_summary

# Five NS curves: two test curves, folds of size 2 and 1, one flagged for the ablation.
NS, BH = "neutron_stars", "black_holes"
SPLIT = pd.DataFrame(
    {
        "dataset": [NS] * 5 + [BH] * 2,
        "label": ["test", "test", "fold0", "fold0", "fold1", "test", "fold0"],
        "ablation": [True, False, False, False, False, False, True],
    }
)


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


class TestNumbers:
    @pytest.fixture
    def found(self):
        return numbers((split_summary(SPLIT, NS), split_summary(SPLIT, BH)), seed=7)

    def test_names_the_test_curves_per_dataset(self, found) -> None:
        assert found["prepNsTestCurves"] == "2"

    def test_names_the_seed(self, found) -> None:
        assert found["prepSeed"] == "7"


class TestMain:
    @pytest.fixture
    def assets(self, tmp_path: Path) -> Path:
        split = tmp_path / "s.parquet"
        SPLIT.to_parquet(split)
        assets = tmp_path / "assets"
        (assets / "43_preprocessing").mkdir(parents=True)
        (assets / "43_preprocessing" / "old.tex").write_text("from an earlier run")
        (assets / "41_neutron_stars").mkdir()
        (assets / "41_neutron_stars" / "kept.tex").write_text("another section's asset")
        main([str(split), str(assets), "--seed", "7"])
        return assets

    def test_writes_the_numbers_into_its_section_folder_alone(self, assets) -> None:
        assert [p.name for p in (assets / "43_preprocessing").iterdir()] == [
            "43_preprocessing_num.tex"
        ]

    def test_writes_the_numbers_as_macros(self, assets) -> None:
        numbers_tex = assets / "43_preprocessing" / "43_preprocessing_num.tex"
        assert "\\newcommand{\\prepBhTestCurves}{1}" in numbers_tex.read_text()

    def test_leaves_other_sections_alone(self, assets) -> None:
        assert (assets / "41_neutron_stars" / "kept.tex").is_file()
