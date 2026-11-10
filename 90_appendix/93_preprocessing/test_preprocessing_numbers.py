"""Facts about the Section 3.3 numbers, on tiny synthetic split and tables."""

import json
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
    pytestmark = pytest.mark.view

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
    pytestmark = pytest.mark.view

    @pytest.fixture
    def found(self):
        return numbers((split_summary(SPLIT, NS), split_summary(SPLIT, BH)), seed=7)

    def test_names_the_test_curves_per_dataset(self, found) -> None:
        assert found["prepNsTestCurves"] == "2"

    def test_names_the_seed(self, found) -> None:
        assert found["prepSeed"] == "7"

    def test_numbers_name_the_training_curves(self, found) -> None:
        assert (found["prepNsTrainCurves"], found["prepBhTrainCurves"]) == ("3", "1")

    def test_numbers_name_the_decisions_layout(self) -> None:
        found = numbers((split_summary(SPLIT, NS),), seed=7, decisions="narrow")
        assert found["prepDecisionsLayout"] == "narrow"


class TestMain:
    pytestmark = pytest.mark.functional

    @pytest.fixture
    def state(self, tmp_path: Path) -> Path:
        split = tmp_path / "s.parquet"
        SPLIT.to_parquet(split)
        state = tmp_path / "state"
        (state / "91_neutron_stars").mkdir(parents=True)
        (state / "91_neutron_stars" / "kept.json").write_text("another section's state")
        main([str(split), str(state), "--seed", "7"])
        return state

    def test_stores_the_numbers_in_its_section_folder_alone(self, state) -> None:
        assert [p.name for p in (state / "93_preprocessing").iterdir()] == ["numbers.json"]

    def test_stores_the_numbers_by_macro_name(self, state) -> None:
        stored = json.loads((state / "93_preprocessing" / "numbers.json").read_text())
        assert stored["prepBhTestCurves"] == "1"

    def test_leaves_other_sections_alone(self, state) -> None:
        assert (state / "91_neutron_stars" / "kept.json").is_file()
