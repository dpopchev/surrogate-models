"""Facts about the design-matrix contract, on a tiny synthetic dataset."""

import math

import numpy as np
import pandas as pd
import pytest

from shared.design import DesignSpec, UnlabelledCurveError, design
from shared.eda import make_curve_space

# Two curves keyed by p, sampled along x; the split also carries another dataset's curve p = 1.
SPEC = DesignSpec("toy", make_curve_space({"x": "log10", "p": "raw"}, ("p",)))


@pytest.fixture
def table() -> pd.DataFrame:
    """Three rows on two curves; the second row's charge is tiny."""
    return pd.DataFrame(
        {
            "p": [1.0, 1.0, 2.0],
            "x": [10.0, 100.0, 10.0],
            "M": [2.0, 4.0, 1.0],
            "D": [0.2, 1e-9, 0.1],
        }
    )


@pytest.fixture
def split() -> pd.DataFrame:
    """Curve p = 1 in a fold, p = 2 in the test set and flagged; another dataset's p = 1 in test."""
    return pd.DataFrame(
        {
            "dataset": ["toy", "toy", "other"],
            "p": [1.0, 2.0, 1.0],
            "label": ["fold0", "test", "test"],
            "ablation": [False, True, False],
        }
    )


def test_mass_target_is_m(table: pd.DataFrame, split: pd.DataFrame) -> None:
    assert design(table, split, SPEC, "mass").y.tolist() == [2.0, 4.0, 1.0]


def test_inputs_follow_the_spec_order_and_scale(table: pd.DataFrame, split: pd.DataFrame) -> None:
    inputs = design(table, split, SPEC, "mass").X
    assert inputs.tolist() == [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]


def test_rows_of_this_dataset_s_test_curves_are_test(
    table: pd.DataFrame, split: pd.DataFrame
) -> None:
    assert design(table, split, SPEC, "mass").test.tolist() == [False, False, True]


def test_each_row_carries_its_fold_and_test_rows_minus_one(
    table: pd.DataFrame, split: pd.DataFrame
) -> None:
    assert design(table, split, SPEC, "mass").fold.tolist() == [0, 0, -1]


def test_rows_of_flagged_curves_are_ablation(table: pd.DataFrame, split: pd.DataFrame) -> None:
    assert design(table, split, SPEC, "mass").ablation.tolist() == [False, False, True]


def test_rows_of_one_curve_share_a_group(table: pd.DataFrame, split: pd.DataFrame) -> None:
    assert design(table, split, SPEC, "mass").groups.tolist() == [0, 0, 1]


def test_a_curve_missing_from_the_split_raises(table: pd.DataFrame, split: pd.DataFrame) -> None:
    unlabelled = split[~((split["dataset"] == "toy") & (split["p"] == 2.0))]
    with pytest.raises(UnlabelledCurveError):
        design(table, unlabelled, SPEC, "mass")


class TestFloat32:
    """skorch trains on float32 inputs and targets."""

    def test_inputs(self, table: pd.DataFrame, split: pd.DataFrame) -> None:
        assert design(table, split, SPEC, "mass").X.dtype == np.float32

    def test_target(self, table: pd.DataFrame, split: pd.DataFrame) -> None:
        assert design(table, split, SPEC, "charge").y.dtype == np.float32


class TestChargeTarget:
    """The charge target is log10(D/M) for every D > 0, with no floor (W-063)."""

    def test_is_log10_d_over_m(self, table: pd.DataFrame, split: pd.DataFrame) -> None:
        assert design(table, split, SPEC, "charge").y[0] == pytest.approx(-1.0)

    def test_a_tiny_charge_keeps_its_own_log10_d_over_m(
        self, table: pd.DataFrame, split: pd.DataFrame
    ) -> None:
        expected = math.log10(1e-9 / 4.0)
        assert design(table, split, SPEC, "charge").y[1] == pytest.approx(expected)
