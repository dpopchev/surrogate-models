"""Facts about the frozen curve-grouped split, on tiny synthetic tables."""

from pathlib import Path

import pandas as pd
import pytest
from split_datasets import (
    BLACK_HOLES,
    NEUTRON_STARS,
    ablation_flags,
    curves_of,
    labels,
    main,
    split,
)


def ns_table(size: int = 3) -> pd.DataFrame:
    """Two rows per (beta, lambda) cell of a size x size grid."""
    rows = [
        {"beta": float(b), "lambda": float(lam), "rho_c": rho}
        for b in range(1, size + 1)
        for lam in range(1, size + 1)
        for rho in (1.0, 2.0)
    ]
    return pd.DataFrame(rows)


def bh_table(curves: int = 3) -> pd.DataFrame:
    """Two rows per beta curve."""
    rows = [{"beta": float(b), "r_h": r} for b in range(1, curves + 1) for r in (4.0, 5.0)]
    return pd.DataFrame(rows)


def keys(count: int) -> pd.DataFrame:
    """count single-key curves."""
    return pd.DataFrame({"beta": [float(b) for b in range(count)]})


def test_curves_of_gives_one_row_per_curve() -> None:
    assert len(curves_of(ns_table(), NEUTRON_STARS)) == 9


def test_the_test_set_takes_the_rounded_fraction_of_curves() -> None:
    assert int((labels(keys(20), seed=0, test_fraction=0.15, folds=5) == "test").sum()) == 3


def test_the_test_set_holds_at_least_one_curve() -> None:
    assert int((labels(keys(3), seed=0, test_fraction=0.15, folds=2) == "test").sum()) == 1


def test_the_other_curves_fill_every_fold() -> None:
    found = set(labels(keys(20), seed=0, test_fraction=0.15, folds=5))
    assert found == {"test", "fold0", "fold1", "fold2", "fold3", "fold4"}


def test_the_same_seed_gives_the_same_labels() -> None:
    first = labels(keys(20), seed=7, test_fraction=0.15, folds=5)
    assert first.equals(labels(keys(20), seed=7, test_fraction=0.15, folds=5))


def test_only_the_centre_of_a_full_ns_grid_is_off_the_rim() -> None:
    flags = ablation_flags(curves_of(ns_table(), NEUTRON_STARS), NEUTRON_STARS)
    assert int((~flags).sum()) == 1


def test_the_bh_ablation_flags_the_outer_curves() -> None:
    flags = ablation_flags(curves_of(bh_table(), BLACK_HOLES), BLACK_HOLES)
    assert flags.tolist() == [True, False, True]


def test_main_writes_both_datasets_to_one_file(tmp_path: Path) -> None:
    ns, bh, out = tmp_path / "ns.parquet", tmp_path / "bh.parquet", tmp_path / "split.parquet"
    ns_table().to_parquet(ns)
    bh_table().to_parquet(bh)
    main([str(ns), str(bh), str(out), "--folds", "2"])
    assert set(pd.read_parquet(out)["dataset"]) == {"neutron_stars", "black_holes"}


class TestSplit:
    @pytest.fixture
    def result(self):
        return split(ns_table(), NEUTRON_STARS, seed=0, test_fraction=0.15, folds=2)

    def test_one_row_per_curve(self, result) -> None:
        assert len(result) == 9

    def test_the_columns(self, result) -> None:
        assert list(result.columns) == ["dataset", "beta", "lambda", "label", "ablation"]

    def test_the_dataset_is_named(self, result) -> None:
        assert set(result["dataset"]) == {"neutron_stars"}
