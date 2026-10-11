"""The toy inputs the three result screens' tests share (W-091): tiny NS- and BH-shaped tables and
one split for both, written into a test's folder.

Generic, hand-made rows, never the real data's shape or scale. Ten curves per dataset, eight rows
each; the NS keys lie on a parabola, so no three are collinear and the local RBF's linear part has
full rank on any three curves. Two curves of each dataset are test curves, four lie in each of two
folds.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

CURVES = range(1, 11)
TOY_LABELS = ["test", "test", *["fold0"] * 4, *["fold1"] * 4]


def _write_toy_inputs(folder: Path) -> tuple[Path, Path, Path]:
    """Write the toy NS and BH tables and their split into folder; return their paths."""
    ns_rows = [
        {"beta": float(i), "lambda": float(i * i), "rho_c": 10.0 ** (1 + j / 7)}
        | {"M": 1.0 + 0.1 * j + 0.01 * i, "D": 0.1 + 0.01 * j}
        for i in CURVES
        for j in range(8)
    ]
    bh_rows = [
        {"r_h": 4.0 + j, "beta": float(i), "M": 2.0 + 0.5 * j + 0.01 * i, "D": 0.3 - 0.02 * j}
        for i in CURVES
        for j in range(8)
    ]
    split = [
        {"dataset": "neutron_stars", "beta": float(i), "lambda": float(i * i)}
        | {"label": label, "ablation": False}
        for i, label in zip(CURVES, TOY_LABELS, strict=True)
    ] + [
        {"dataset": "black_holes", "beta": float(i), "label": label, "ablation": False}
        for i, label in zip(CURVES, TOY_LABELS, strict=True)
    ]
    ns, bh, split_file = folder / "ns.parquet", folder / "bh.parquet", folder / "split.parquet"
    pd.DataFrame(ns_rows).to_parquet(ns)
    pd.DataFrame(bh_rows).to_parquet(bh)
    pd.DataFrame(split).to_parquet(split_file)
    return ns, bh, split_file


@pytest.fixture(scope="session")
def toy_inputs() -> Callable[[Path], tuple[Path, Path, Path]]:
    """The writer of the toy tables: toy_inputs(folder) -> (ns, bh, split) parquet paths."""
    return _write_toy_inputs
