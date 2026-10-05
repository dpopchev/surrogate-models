"""The frozen, curve-grouped split of both prepared tables (W-010, decided in W-019).

Whole curves are assigned, never rows: a frozen test set of a fraction of the curves, chosen
with the seed, and the remaining curves to GroupKFold folds for model selection. Each curve also
carries an ablation flag -- the NS rim, the BH outer curves -- for the extrapolation test that is
reported separately and never shapes the split.

Run as `uv run python <this file> <ns.parquet> <bh.parquet> <split.parquet>` (mk/data.mk does).
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from shared.eda import rim_mask

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

Ablation = Literal["rim", "outer"]


@dataclass(frozen=True)
class SplitSpec:
    """How one dataset is split: its curve key columns and which curves its ablation flags."""

    dataset: str
    curve: tuple[str, ...]
    ablation: Ablation


NEUTRON_STARS = SplitSpec("neutron_stars", ("beta", "lambda"), "rim")
BLACK_HOLES = SplitSpec("black_holes", ("beta",), "outer")


# --- pure functions ---------------------------------------------------------------------------


def curves_of(table: pd.DataFrame, spec: SplitSpec) -> pd.DataFrame:
    """One row per curve of the table: its key columns, sorted."""
    curve = list(spec.curve)
    return table[curve].drop_duplicates().sort_values(curve).reset_index(drop=True)


def labels(curves: pd.DataFrame, seed: int, test_fraction: float, folds: int) -> pd.Series:
    """Label each curve "test" or "fold<k>": a seeded test set of round(test_fraction * curves)
    curves (at least one), the rest shuffled into the folds.

    Every curve is its own group, so the folds over the remaining curves are GroupKFold folds.
    """
    order = np.random.default_rng(seed).permutation(len(curves))
    tested = max(1, round(test_fraction * len(curves)))
    result = pd.Series("", index=curves.index, dtype="str")
    result.iloc[order[:tested]] = "test"
    rest = order[tested:]
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(splitter.split(rest)):
        result.iloc[rest[held]] = f"fold{fold}"
    return result


def ablation_flags(curves: pd.DataFrame, spec: SplitSpec) -> pd.Series:
    """Flag the curves the extrapolation ablation holds out."""
    match spec.ablation:
        case "rim":
            outer, inner = spec.curve
            return rim_mask(curves, outer, inner)
        case "outer":
            first = curves[spec.curve[0]]
            return first.isin([first.min(), first.max()])
        case _:
            assert_never(spec.ablation)


def split(
    table: pd.DataFrame, spec: SplitSpec, seed: int, test_fraction: float, folds: int
) -> pd.DataFrame:
    """One row per curve: dataset, key columns, label and ablation flag."""
    curves = curves_of(table, spec)
    return pd.concat(
        [
            pd.DataFrame({"dataset": spec.dataset}, index=curves.index),
            curves,
            pd.DataFrame(
                {
                    "label": labels(curves, seed, test_fraction, folds),
                    "ablation": ablation_flags(curves, spec),
                }
            ),
        ],
        axis=1,
    )


# --- shell ------------------------------------------------------------------------------------


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the table paths, the output path and the split knobs."""
    parser = argparse.ArgumentParser(prog="split_datasets.py")
    parser.add_argument("neutron_stars", type=Path)
    parser.add_argument("black_holes", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    """Split both prepared tables into one file of curve labels, then verify it."""
    args = _arguments(argv)
    for source in (args.neutron_stars, args.black_holes):
        assert source.is_file(), f"prepared table not found: {source}"
    parts = [
        split(pd.read_parquet(source), spec, args.seed, args.test_fraction, args.folds)
        for source, spec in ((args.neutron_stars, NEUTRON_STARS), (args.black_holes, BLACK_HOLES))
    ]
    for part, spec in zip(parts, (NEUTRON_STARS, BLACK_HOLES), strict=True):
        assert not part.duplicated(list(spec.curve)).any(), f"{spec.dataset}: a curve has 2 labels"
        logger.info(
            "%s: %d curves -- %s; %d flagged for the ablation",
            spec.dataset,
            len(part),
            part["label"].value_counts().sort_index().to_dict(),
            int(part["ablation"].sum()),
        )
    table = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    assert pd.read_parquet(args.out).equals(table), f"{args.out} does not round-trip"
    logger.info("done: %d curve labels -> %s", len(table), args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
