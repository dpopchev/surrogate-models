"""The numbers Section 4.3 cites about the split and the charge floor (W-020).

Reads the frozen split (local/state/split.parquet) and both prepared tables; writes \\prep...
LaTeX macros: per dataset the curves in the test set and in each GroupKFold fold, the curves the
extrapolation ablation flags, and the rows whose charge D lies below the floor eps that the
charge target log10(max(D, eps)/M) applies (W-019).

Run as `uv run python <this file> <split> <ns> <bh> <asset dir> --seed <n>` (mk/paper.mk does).
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import PaperConfig, load_config
from shared.eda import render_macros, sci_tex

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSummary:
    """How one dataset's curves are split."""

    dataset: str
    curves: int
    test: int
    fold_min: int
    fold_max: int
    folds: int
    ablation: int

    @property
    def test_percent(self) -> float:
        """The share of curves in the test set, in percent."""
        return 100 * self.test / self.curves


@dataclass(frozen=True)
class FloorShare:
    """How many of one dataset's rows have a charge below the floor."""

    dataset: str
    rows: int
    below: int
    orders_above: float

    @property
    def percent(self) -> float:
        """The share of rows below the floor, in percent."""
        return 100 * self.below / self.rows


# --- pure functions ---------------------------------------------------------------------------


def split_summary(split: pd.DataFrame, dataset: str) -> SplitSummary:
    """Summarize one dataset's rows of the split file."""
    rows = split[split["dataset"] == dataset]
    fold_sizes = rows.loc[rows["label"] != "test", "label"].value_counts()
    return SplitSummary(
        dataset,
        len(rows),
        int((rows["label"] == "test").sum()),
        int(fold_sizes.min()),
        int(fold_sizes.max()),
        len(fold_sizes),
        int(rows["ablation"].sum()),
    )


def floor_share(table: pd.DataFrame, dataset: str, eps: float) -> FloorShare:
    """Count the rows of a table with D < eps, and the orders of magnitude D spans above eps."""
    orders = float(np.log10(table["D"].max() / eps))
    return FloorShare(dataset, len(table), int((table["D"] < eps).sum()), orders)


# Macro tags of the datasets: \prepNs..., \prepBh...
TAGS = {"neutron_stars": "Ns", "black_holes": "Bh"}


def numbers(
    splits: tuple[SplitSummary, ...], floors: tuple[FloorShare, ...], eps: float, seed: int
) -> dict[str, str]:
    """Every number Section 4.3 cites, keyed by its \\prep macro name."""
    found = {"prepChargeFloor": sci_tex(eps), "prepSeed": f"{seed}"}
    for s in splits:
        tag = TAGS[s.dataset]
        found |= {
            f"prep{tag}Curves": f"{s.curves}",
            f"prep{tag}TestCurves": f"{s.test}",
            f"prep{tag}TestPercent": f"{s.test_percent:.1f}",
            f"prep{tag}FoldMin": f"{s.fold_min}",
            f"prep{tag}FoldMax": f"{s.fold_max}",
            f"prep{tag}Folds": f"{s.folds}",
            f"prep{tag}AblationCurves": f"{s.ablation}",
        }
    for f in floors:
        tag = TAGS[f.dataset]
        found |= {
            f"prep{tag}FloorRows": f"{f.below}",
            f"prep{tag}FloorPercent": f"{f.percent:.1f}",
            f"prep{tag}FloorOrders": f"{f.orders_above:.1f}",
        }
    return found


# --- shell ------------------------------------------------------------------------------------

# The section's asset folder, and the prefix that keeps its basenames unique in the flat build.
SECTION = "43_preprocessing"


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the input paths, the asset dir and the split seed."""
    parser = argparse.ArgumentParser(prog="preprocessing_numbers.py")
    for name in ("split", "neutron_stars", "black_holes", "assets"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str], config: PaperConfig | None = None) -> None:
    """Write the Section 4.3 numbers from the split file and both prepared tables into the
    section's folder of the asset dir, after emptying that folder of an earlier run."""
    args = _arguments(argv)
    for source in (args.split, args.neutron_stars, args.black_holes):
        assert source.is_file(), f"input not found: {source}"
    eps = (config or load_config()).data_analysis.charge_floor
    split = pd.read_parquet(args.split)
    datasets = (("neutron_stars", args.neutron_stars), ("black_holes", args.black_holes))
    splits = tuple(split_summary(split, name) for name, _ in datasets)
    floors = tuple(floor_share(pd.read_parquet(path), name, eps) for name, path in datasets)
    out = args.assets / SECTION
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    numbers_tex = out / f"{SECTION}_num.tex"
    numbers_tex.write_text(render_macros(numbers(splits, floors, eps, args.seed)))
    logger.info("done: Section 4.3 numbers -> %s", numbers_tex)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
