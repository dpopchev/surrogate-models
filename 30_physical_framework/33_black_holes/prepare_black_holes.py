"""Prepare the zero-phi0 black-hole dataset: (r_h, beta) -> (M, D) with the fixed settings checked.

Run as `uv run python <this file> <in.dat> <out.parquet>` (mk/data.mk does).
"""

import io
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------


class FixedColumnVariesError(ValueError):
    """A column declared as a fixed setting holds more than one value."""


class NonPositiveTargetError(ValueError):
    """The target that must stay positive (it is log-transformed later) has rows <= 0."""


@dataclass(frozen=True)
class TableSpec:
    """Which raw columns to keep and their new names, which must be constant, which must be > 0."""

    rename: tuple[tuple[str, str], ...]
    fixed: tuple[str, ...]
    positive: str


def make_table_spec(rename: Mapping[str, str], fixed: tuple[str, ...], positive: str) -> TableSpec:
    """Build a spec; raise ValueError when the positive column is not among the kept names."""
    if positive not in rename.values():
        raise ValueError(f"positive column {positive!r} is not among the kept names")
    return TableSpec(tuple(rename.items()), fixed, positive)


# rh: horizon radius; kappa, lambda2, phi0 and choice_theory are fixed settings of this file.
BLACK_HOLES = make_table_spec(
    rename={"rh": "r_h", "beta": "beta", "M": "M", "D": "D"},
    fixed=("kappa", "lambda2", "phi0", "choice_theory"),
    positive="D",
)

# --- pure functions ---------------------------------------------------------------------------


def parse_table(text: str) -> pd.DataFrame:
    """Parse whitespace-separated rows under a '#'-prefixed header line of column names.

    The data may come in blocks: later '#' lines (repeated headers) and blank lines are skipped.
    """
    header, _, body = text.partition("\n")
    if not header.startswith("#"):
        raise ValueError(f"first line must be a '#' column header, got {header[:40]!r}")
    names = header.lstrip("#").split()
    return pd.read_csv(io.StringIO(body), sep=r"\s+", header=None, names=names, comment="#")


def check_fixed(table: pd.DataFrame, fixed: tuple[str, ...]) -> pd.DataFrame:
    """Return the table unchanged; raise FixedColumnVariesError naming a column with > 1 value."""
    for column in fixed:
        if (values := table[column].nunique()) != 1:
            raise FixedColumnVariesError(f"fixed column {column!r} holds {values} values")
    return table


def select(table: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Keep and rename the spec's columns; raise NonPositiveTargetError on rows <= 0."""
    kept = table[[raw for raw, _ in spec.rename]].rename(columns=dict(spec.rename))
    if (bad := int((kept[spec.positive] <= 0).sum())) > 0:
        raise NonPositiveTargetError(f"{spec.positive} <= 0 in {bad} row(s)")
    return kept.reset_index(drop=True)


def prepare(text: str, spec: TableSpec) -> pd.DataFrame:
    """Parse, check the fixed settings, then select the kept columns."""
    return select(check_fixed(parse_table(text), spec.fixed), spec)


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Read the raw file, write the prepared parquet, verify it round-trips."""
    assert len(argv) == 2, f"usage: prepare_black_holes.py <in.dat> <out.parquet>, got {argv}"
    source, out = Path(argv[0]), Path(argv[1])
    assert source.is_file(), f"input dataset not found: {source}"
    table = prepare(source.read_text(), BLACK_HOLES)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    assert pd.read_parquet(out).equals(table), f"{out} does not round-trip"
    logger.info("done: %d rows %s -> %s", len(table), list(table.columns), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
