"""Prepare the neutron-star dataset: converged, stable (beta, lambda, rho_c) -> (M, D) curves.

Run as `uv run python <this file> <in.dat> <out.parquet>` (mk/data.mk does). The rules are
W-008's spike.decision: parse per block, drop empty blocks (non-converged runs), assert the
fixed settings, cut each curve after its first M maximum (keeping it), require D > 0.
"""

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------


class HeaderLayoutError(ValueError):
    """The non-empty blocks do not share one column header."""


class FixedColumnVariesError(ValueError):
    """A column or block parameter declared as a fixed setting holds more than one value."""


class CurveParamMismatchError(ValueError):
    """A block's header parameter differs from the column it names in the block's rows."""


class NonRisingOrderError(ValueError):
    """The column that orders a curve does not rise strictly within a block."""


class NonPositiveTargetError(ValueError):
    """The target that must stay positive (it is log-transformed later) has rows <= 0."""


@dataclass(frozen=True)
class Block:
    """One run: its '#' header parameters, its column names and its rows (none when empty)."""

    params: tuple[tuple[str, float], ...]
    names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CurveSpec:
    """Which columns to keep and rename, what must be constant, and how a curve is ordered and cut.

    curve_params pairs a header parameter with the column that must equal it; order rises
    strictly along a curve; each curve ends at its first maximum of peak; positive stays > 0.
    """

    rename: tuple[tuple[str, str], ...]
    fixed_params: tuple[str, ...]
    fixed_columns: tuple[str, ...]
    curve_params: tuple[tuple[str, str], ...]
    order: str
    peak: str
    positive: str


def make_curve_spec(
    rename: Mapping[str, str],
    fixed_params: tuple[str, ...],
    fixed_columns: tuple[str, ...],
    curve_params: Mapping[str, str],
    order: str,
    peak: str,
    positive: str,
) -> CurveSpec:
    """Build a spec; raise ValueError when positive is not a kept name."""
    if positive not in rename.values():
        raise ValueError(f"positive column {positive!r} is not among the kept names")
    return CurveSpec(
        tuple(rename.items()),
        fixed_params,
        fixed_columns,
        tuple(curve_params.items()),
        order,
        peak,
        positive,
    )


# The 16 varying columns, renamed to identifiers; inputs (beta, lambda, rho_c), targets (M, D),
# the rest kept for EDA (W-008 R3). The 11 constant columns and 4 header settings are asserted.
NEUTRON_STARS = make_curve_spec(
    rename={
        "beta": "beta",
        "lambda": "lambda",
        "rhoc": "rho_c",
        "M": "M",
        "D": "D",
        "Rs": "R",
        "M0": "M0",
        "rho_rest_mass": "rho_rest_mass",
        "Pc": "P_c",
        "phi0": "phi0",
        "Phi0": "Phi0",
        "MassdPhidr": "mass_dphi_dr",
        "M/R": "compactness",
        "lambda^2/M^2": "lambda2_over_M2",
        "bindingEn": "binding_energy",
        "rho-3P": "rho_minus_3P",
    },
    fixed_params=("x1", "x2", "Pc", "choice_theory"),
    fixed_columns=(
        "kappa",
        "mphi",
        "lambda_phi",
        "x1",
        "x2",
        "EPS_P",
        "EOS_Type",
        "Gamma_poly",
        "K_poly",
        "vPertEq[0]",
        "TheoryType",
    ),
    curve_params={"beta": "beta", "Lambda": "lambda"},
    order="rhoc",
    peak="M",
    positive="D",
)

# --- pure functions ---------------------------------------------------------------------------


def parse_params(line: str) -> dict[str, float]:
    """Read 'name = value' pairs from a '#' parameter line."""
    pairs = (pair.split("=") for pair in line.lstrip("#").split(","))
    return {name.strip(): float(value) for name, value in pairs}


def _block(param_line: str, lines: list[str]) -> Block:
    """Build one block from its '#' line and the lines under it (header, then rows)."""
    params = tuple(parse_params(param_line).items())
    if not lines:
        return Block(params, (), ())
    header, *rows = lines
    return Block(params, tuple(header.split()), tuple(tuple(map(float, r.split())) for r in rows))


def parse_blocks(text: str) -> tuple[Block, ...]:
    """Split the text into blocks, each opened by a '#' line; an empty block has no rows.

    Blank lines are skipped; text before the first '#' line raises ValueError.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("#"):
        raise ValueError("the text must open with a '#' parameter line")
    starts = [i for i, line in enumerate(lines) if line.startswith("#")] + [len(lines)]
    return tuple(_block(lines[a], lines[a + 1 : b]) for a, b in pairwise(starts))


def check_layout(blocks: list[Block]) -> list[Block]:
    """Return the blocks unchanged; raise HeaderLayoutError when their headers differ."""
    if (layouts := len({block.names for block in blocks})) != 1:
        raise HeaderLayoutError(f"{layouts} header layouts among the non-empty blocks")
    return blocks


def check_fixed_params(blocks: list[Block], fixed: tuple[str, ...]) -> list[Block]:
    """Return the blocks unchanged; raise FixedColumnVariesError naming a varying parameter."""
    for name in fixed:
        if (values := len({dict(block.params)[name] for block in blocks})) != 1:
            raise FixedColumnVariesError(f"fixed parameter {name!r} holds {values} values")
    return blocks


def check_fixed(table: pd.DataFrame, fixed: tuple[str, ...]) -> pd.DataFrame:
    """Return the table unchanged; raise FixedColumnVariesError naming a column with > 1 value."""
    for column in fixed:
        if (values := table[column].nunique()) != 1:
            raise FixedColumnVariesError(f"fixed column {column!r} holds {values} values")
    return table


def check_curve_params(
    curve: pd.DataFrame, params: Mapping[str, float], pairs: tuple[tuple[str, str], ...]
) -> pd.DataFrame:
    """Return the curve unchanged; raise CurveParamMismatchError when a column differs from
    the header parameter it is paired with."""
    for param, column in pairs:
        if not (curve[column] == params[param]).all():
            raise CurveParamMismatchError(f"column {column!r} differs from header {param!r}")
    return curve


def check_rising(curve: pd.DataFrame, order: str) -> pd.DataFrame:
    """Return the curve unchanged; raise NonRisingOrderError unless order rises strictly."""
    if not (curve[order].diff().dropna() > 0).all():
        raise NonRisingOrderError(f"column {order!r} does not rise strictly along a curve")
    return curve


def cut_at_peak(curve: pd.DataFrame, peak: str) -> pd.DataFrame:
    """Keep the rows up to and including the first maximum of peak."""
    return curve.iloc[: int(curve[peak].to_numpy().argmax()) + 1]


def _curve(block: Block, spec: CurveSpec) -> pd.DataFrame:
    """Turn one non-empty block into its checked curve, cut at its peak."""
    curve = pd.DataFrame(list(block.rows), columns=list(block.names))
    checked = check_curve_params(curve, dict(block.params), spec.curve_params)
    return cut_at_peak(check_rising(checked, spec.order), spec.peak)


def select(table: pd.DataFrame, spec: CurveSpec) -> pd.DataFrame:
    """Keep and rename the spec's columns; raise NonPositiveTargetError on rows <= 0."""
    kept = table[[raw for raw, _ in spec.rename]].rename(columns=dict(spec.rename))
    if (bad := int((kept[spec.positive] <= 0).sum())) > 0:
        raise NonPositiveTargetError(f"{spec.positive} <= 0 in {bad} row(s)")
    return kept.reset_index(drop=True)


def prepare_blocks(blocks: tuple[Block, ...], spec: CurveSpec) -> pd.DataFrame:
    """Drop empty blocks, check every fixed setting, cut each curve at its peak, select.

    Raises ValueError when no block has rows.
    """
    kept = [block for block in blocks if block.rows]
    if not kept:
        raise ValueError("no rows: every block is empty")
    checked = check_fixed_params(check_layout(kept), spec.fixed_params)
    table = pd.concat([_curve(block, spec) for block in checked], ignore_index=True)
    return select(check_fixed(table, spec.fixed_columns), spec)


def prepare(text: str, spec: CurveSpec) -> pd.DataFrame:
    """Parse the blocks, then prepare them."""
    return prepare_blocks(parse_blocks(text), spec)


# --- shell ------------------------------------------------------------------------------------


def main(argv: list[str], spec: CurveSpec = NEUTRON_STARS) -> None:
    """Read the raw file, write the prepared parquet, verify it round-trips."""
    assert len(argv) == 2, f"usage: prepare_neutron_stars.py <in.dat> <out.parquet>, got {argv}"
    source, out = Path(argv[0]), Path(argv[1])
    assert source.is_file(), f"input dataset not found: {source}"
    blocks = parse_blocks(source.read_text())
    table = prepare_blocks(blocks, spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    assert pd.read_parquet(out).equals(table), f"{out} does not round-trip"
    empty = sum(1 for block in blocks if not block.rows)
    logger.info(
        "read %d blocks (%d empty), %d rows; kept %d rows",
        len(blocks),
        empty,
        sum(len(block.rows) for block in blocks),
        len(table),
    )
    logger.info("done: %d rows %s -> %s", len(table), list(table.columns), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
