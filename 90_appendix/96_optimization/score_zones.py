"""The zones of every tuned candidate (W-169): its recorded test predictions scored on the
interior curves, the edge curves and the physical edge, at the 95th percentile and at the worst
row, beside the ceiling of its pair. Nothing is fitted.

Per tuned outcome of a recorded tuning batch, the ledger entry of its refit holds the test
predictions; they are scored against the test rows of the pair's design. One line per pair and
candidate is printed and the same is saved as JSON under <state>/96_optimization/zones.json.
A section never imports another: the outcomes and the ceilings are read from the state folder.
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paperlib.batches import outcomes_path, read_ledger, write_checked
from paperlib.design import BLACK_HOLES, NEUTRON_STARS, SPECS, TARGETS, Design, Target, pairs
from paperlib.runs import LedgerEntry
from paperlib.scorecard import relative_errors, scorecard
from paperlib.zones import Side, ZoneFigures, curve_position, zone_figures, zone_masks

logger = logging.getLogger(__name__)

SECTION = "96_optimization"
TUNING_SECTION = "99_precision"
# The audit sections whose state folder holds each dataset's ceilings.
CEILING_FOLDERS = {NEUTRON_STARS.dataset: "91_neutron_stars", BLACK_HOLES.dataset: "92_black_holes"}
# Per dataset the coordinate its curves run along and the side of the physical edge: a black
# hole starts at the existence edge, a neutron star ends at its stable edge.
ALONG: Mapping[str, tuple[str, Side]] = {
    NEUTRON_STARS.dataset: ("rho_c", "end"),
    BLACK_HOLES.dataset: ("r_h", "start"),
}

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Zoned:
    """One tuned candidate on one pair: its figures per zone (None for an empty zone), the
    spread of its errors per decade of the true D (empty for the mass) and the pair's ceiling."""

    dataset: str
    target: str
    family: str
    unit: str
    entry: str
    zones: Mapping[str, ZoneFigures | None]
    decades: Mapping[int, Mapping[str, float]]
    ceiling: float


# --- pure functions ---------------------------------------------------------------------------


def figures_text(found: ZoneFigures | None) -> str:
    """A zone's figures as "p95 <x> worst <y>", or n/a for an empty zone."""
    if found is None:
        return "n/a"
    return f"p95 {found.p95:.2f} worst {found.worst:.2f}"


def line(zoned: Zoned) -> str:
    """The printed line of one pair and candidate."""
    zones = ", ".join(f"{name} {figures_text(found)}" for name, found in zoned.zones.items())
    return (
        f"{zoned.dataset} {zoned.target} {zoned.family} {zoned.unit}: {zones}, "
        f"ceiling {zoned.ceiling:.2f}"
    )


def score(
    entry: LedgerEntry,
    design: Design,
    along: np.ndarray,
    charge: np.ndarray,
    target: Target,
    side: Side,
    fraction: float,
) -> tuple[Mapping[str, ZoneFigures | None], Mapping[int, Mapping[str, float]]]:
    """The zone figures and the per-decade spread of the entry's stored test predictions.

    Raises ValueError naming the entry when its predictions do not match the test rows.
    """
    truth = design.y[design.test]
    if len(entry.predictions) != len(truth):
        raise ValueError(
            f"entry {entry.meta.id}: {len(entry.predictions)} predictions "
            f"for {len(truth)} test rows"
        )
    errors = relative_errors(truth, np.array(entry.predictions), target)
    position = curve_position(design.groups[design.test], along[design.test])
    masks = zone_masks(design.ablation[design.test], position, side, fraction)
    decades = scorecard(errors, {}, charge[design.test] if target == "charge" else None).decades
    return zone_figures(errors, masks), {d: asdict(s) for d, s in decades.items()}


def saved_text(found: Sequence[Zoned]) -> str:
    """The candidates as indented JSON."""
    return json.dumps([asdict(zoned) for zoned in found], indent=2) + "\n"


# --- shell: reads the state ---------------------------------------------------------------------


def read_tuned(path: Path) -> list[dict[str, Any]]:
    """The tuned outcomes of a saved batch; a walled job has no refit to score."""
    return [fields for fields in json.loads(path.read_text()) if fields["kind"] == "tuned"]


def read_ceilings(state: Path) -> Mapping[str, Mapping[str, float]]:
    """The ceilings of each dataset by target; a missing file raises naming the path."""
    found: dict[str, Mapping[str, float]] = {}
    for dataset, folder in CEILING_FOLDERS.items():
        path = state / folder / "reachable_figures.json"
        if not path.is_file():
            raise FileNotFoundError(f"no ceilings at {path}: run the audit first")
        found[dataset] = json.loads(path.read_text())
    return found


def run(args: argparse.Namespace, say: Callable[[str], None]) -> tuple[Zoned, ...]:
    """Score every tuned outcome of the batch from its ledger entry, print and save the lines."""
    tables = {
        NEUTRON_STARS.dataset: pd.read_parquet(args.ns),
        BLACK_HOLES.dataset: pd.read_parquet(args.bh),
    }
    split = pd.read_parquet(args.split)
    by_key = {(p.dataset, p.target): p for p in pairs(tables, split, SPECS, TARGETS)}
    entries = {e.meta.id: e for e in read_ledger(args.state / "ledger")}
    ceilings = read_ceilings(args.state)
    fractions = {NEUTRON_STARS.dataset: args.ns_edge, BLACK_HOLES.dataset: args.bh_edge}
    found: list[Zoned] = []
    for fields in read_tuned(outcomes_path(args.state, TUNING_SECTION, args.batch)):
        job = fields["job"]
        dataset, target = job["dataset"], job["target"]
        if fields["entry"] not in entries:
            raise ValueError(f"no ledger entry {fields['entry']} in {args.state / 'ledger'}")
        column, side = ALONG[dataset]
        table = tables[dataset]
        zones, decades = score(
            entries[fields["entry"]],
            by_key[dataset, target].design,
            table[column].to_numpy(),
            table["D"].to_numpy(),
            target,
            side,
            fractions[dataset],
        )
        candidate = job["candidate"]
        found.append(
            Zoned(
                dataset,
                target,
                candidate["family"],
                candidate["unit"],
                fields["entry"],
                zones,
                decades,
                ceilings[dataset][target],
            )
        )
    for zoned in found:
        say(line(zoned))
    (args.state / SECTION).mkdir(parents=True, exist_ok=True)
    write_checked(args.state / SECTION / "zones.json", saved_text(found))
    return tuple(found)


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    """The inputs, the state folder, the tuning batch and the physical-edge fractions."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ns", "bh", "split", "state"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--batch", required=True, help="the recorded tuning batch to score")
    parser.add_argument("--bh-edge", type=float, required=True, help="the black holes' fraction")
    parser.add_argument("--ns-edge", type=float, required=True, help="the neutron stars' fraction")
    return parser.parse_args(argv)


def main(argv: Sequence[str], say: Callable[[str], None] = print) -> None:
    """Score the batch and print one line per pair and candidate."""
    run(_arguments(argv), say)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    main(sys.argv[1:])
