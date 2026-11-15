"""The zones of every tuned candidate (W-169): its recorded test predictions scored on the
interior curves, the edge curves and the physical edge, at the 95th percentile and at the worst
row, and the steps along its curves whose slope turns against the data, beside the ceiling of
its pair. Nothing is fitted.

Per tuned outcome of a recorded tuning batch, the ledger entry of its refit holds the test
predictions; they are scored against the test rows of the pair's design. One line per pair and
candidate is printed and the same is saved as JSON under <state>/96_optimization/zones.json.
A section never imports another: the outcomes and the ceilings are read from the state folder.

With --extrapolation-batch the zone table (label tab:zones) and its macros are stored as JSON
under <state>/96_optimization/ (tables/zones.json, numbers.json) for render_assets.py: the
interior and physical-edge figures are the zones run's, the edge figures the batch's refit
without the edge curves, whose interior only decides the margin.
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
from paperlib.design import (
    BLACK_HOLES,
    DATASET_TEX,
    NEUTRON_STARS,
    SPECS,
    TARGETS,
    Design,
    Target,
    pair_tex,
    pairs,
)
from paperlib.runs import LedgerEntry
from paperlib.scorecard import relative_errors, scorecard
from paperlib.tables import Cell, Column, Plain, Table, Text, make_table
from paperlib.tools.render_assets import store_section
from paperlib.zones import (
    Side,
    SlopeSteps,
    ZoneFigures,
    curve_position,
    margin_steps,
    slope_violations,
    zone_figures,
    zone_masks,
)

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
# The data ceilings of the relative-error section in significant figures, as their macros.
CEILING_TEX: Mapping[tuple[str, str], str] = {
    ("neutron_stars", "mass"): "\\nsEdaCeilMFigures",
    ("neutron_stars", "charge"): "\\nsEdaCeilDFigures",
    ("black_holes", "mass"): "\\bhEdaCeilMFigures",
    ("black_holes", "charge"): "\\bhEdaCeilDFigures",
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
    slopes: Mapping[str, SlopeSteps]
    decades: Mapping[int, Mapping[str, float]]
    ceiling: float


# --- pure functions ---------------------------------------------------------------------------


def figures_text(found: ZoneFigures | None) -> str:
    """A zone's figures as "p95 <x> worst <y>", or n/a for an empty zone."""
    if found is None:
        return "n/a"
    return f"p95 {found.p95:.2f} worst {found.worst:.2f}"


def slopes_text(reversed_steps: int, steps: int) -> str:
    """A zone's slope count as "<reversed>/<steps>", or n/a for a zone without a step."""
    return f"{reversed_steps}/{steps}" if steps else "n/a"


def zone_slopes_text(found: SlopeSteps) -> str:
    """A zone's slope count with its flat steps: "<reversed>/<steps> flat <flat>"."""
    if not found.steps:
        return "n/a"
    return f"{slopes_text(found.reversed, found.steps)} flat {found.flat}"


def line(zoned: Zoned) -> str:
    """The printed line of one pair and candidate."""
    zones = ", ".join(
        f"{name} {figures_text(found)} slopes {zone_slopes_text(zoned.slopes[name])}"
        for name, found in zoned.zones.items()
    )
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
) -> tuple[
    Mapping[str, ZoneFigures | None],
    Mapping[str, SlopeSteps],
    Mapping[int, Mapping[str, float]],
]:
    """The zone figures, the slope violations per zone and the per-decade spread of the entry's
    stored test predictions.

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
    slopes = slope_violations(
        truth,
        np.array(entry.predictions),
        design.groups[design.test],
        along[design.test],
        masks,
    )
    return zone_figures(errors, masks), slopes, {d: asdict(s) for d, s in decades.items()}


def saved_text(found: Sequence[Zoned]) -> str:
    """The candidates as indented JSON."""
    return json.dumps([asdict(zoned) for zoned in found], indent=2) + "\n"


# A candidate's identity on a pair, the key its refit and its verdict are found by.
Key = tuple[str, str, str, str]
# The zone figures of a candidate's refit without the edge curves; None when its fit was walled.
Refit = Mapping[str, ZoneFigures | None] | None

TABLE_CAPTION = (
    "Every tuned candidate of \\cref{tab:tuning} per zone: the significant figures kept at the "
    "95th percentile of the error ($-\\log_{10}$) on the interior test curves and at the "
    "physical edge, from the refit of the tuning; on the edge curves from a refit made without "
    "them, at the 95th percentile and at the worst row; the steps of all test "
    "curves, each counted once, whose predicted slope has the other sign than the data "
    "(reversed over steps) and, in the next column, the flat steps, whose predicted difference "
    "is zero where the data move, counted apart from a reversal; the pair's data ceiling; and "
    "the verdict of the tuning. A wall: the refit ran past the time budget."
)
HEADS = ("Pair", "Family", "Unit", "Interior", "Edge", "Edge worst", "Physical", "Slopes", "Flat")


def _figure(found: ZoneFigures | None, worst: bool = False) -> Cell:
    """A zone's figures in math mode, n/a for an empty zone."""
    if found is None:
        return Text("n/a")
    return Plain(f"{found.worst if worst else found.p95:.2f}", math=True)


def _edge_cells(refit: Refit) -> tuple[Cell, Cell]:
    """The edge p95 and worst figures of a refit, wall when it was walled."""
    if refit is None:
        return Text("wall"), Text("wall")
    return _figure(refit["edge"]), _figure(refit["edge"], worst=True)


def zone_table(
    found: Sequence[Zoned], refits: Mapping[Key, Refit], verdicts: Mapping[Key, str]
) -> Table:
    """The zone table: one row per pair and tuned candidate, a candidate without a refit
    showing -- in its edge columns."""
    rows: list[tuple[Cell, ...]] = []
    for zoned in found:
        key = (zoned.dataset, zoned.target, zoned.family, zoned.unit)
        edge = (Text("--"), Text("--")) if key not in refits else _edge_cells(refits[key])
        # The physical zone overlaps the interior and edge zones, which partition the test rows.
        partition = [zoned.slopes[name] for name in ("interior", "edge")]
        reversed_steps = sum(p.reversed for p in partition)
        flat = sum(p.flat for p in partition)
        steps = sum(p.steps for p in partition)
        rows.append(
            (
                Text(pair_tex(zoned.dataset, zoned.target)),
                Text(zoned.family),
                Text(zoned.unit),
                _figure(zoned.zones["interior"]),
                *edge,
                _figure(zoned.zones["physical"]),
                Text(slopes_text(reversed_steps, steps)),
                Text(str(flat) if steps else "n/a"),
                Text(f"${CEILING_TEX[zoned.dataset, zoned.target]}$"),
                Text(verdicts[key]),
            )
        )
    columns = (
        Column("l", before="@{}"),
        Column("l"),
        Column("l"),
        *(Column("r") for _ in range(7)),
        Column("l", after="@{}"),
    )
    header = (tuple(Text(h) for h in (*HEADS, "Ceiling", "Verdict")),)
    return make_table("zones", TABLE_CAPTION, columns, header, tuple(rows), "@{}lll*{7}{r}l@{}")


def zone_numbers(
    found: Sequence[Zoned], refits: Mapping[Key, Refit], winners: Sequence[Key]
) -> dict[str, str]:
    """Per pair the three zone figures of its winner (zone<Pair>Interior, Edge, Physical) and
    whether its margin holds (zone<Pair>Margin: holds, or fails).

    Raises ValueError naming the winner when no tuned candidate of the zones run is it.
    """
    zoned_by = {(z.dataset, z.target, z.family, z.unit): z for z in found}
    macros: dict[str, str] = {}
    for key in winners:
        if key not in zoned_by:
            raise ValueError(f"winner {key} is not a tuned candidate of the zones run")
        zoned, refit = zoned_by[key], refits.get(key)
        dataset, target = key[:2]
        name = f"zone{DATASET_TEX[dataset].capitalize()}{target.capitalize()}"
        edge = None if refit is None else refit["edge"]
        refit_interior = None if refit is None else refit["interior"]
        held = (
            edge is not None
            and refit_interior is not None
            and margin_steps(edge.p95, edge.worst, refit_interior.p95) > 0
        )
        macros |= {
            f"{name}Interior": figures_value(zoned.zones["interior"]),
            f"{name}Edge": figures_value(edge),
            f"{name}Physical": figures_value(zoned.zones["physical"]),
            f"{name}Margin": "holds" if held else "fails",
        }
    return macros


def figures_value(found: ZoneFigures | None) -> str:
    """A zone's p95 figures as the text a macro holds, n/a for an empty zone."""
    return "n/a" if found is None else f"{found.p95:.2f}"


# --- shell: reads the state ---------------------------------------------------------------------


def read_refits(path: Path) -> dict[Key, Refit]:
    """The outcomes of a saved extrapolation batch by candidate; a walled one is None."""
    found: dict[Key, Refit] = {}
    for fields in json.loads(path.read_text()):
        key = (fields["dataset"], fields["target"], fields["family"], fields["unit"])
        if fields["kind"] == "walled":
            found[key] = None
            continue
        found[key] = {
            name: None if zone is None else ZoneFigures(**zone)
            for name, zone in fields["zones"].items()
        }
    return found


def read_winners(path: Path) -> list[Key]:
    """The winners of a recorded tuning, per pair its candidate."""
    return [
        (w["dataset"], w["target"], w["family"], w["unit"]) for w in json.loads(path.read_text())
    ]


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
    verdicts: dict[Key, str] = {}
    for fields in read_tuned(outcomes_path(args.state, TUNING_SECTION, args.batch)):
        job = fields["job"]
        dataset, target = job["dataset"], job["target"]
        if fields["entry"] not in entries:
            raise ValueError(f"no ledger entry {fields['entry']} in {args.state / 'ledger'}")
        column, side = ALONG[dataset]
        table = tables[dataset]
        zones, slopes, decades = score(
            entries[fields["entry"]],
            by_key[dataset, target].design,
            table[column].to_numpy(),
            table["D"].to_numpy(),
            target,
            side,
            fractions[dataset],
        )
        candidate = job["candidate"]
        # A batch recorded before the verdicts (W-171) holds none: its searches ran whole.
        verdicts[dataset, target, candidate["family"], candidate["unit"]] = fields.get(
            "verdict", "searched"
        )
        found.append(
            Zoned(
                dataset,
                target,
                candidate["family"],
                candidate["unit"],
                fields["entry"],
                zones,
                slopes,
                decades,
                ceilings[dataset][target],
            )
        )
    for zoned in found:
        say(line(zoned))
    (args.state / SECTION).mkdir(parents=True, exist_ok=True)
    write_checked(args.state / SECTION / "zones.json", saved_text(found))
    if args.extrapolation_batch:
        refits = read_refits(
            args.state / SECTION / f"{args.extrapolation_batch}-extrapolation.json"
        )
        winners = read_winners(args.state / TUNING_SECTION / "winners.json")
        store_section(
            args.state,
            SECTION,
            {"zones": zone_table(found, refits, verdicts)},
            zone_numbers(found, refits, winners),
        )
    return tuple(found)


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    """The inputs, the state folder, the tuning batch and the physical-edge fractions."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ns", "bh", "split", "state"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--batch", required=True, help="the recorded tuning batch to score")
    parser.add_argument(
        "--extrapolation-batch",
        default="",
        help="the recorded extrapolation batch whose edge figures join the stored zone table",
    )
    parser.add_argument("--bh-edge", type=float, required=True, help="the black holes' fraction")
    parser.add_argument("--ns-edge", type=float, required=True, help="the neutron stars' fraction")
    return parser.parse_args(argv)


def main(argv: Sequence[str], say: Callable[[str], None] = print) -> None:
    """Score the batch and print one line per pair and candidate."""
    run(_arguments(argv), say)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    main(sys.argv[1:])
