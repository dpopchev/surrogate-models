"""The algorithm of Section 4.1 (W-152): one fitted surrogate per map and target.

The tuning's winner of each pair (recorded as JSON under the state folder, no new search) is fitted
on every non-test row, scored on the test rows beside the dataset's ceiling (the audit's, also
read as JSON: an input of the pipeline, never part of it) and saved under <state>/41_algorithm/.
A section never imports another: what it needs from one it reads from the state folder.
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paperlib.candidates import Knobs
from paperlib.config import load_config
from paperlib.design import BLACK_HOLES, NEUTRON_STARS, SPECS, TARGETS, pairs
from paperlib.domain import Domain, make_black_hole_domain, make_neutron_star_domain
from paperlib.final import (
    FinalSurrogate,
    Winner,
    fit_winner,
    read_winners,
    save_domain,
    save_surrogate,
)
from paperlib.surrogate import Training, make_training

logger = logging.getLogger(__name__)

SECTION = "41_algorithm"
# The audit sections whose state folder holds each dataset's ceilings.
CEILING_FOLDERS = {NEUTRON_STARS.dataset: "91_neutron_stars", BLACK_HOLES.dataset: "92_black_holes"}

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """The prepared tables, the split and the state folder the inputs are read from."""

    ns: Path
    bh: Path
    split: Path
    state: Path


# --- pure functions ---------------------------------------------------------------------------


def surrogate_path(state: Path, dataset: str, target: str) -> Path:
    """Where the surrogate of a pair is saved."""
    return state / SECTION / f"{dataset}-{target}.pkl"


def domain_path(state: Path, dataset: str) -> Path:
    """Where the physical domain of a dataset is saved."""
    return state / SECTION / f"{dataset}-domain.json"


def ceiling_path(state: Path, dataset: str) -> Path:
    """The audit's stored ceilings of a dataset."""
    return state / CEILING_FOLDERS[dataset] / "reachable_figures.json"


def pair_line(surrogate: FinalSurrogate) -> str:
    """One pair as printed: its family and unit, its test figures beside its ceiling."""
    return (
        f"{surrogate.dataset} {surrogate.target}: {surrogate.family} {surrogate.unit}, "
        f"test p95 {surrogate.test:.2f} figures, ceiling {surrogate.ceiling:.1f}"
    )


def summary_json(surrogates: Sequence[FinalSurrogate]) -> str:
    """The pairs' family, unit, params, test figures and ceiling as plain JSON."""
    rows = [
        {
            "dataset": s.dataset,
            "target": s.target,
            "family": s.family,
            "unit": s.unit,
            "params": dict(s.params),
            "test": s.test,
            "ceiling": s.ceiling,
        }
        for s in surrogates
    ]
    return json.dumps(rows, indent=2) + "\n"


# --- adapters and the run ---------------------------------------------------------------------


def read_ceilings(state: Path) -> Mapping[str, Mapping[str, float]]:
    """The ceilings of each dataset by target; a missing file raises naming the path."""
    found: dict[str, Mapping[str, float]] = {}
    for dataset in CEILING_FOLDERS:
        path = ceiling_path(state, dataset)
        if not path.is_file():
            raise FileNotFoundError(f"no ceilings at {path}: run the audit first")
        found[dataset] = json.loads(path.read_text())
    return found


def run(
    sources: Sources,
    knobs: Knobs,
    training: Training,
    seed: int,
    save: Callable[[Path, FinalSurrogate], None],
    save_domain: Callable[[Path, Domain], None],
    write: Callable[[Path, str], None],
    say: Callable[[str], None],
) -> tuple[FinalSurrogate, ...]:
    """Fit, save and print the winner of every pair, save the physical domain of each map, then
    write the summary; every effect past reading the inputs is injected."""
    state = sources.state
    winners = {
        (w.dataset, w.target): w for w in read_winners(state / "99_precision" / "winners.json")
    }
    ceilings = read_ceilings(state)
    tables = {NEUTRON_STARS.dataset: pd.read_parquet(sources.ns)}
    tables[BLACK_HOLES.dataset] = pd.read_parquet(sources.bh)
    split = pd.read_parquet(sources.split)
    save_domain(
        domain_path(state, NEUTRON_STARS.dataset),
        make_neutron_star_domain(tables[NEUTRON_STARS.dataset]),
    )
    save_domain(
        domain_path(state, BLACK_HOLES.dataset), make_black_hole_domain(tables[BLACK_HOLES.dataset])
    )
    fitted: list[FinalSurrogate] = []
    for pair in pairs(tables, split, SPECS, TARGETS):
        winner: Winner = winners[(pair.dataset, pair.target)]
        ceiling = float(ceilings[pair.dataset][pair.target])
        surrogate = fit_winner(
            pair, winner, knobs, training, seed, training.valid_fraction, ceiling
        )
        save(surrogate_path(state, pair.dataset, pair.target), surrogate)
        say(pair_line(surrogate))
        fitted.append(surrogate)
    write(state / SECTION / "summary.json", summary_json(fitted))
    return tuple(fitted)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    """The inputs, the state folder and the knobs of the tuning the winners' params lie over."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ns", "bh", "split", "state"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--neighbours", type=int, required=True)
    parser.add_argument("--knots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
    """Wire the real effects: the paper's network settings, saving, writing and printing."""
    args = _arguments(argv)
    training = make_training(load_config().network)
    knobs = Knobs(args.k, args.neighbours, args.knots, per_curve=0)
    sources = Sources(args.ns, args.bh, args.split, args.state)
    run(sources, knobs, training, args.seed, save_surrogate, save_domain, _write_text, print)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    main(sys.argv[1:])
