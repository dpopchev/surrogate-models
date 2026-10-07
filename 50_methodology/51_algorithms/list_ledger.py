"""The run ledger at a glance (W-077): make ledger lists every harness run, newest first.

Every <id>.json under <state dir>/ledger/ is one entry (fit_baseline.py writes one per
predictor, pair and seed as it ends); a line gives its batch, pair, candidate and seed, the
significant figures at the 95th percentile of the error on the frozen folds (their mean) and on
the test curves, and the seconds of the fit and of a one-row call.

Run as `uv run python <this file> <state dir> [--batch <batch id or its prefix>]`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from shared.runs import LedgerEntry, entry_from_json
from shared.scorecard import significant_figures
from shared.surrogate import duration_text

# --- pure functions ---------------------------------------------------------------------------


def ledger_table(entries: list[LedgerEntry], batch: str | None) -> list[str]:
    """One line per entry, newest first; only the batch whose id starts with batch, if given."""
    newest = sorted(entries, key=lambda entry: entry.meta.id, reverse=True)
    return [_line(e) for e in newest if batch is None or e.meta.batch.startswith(batch)]


def _line(entry: LedgerEntry) -> str:
    """One entry's line; the batch by its first 13 characters, a uuid7's millisecond stamp."""
    meta, timing = entry.meta, entry.timing
    folds = np.mean([significant_figures(fold.p95) for fold in entry.folds])
    test = significant_figures(entry.test.zones["test"].p95)
    return (
        f"{meta.batch[:13]}  {meta.dataset} {meta.target}  {meta.candidate}  seed {entry.seed}  "
        f"figures folds {folds:.1f} test {test:.1f}  fit {duration_text(timing.fit)}  "
        f"call {timing.predict_one:.2e} s"
    )


# --- shell ------------------------------------------------------------------------------------


def read_ledger(folder: Path) -> list[LedgerEntry]:
    """Every entry in the ledger folder."""
    return [entry_from_json(path.read_text()) for path in sorted(folder.glob("*.json"))]


def main(argv: list[str]) -> None:
    """Print the ledger table."""
    args = _arguments(argv)
    entries = read_ledger(args.state / "ledger")
    lines = ledger_table(entries, args.batch)
    if not entries:
        print(f"no ledger entries yet -- make baseline writes them (looked in {args.state})")
    elif not lines:
        print(f"no ledger entries of batch {args.batch} -- make ledger lists every batch")
    for line in lines:
        print(line)


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the state dir and the optional batch."""
    parser = argparse.ArgumentParser(prog="list_ledger.py")
    parser.add_argument("state", type=Path)
    parser.add_argument("--batch", help="a batch id, or its first characters")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
