"""Throwaway diagnostic: verify which comment-line params are already constant columns.

Hypothesis under test: of the initial-data params in each batch's leading `#` comment
(x1, x2, beta, Lambda, Pc, choice_theory), only `Pc` is NOT already reproduced as a
constant column across the batch -- so only `pc_init` must be appended.

Read-only. Prints, per comment param, whether a matching column exists, whether it is
constant across the batch, and whether that constant equals the comment value.
"""

from __future__ import annotations

import re
from pathlib import Path

SAMPLE = Path("neutron-stars.dat.sample")

# comment token -> column name (case/alias mapping we would encode in the reader)
ALIASES = {
    "x1": "x1",
    "x2": "x2",
    "beta": "beta",
    "Lambda": "lambda",
    "Pc": "Pc",
    "choice_theory": "TheoryType",
}


def parse_comment(line: str) -> dict[str, float]:
    """`# x1 = 0.005, beta = 0.4, ...` -> {'x1': 0.005, 'beta': 0.4, ...}."""
    body = line.lstrip("#").strip()
    out: dict[str, float] = {}
    for pair in body.split(","):
        key, _, val = pair.partition("=")
        out[key.strip()] = float(val.strip())
    return out


def batches(path: Path):
    """Yield (comment_params, header_cols, data_rows) per `#`-delimited batch."""
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        assert lines[i].startswith("#"), lines[i]
        params = parse_comment(lines[i])
        header = lines[i + 1].split()
        rows = []
        i += 2
        while i < len(lines) and not lines[i].startswith("#"):
            rows.append([float(x) for x in lines[i].split()])
            i += 1
        yield params, header, rows


for bn, (params, header, rows) in enumerate(batches(SAMPLE)):
    print(f"=== batch {bn}: {params} ===")
    for token, value in params.items():
        col = ALIASES.get(token)
        if col is None or col not in header:
            print(f"  {token:14s} -> NO MATCHING COLUMN  (would append)")
            continue
        idx = header.index(col)
        colvals = {r[idx] for r in rows}
        constant = len(colvals) == 1
        first = rows[0][idx]
        eq_comment = abs(first - value) <= 1e-12 * max(1.0, abs(value))
        verdict = "CONSTANT==comment" if constant and eq_comment else (
            "VARIES (first==comment)" if eq_comment else "MISMATCH"
        )
        print(
            f"  {token:14s} -> col '{col}' constant={constant} "
            f"first={first:g} comment={value:g} :: {verdict}"
        )
