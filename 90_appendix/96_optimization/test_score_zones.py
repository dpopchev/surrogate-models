"""Facts about the zones run, on the toy inputs of the root conftest.py and one hand-written
ledger entry."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from score_zones import main

from paperlib.batches import write_checked
from paperlib.ceilings import Spread
from paperlib.harness import Run, Timing
from paperlib.runs import RunMetadata, entry_from_run, entry_name, entry_to_json
from paperlib.scorecard import Scorecard
from paperlib.tables import Cell, Text, from_json

BATCH = "toy-batch"
ENTRY = "toy-entry"
# The relative error each of the 16 test rows of the toy NS mass is predicted with: 0.01 all
# over but the last row of the first test curve (0.1); the toy's last rows lie at the stable edge.
ERROR = 0.01
LAST_ROW_ERROR = 0.1
# The factor of the true mass the flipped row is predicted with: far below its neighbour.
FLIPPED = 0.5
KNOBS = ["--batch", BATCH, "--bh-edge", "0.005", "--ns-edge", "0.05"]
EXTRAPOLATION = "toy-extrapolation"
# The p95 figures of the toy refit's edge curves, 0.9 below its interior's 3.4: the margin holds.
EDGE_P95 = 2.5


def write_state(state: Path, ns: Path, flip: bool = False) -> None:
    """A state folder with one tuned NS mass outcome, its ledger entry and the audit ceilings.

    With flip, the prediction of the last row of the first test curve drops below the one before
    it, so that one step of the rising curve turns against the data.
    """
    table = pd.read_parquet(ns)
    truth = table["M"].to_numpy()[:16]
    errors = np.full(16, ERROR)
    errors[7] = LAST_ROW_ERROR
    predictions = truth * (1.0 + errors)
    if flip:
        predictions[7] = truth[7] * FLIPPED
    spread = Spread(median=ERROR, p95=ERROR, max=ERROR, mean=ERROR)
    run = Run(
        seed=0,
        test=Scorecard(zones={"test": spread}, decades={}),
        folds=(spread,),
        timing=Timing(fit=1.0, predict_one=1e-3, predict_batch=2e-3),
        predictions=predictions,
    )
    meta = RunMetadata(
        id=ENTRY,
        batch=BATCH,
        dataset="neutron_stars",
        target="mass",
        candidate="local RBF",
        settings={"unit": "curve-wise"},
        commit="abc1234",
        dirty=False,
        started=datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC),
        epochs=None,
    )
    entry = entry_from_run(run, meta)
    folders = ("ledger", "99_precision", "96_optimization", "91_neutron_stars", "92_black_holes")
    for folder in folders:
        (state / folder).mkdir(parents=True)
    write_checked(state / "ledger" / entry_name(entry), entry_to_json(entry))
    job = {
        "dataset": "neutron_stars",
        "target": "mass",
        "candidate": {"family": "local RBF", "unit": "curve-wise"},
    }
    tuned = {
        "kind": "tuned",
        "job": job,
        "params": {},
        "folds": 3.0,
        "test": 3.0,
        "trials": 1,
        "walls": 0,
        "entry": ENTRY,
        "verdict": "grid",
    }
    write_checked(state / "99_precision" / f"{BATCH}.json", json.dumps([tuned]))
    winner = {**job["candidate"], "dataset": "neutron_stars", "target": "mass", "params": {}}
    write_checked(state / "99_precision" / "winners.json", json.dumps([winner]))
    # The refit without the edge curves: its edge figures stand beside the refit's interior.
    measured = {
        **job,
        **job["candidate"],
        "params": {},
        "kind": "measured",
        "zones": {"edge": {"p95": EDGE_P95, "worst": 1.0}, "interior": {"p95": 3.4, "worst": 2.0}},
        "fit_seconds": 1.0,
        "predict_seconds": 1.0,
    }
    write_checked(
        state / "96_optimization" / f"{EXTRAPOLATION}-extrapolation.json", json.dumps([measured])
    )
    for folder, found in (("91_neutron_stars", 4.1), ("92_black_holes", 7.1)):
        write_checked(
            state / folder / "reachable_figures.json", json.dumps({"mass": found, "charge": 2.4})
        )


@pytest.mark.functional
def test_zones_run_prints_one_line_per_candidate(
    tmp_path: Path, toy_inputs: Callable[[Path], tuple[Path, Path, Path]]
) -> None:
    ns, bh, split = toy_inputs(tmp_path)
    state = tmp_path / "state"
    write_state(state, ns)
    said: list[str] = []
    main([str(ns), str(bh), str(split), str(state), *KNOBS], said.append)
    saved = json.loads((state / "96_optimization" / "zones.json").read_text())
    assert (said, saved[0]["zones"]["physical"]) == (
        [
            "neutron_stars mass local RBF curve-wise: interior p95 1.49 worst 1.00 slopes 0/14, "
            "edge n/a slopes n/a, physical p95 1.02 worst 1.00 slopes n/a, ceiling 4.10"
        ],
        {"p95": pytest.approx(1.02, abs=0.005), "worst": pytest.approx(1.0)},
    )


@pytest.mark.functional
def test_zones_run_prints_the_slope_violations_per_zone(
    tmp_path: Path, toy_inputs: Callable[[Path], tuple[Path, Path, Path]]
) -> None:
    ns, bh, split = toy_inputs(tmp_path)
    state = tmp_path / "state"
    write_state(state, ns, flip=True)
    said: list[str] = []
    main([str(ns), str(bh), str(split), str(state), *KNOBS], said.append)
    saved = json.loads((state / "96_optimization" / "zones.json").read_text())
    printed = "worst" in said[0] and "slopes 1/14, edge n/a slopes n/a" in said[0]
    assert (printed, saved[0]["slopes"]) == (
        True,
        {"interior": [1, 14], "edge": [0, 0], "physical": [0, 0]},
    )


def cell_text(cell: Cell) -> str:
    """The text a cell of the stored table prints."""
    return cell.tex if isinstance(cell, Text) else getattr(cell, "mantissa", "")


@pytest.mark.functional
def test_zone_table_holds_one_row_per_candidate_with_its_verdict(
    tmp_path: Path, toy_inputs: Callable[[Path], tuple[Path, Path, Path]]
) -> None:
    ns, bh, split = toy_inputs(tmp_path)
    state = tmp_path / "state"
    write_state(state, ns)
    main(
        [str(ns), str(bh), str(split), str(state), *KNOBS, "--extrapolation-batch", EXTRAPOLATION],
        lambda _: None,
    )
    stored = from_json((state / "96_optimization" / "tables" / "zones.json").read_text())
    assert [tuple(cell_text(c) for c in row) for row in stored.rows] == [
        (
            "NS $M$",
            "local RBF",
            "curve-wise",
            "1.49",
            "2.50",
            "1.00",
            "1.02",
            "0/14",
            "$\\nsEdaCeilMFigures$",
            "grid",
        )
    ]
