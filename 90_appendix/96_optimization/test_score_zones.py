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

BATCH = "toy-batch"
ENTRY = "toy-entry"
# The relative error each of the 16 test rows of the toy NS mass is predicted with: 0.01 all
# over but the last row of the first test curve (0.1); the toy's last rows lie at the stable edge.
ERROR = 0.01
LAST_ROW_ERROR = 0.1


def write_state(state: Path, ns: Path) -> None:
    """A state folder with one tuned NS mass outcome, its ledger entry and the audit ceilings."""
    table = pd.read_parquet(ns)
    truth = table["M"].to_numpy()[:16]
    errors = np.full(16, ERROR)
    errors[7] = LAST_ROW_ERROR
    spread = Spread(median=ERROR, p95=ERROR, max=ERROR, mean=ERROR)
    run = Run(
        seed=0,
        test=Scorecard(zones={"test": spread}, decades={}),
        folds=(spread,),
        timing=Timing(fit=1.0, predict_one=1e-3, predict_batch=2e-3),
        predictions=truth * (1.0 + errors),
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
    for folder in ("ledger", "99_precision", "91_neutron_stars", "92_black_holes"):
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
    }
    write_checked(state / "99_precision" / f"{BATCH}.json", json.dumps([tuned]))
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
    knobs = ["--batch", BATCH, "--bh-edge", "0.005", "--ns-edge", "0.05"]
    main([str(ns), str(bh), str(split), str(state), *knobs], said.append)
    saved = json.loads((state / "96_optimization" / "zones.json").read_text())
    assert (said, saved[0]["zones"]["physical"]) == (
        [
            "neutron_stars mass local RBF curve-wise: interior p95 1.49 worst 1.00, "
            "edge n/a, physical p95 1.02 worst 1.00, ceiling 4.10"
        ],
        {"p95": pytest.approx(1.02, abs=0.005), "worst": pytest.approx(1.0)},
    )
