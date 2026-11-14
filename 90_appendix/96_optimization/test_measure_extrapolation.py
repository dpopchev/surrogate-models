"""Facts about the extrapolation run, on the toy inputs of the root conftest.py, a toy tuning
outcome file and a fitter that predicts the mean of what it was fitted on."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from measure_extrapolation import arguments, run

from paperlib.batches import write_checked
from paperlib.candidates import Candidate
from paperlib.design import Pair
from paperlib.harness import FitRows, Fitter, Predictor

TUNING_BATCH = "toy-tuning"
BATCH = "toy-extrapolation"


def mean_fitter(pair: Pair, candidate: Candidate, params: Mapping[str, Any]) -> Fitter:
    """The toy fitter: every row predicted as the mean of the rows fitted."""

    def fit(rows: FitRows, seed: int) -> Predictor:
        return lambda x: np.full(len(x), rows.y_fit.mean())

    return fit


def no_fit(pair: Pair, candidate: Candidate, params: Mapping[str, Any]) -> Fitter:
    """A fitter builder that fails the test when the run fits anything."""
    raise AssertionError("nothing is fitted when a saved batch is reprinted")


def write_state(state: Path, split: Path) -> None:
    """The toy split with curves 3 and 10 flagged as the edge, and a tuning batch of two tuned
    outcomes and one walled outcome."""
    table = pd.read_parquet(split)
    table["ablation"] = table["beta"].isin([3.0, 10.0])
    table.to_parquet(split)
    for folder in ("99_precision", "96_optimization"):
        (state / folder).mkdir(parents=True)

    def job(dataset: str, family: str) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "target": "mass",
            "candidate": {"family": family, "unit": "curve-wise"},
        }

    tuned = {"kind": "tuned", "params": {}, "folds": 3.0, "test": 3.0, "trials": 1, "walls": 0}
    outcomes = [
        {**tuned, "job": job("neutron_stars", "local RBF"), "entry": "a"},
        {**tuned, "job": job("black_holes", "k-NN"), "entry": "b"},
        {"kind": "walled", "job": job("black_holes", "GPR"), "reason": "all 1 trials past 30 min"},
    ]
    write_checked(state / "99_precision" / f"{TUNING_BATCH}.json", json.dumps(outcomes))


@pytest.mark.functional
def test_extrapolation_run_saves_one_outcome_per_candidate(
    tmp_path: Path, toy_inputs: Callable[[Path], tuple[Path, Path, Path]]
) -> None:
    ns, bh, split = toy_inputs(tmp_path)
    state = tmp_path / "state"
    write_state(state, split)
    base = [str(ns), str(bh), str(split), str(state)]
    knobs = ["--k", "3", "--neighbours", "10", "--knots", "4", "--minutes", "1", "--seed", "0"]
    fit = [*base, *knobs, "--workers", "1", "--tuning-batch", TUNING_BATCH]
    first: list[str] = []
    run(arguments(fit), first.append, mean_fitter, 0.34, lambda: BATCH)
    saved = json.loads((state / "96_optimization" / f"{BATCH}-extrapolation.json").read_text())
    again: list[str] = []
    run(arguments([*fit, "--batch", BATCH]), again.append, no_fit, 0.34, lambda: "unused")
    assert (
        [(s["kind"], s["dataset"], s["family"]) for s in saved],
        sorted(saved[0]["zones"]),
        len(first),
        again,
    ) == (
        [("measured", "neutron_stars", "local RBF"), ("measured", "black_holes", "k-NN")],
        ["edge", "interior"],
        2,
        first,
    )
