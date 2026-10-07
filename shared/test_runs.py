"""Facts about the run record, on hand-written values."""

import uuid
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from shared.ceilings import Spread
from shared.harness import Run, Timing
from shared.runs import (
    RunMetadata,
    RunRecord,
    entry_from_json,
    entry_from_run,
    entry_name,
    entry_to_json,
    from_json,
    run_name,
    run_stem,
    to_json,
)
from shared.scorecard import Scorecard
from shared.surrogate import Training

TRAINING = Training(
    width=8,
    depth=1,
    activation="relu",
    loss="mse",
    lr=1e-2,
    max_epochs=3,
    batch_size=16,
    patience=3,
    valid_fraction=0.25,
    seed=0,
)
RECORD = RunRecord(
    dataset="toy",
    target="mass",
    training=TRAINING,
    commit="abc1234",
    dirty=True,
    started=datetime(2026, 10, 6, 10, 56, 0, tzinfo=UTC),
    seconds=12.5,
    epochs=3,
    best_epoch=2,
    mare=0.0123,
    rmse=0.0456,
)
SPREAD = Spread(median=0.01, p95=0.02, max=0.03, mean=0.015)
RUN = Run(
    seed=1,
    test=Scorecard(zones={"test": SPREAD}, decades={-2: SPREAD}),
    folds=(SPREAD, SPREAD),
    timing=Timing(fit=2.0, predict_one=0.001, predict_batch=0.002),
    predictions=np.array([0.5, 1.5]),
)
META = RunMetadata(
    id="0190a000-0000-7000-8000-000000000001",
    batch="0190a000-0000-7000-8000-000000000000",
    dataset="toy",
    target="charge",
    candidate="mlp",
    settings={"width": 8, "lr": 0.01, "activation": "relu", "seed": None},
    commit="abc1234",
    dirty=False,
    started=datetime(2026, 10, 7, 13, 0, 0, tzinfo=UTC),
    epochs=3,
    best_epoch=2,
)


def test_ledger_entries_made_in_order_sort_by_name_in_order() -> None:
    first = entry_from_run(RUN, replace(META, id=str(uuid.uuid7())))
    second = entry_from_run(RUN, replace(META, id=str(uuid.uuid7())))
    assert sorted([entry_name(second), entry_name(first)]) == [
        entry_name(first),
        entry_name(second),
    ]


def test_a_ledger_entry_round_trips_through_json() -> None:
    entry = entry_from_run(RUN, META)
    assert entry_from_json(entry_to_json(entry)) == entry


def test_run_names_sort_by_start_time() -> None:
    later = replace(RECORD, started=datetime(2026, 10, 12, 9, 0, 0, tzinfo=UTC))
    assert sorted([run_name(later), run_name(RECORD)]) == [run_name(RECORD), run_name(later)]


def test_the_record_keeps_whether_the_code_was_uncommitted() -> None:
    assert from_json(to_json(RECORD)).dirty is True


def test_the_record_round_trips_through_json() -> None:
    assert from_json(to_json(RECORD)) == RECORD


def test_the_run_folder_is_known_before_the_fit_from_its_start() -> None:
    assert run_stem("toy", "mass", RECORD.started) == run_name(RECORD)


def test_the_run_name_is_dataset_target_and_utc_stamp() -> None:
    assert run_name(RECORD) == "toy-mass-20261006T105600Z"
