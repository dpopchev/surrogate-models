"""Facts about the run record, on hand-written values."""

from dataclasses import replace
from datetime import UTC, datetime

from shared.runs import RunRecord, from_json, run_name, run_stem, to_json
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
