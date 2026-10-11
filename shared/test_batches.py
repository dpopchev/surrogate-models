"""Facts about the run shell every runnable shares, on tmp_path folders and injected effects."""

import logging
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from shared.batches import (
    Batch,
    WallError,
    WriteCheckError,
    at_level,
    git_commit,
    git_dirty,
    make_batch,
    new_id,
    outcomes_path,
    progress_tail,
    read_ledger,
    utc_now,
    wall,
    walled,
    write_checked,
    write_entry,
)
from shared.ceilings import Spread
from shared.harness import Run, Timing
from shared.runs import RunMetadata
from shared.scorecard import Scorecard

ROWS = np.zeros((3, 2))


def slow(x_fit, y_fit, x_valid, y_valid, seed):
    """A toy fitter that takes a second."""
    time.sleep(1.0)
    return lambda x: np.zeros(len(x))


@pytest.mark.edge
def test_write_checked_writes_the_text(tmp_path: Path) -> None:
    write_checked(tmp_path / "a.txt", "text\n")
    assert (tmp_path / "a.txt").read_text() == "text\n"


@pytest.mark.edge
def test_write_checked_refuses_a_text_that_does_not_read_back(tmp_path: Path) -> None:
    with pytest.raises(WriteCheckError, match=r"a\.txt does not read back"):
        write_checked(tmp_path / "a.txt", "text\n", read=lambda path: "other\n")


@pytest.mark.model
def test_the_outcomes_of_a_batch_are_its_file_in_its_section() -> None:
    assert outcomes_path(Path("state"), "62_toy", "b1") == Path("state/62_toy/b1.json")


class TestMakeBatch:
    pytestmark = pytest.mark.model

    @pytest.fixture
    def batch(self) -> Batch:
        return make_batch(
            Path("state"),
            journal=lambda batch: Path("state") / "62_toy" / f"{batch}.journal",
            commit=lambda: "abc1234",
            dirty=lambda: True,
            now=lambda: datetime(2026, 10, 11, 9, 0, 0, tzinfo=UTC),
            new_id=lambda: "id-1",
        )

    def test_takes_its_id_from_the_injected_maker(self, batch: Batch) -> None:
        assert batch.id == "id-1"

    def test_keeps_its_ledger_entries_under_the_state_folder(self, batch: Batch) -> None:
        assert batch.ledger == Path("state/ledger")

    def test_names_its_journal_after_its_id(self, batch: Batch) -> None:
        assert batch.journal == Path("state/62_toy/id-1.journal")

    def test_records_the_injected_commit(self, batch: Batch) -> None:
        assert batch.commit == "abc1234"

    def test_records_whether_the_code_was_uncommitted(self, batch: Batch) -> None:
        assert batch.dirty is True

    def test_records_the_injected_start(self, batch: Batch) -> None:
        assert batch.started == datetime(2026, 10, 11, 9, 0, 0, tzinfo=UTC)


@pytest.mark.view
def test_the_progress_tail_tells_the_elapsed_and_the_time_left() -> None:
    assert progress_tail(done=2, total=6, elapsed=60.0) == "elapsed 1m 0s, about 2m 0s left"


@pytest.mark.model
def test_a_fit_past_its_wall_is_stopped() -> None:
    with pytest.raises(WallError):
        wall(slow, seconds=0.05)(ROWS, ROWS[:, 0], ROWS, ROWS[:, 0], 0)


@pytest.mark.model
def test_the_wall_names_its_budget_in_minutes() -> None:
    with pytest.raises(WallError, match=r"^past 0\.000833333 min$"):
        wall(slow, seconds=0.05)(ROWS, ROWS[:, 0], ROWS, ROWS[:, 0], 0)


@pytest.mark.model
def test_a_walled_fit_predicts_nan() -> None:
    predict = walled(slow, seconds=0.05)(ROWS, ROWS[:, 0], ROWS, ROWS[:, 0], 0)
    assert np.isnan(predict(ROWS)).all()


@pytest.mark.model
def test_a_walled_fitter_skips_its_later_fits() -> None:
    fit = walled(slow, seconds=0.05)
    fit(ROWS, ROWS[:, 0], ROWS, ROWS[:, 0], 0)
    started = time.monotonic()
    fit(ROWS, ROWS[:, 0], ROWS, ROWS[:, 0], 0)
    assert time.monotonic() - started < 0.05


@pytest.mark.edge
def test_at_level_restores_the_root_level_after_the_block() -> None:
    root = logging.getLogger()
    before = root.level
    with at_level("DEBUG"):
        pass
    assert root.level == before


@pytest.mark.edge
def test_at_level_sets_the_root_level_inside_the_block() -> None:
    with at_level("DEBUG"):
        inside = logging.getLogger().level
    assert inside == logging.DEBUG


@pytest.mark.model
def test_new_ids_sort_in_the_order_they_were_made() -> None:
    first, second = new_id(), new_id()
    assert sorted([second, first]) == [first, second]


@pytest.mark.edge
def test_utc_now_is_in_utc() -> None:
    assert utc_now().tzinfo == UTC


def repository(folder: Path) -> Path:
    """A git repository at folder holding one committed file."""
    run = ["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*run[:3], "init", "-q"], check=True)
    (folder / "a.txt").write_text("a\n")
    subprocess.run([*run, "add", "a.txt"], check=True)
    subprocess.run([*run, "commit", "-q", "-m", "a"], check=True)
    return folder


@pytest.mark.edge
def test_git_commit_names_the_head_of_the_repository(tmp_path: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(repository(tmp_path)), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert git_commit(tmp_path) == head


@pytest.mark.edge
def test_a_committed_tree_is_not_dirty(tmp_path: Path) -> None:
    assert git_dirty(repository(tmp_path)) is False


@pytest.mark.edge
def test_an_uncommitted_file_makes_the_tree_dirty(tmp_path: Path) -> None:
    (repository(tmp_path) / "b.txt").write_text("b\n")
    assert git_dirty(tmp_path) is True


SPREAD = Spread(median=0.01, p95=0.02, max=0.03, mean=0.015)
RUN = Run(
    seed=1,
    test=Scorecard(zones={"test": SPREAD}, decades={}),
    folds=(SPREAD,),
    timing=Timing(fit=2.0, predict_one=1e-3, predict_batch=2e-3),
    predictions=np.array([0.5]),
)
META = RunMetadata(
    id="0190b000-0000-7000-8000-000000000001",
    batch="0190a000-0000-7000-8000-000000000000",
    dataset="toy",
    target="mass",
    candidate="k-NN",
    settings={"k": 2},
    commit="abc1234",
    dirty=False,
    started=datetime(2026, 10, 11, 9, 0, 0, tzinfo=UTC),
    epochs=None,
    best_epoch=None,
)


@pytest.mark.edge
def test_a_written_ledger_entry_reads_back(tmp_path: Path) -> None:
    entry = write_entry(tmp_path, RUN, META)
    assert read_ledger(tmp_path) == (entry,)


@pytest.mark.edge
def test_the_ledger_is_read_in_entry_order(tmp_path: Path) -> None:
    later = write_entry(tmp_path, RUN, META)
    earlier = write_entry(tmp_path, RUN, replace(META, id="0190a000-0000-7000-8000-000000000001"))
    assert read_ledger(tmp_path) == (earlier, later)
