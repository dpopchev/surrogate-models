"""Facts about listing the baseline runs, on hand-written records and tmp_path folders."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from list_runs import Finished, Listed, Unscored, main, read_runs, run_files, runs_table

from shared.runs import RunRecord, to_json
from shared.surrogate import Training

RECORD = RunRecord(
    dataset="toy",
    target="mass",
    training=Training(
        width=8,
        depth=1,
        activation="relu",
        loss="mse",
        lr=1e-2,
        max_epochs=50,
        batch_size=16,
        patience=5,
        valid_fraction=0.25,
        seed=0,
    ),
    commit="abc1234",
    dirty=False,
    started=datetime(2026, 10, 6, 10, 0, 0, tzinfo=UTC),
    seconds=246.0,
    epochs=37,
    best_epoch=17,
    mare=0.00662,
    rmse=0.0130,
)
EARLY = Finished("toy-mass-20261006T100000Z", RECORD)
LATE = Finished("toy-mass-20261006T120000Z", replace(RECORD, commit="def5678"))


def test_a_finished_run_shows_its_scores_epochs_time_and_commit() -> None:
    assert runs_table([EARLY]) == [
        "toy-mass-20261006T100000Z  MARE 6.62e-03  RMSE 1.30e-02  epochs 37 (best 17)  "
        "fit 4m 6s  abc1234"
    ]


def test_a_run_from_uncommitted_code_is_flagged() -> None:
    dirty = Finished(EARLY.name, replace(RECORD, dirty=True))
    assert runs_table([dirty])[0].endswith("abc1234 (uncommitted changes)")


def test_a_run_without_scores_is_listed_with_why() -> None:
    running = Unscored("toy-mass-20261006T130000Z", "no run.json yet -- running or failed")
    assert runs_table([running]) == [
        "toy-mass-20261006T130000Z  no scores: no run.json yet -- running or failed"
    ]


class TestReadRuns:
    @pytest.fixture
    def runs(self, tmp_path: Path) -> dict[str, Listed]:
        """A finished run, a run without run.json, one with an older run.json, and latest.log."""
        for name in ("a-finished", "b-running", "c-older"):
            (tmp_path / name).mkdir()
        (tmp_path / "a-finished" / "run.json").write_text(to_json(RECORD))
        older = {k: v for k, v in json.loads(to_json(RECORD)).items() if k != "dirty"}
        (tmp_path / "c-older" / "run.json").write_text(json.dumps(older))
        (tmp_path / "latest.log").symlink_to("a-finished/train.log")
        return {run.name: run for run in read_runs(tmp_path)}

    def test_reads_a_finished_run_s_record(self, runs: dict[str, Listed]) -> None:
        assert runs["a-finished"] == Finished("a-finished", RECORD)

    def test_lists_a_run_without_run_json_as_unscored(self, runs: dict[str, Listed]) -> None:
        assert runs["b-running"] == Unscored("b-running", "no run.json yet -- running or failed")

    def test_lists_an_older_run_json_as_unreadable(self, runs: dict[str, Listed]) -> None:
        assert runs["c-older"] == Unscored("c-older", "run.json in an older format")

    def test_skips_latest_log(self, runs: dict[str, Listed]) -> None:
        assert sorted(runs) == ["a-finished", "b-running", "c-older"]


def test_a_run_s_files_are_listed_with_their_sizes(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text("abc")
    (tmp_path / "loss_curve.png").write_bytes(bytes(2048))
    assert run_files(tmp_path) == ["  loss_curve.png  2 KB", "  run.json  3 B"]


class TestMain:
    @pytest.fixture
    def state(self, tmp_path: Path) -> Path:
        """A state dir holding one finished run."""
        run = tmp_path / "51_algorithms" / EARLY.name
        run.mkdir(parents=True)
        (run / "run.json").write_text(to_json(RECORD))
        return tmp_path

    def test_lists_the_runs(self, state: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main([str(state)])
        assert capsys.readouterr().out.startswith(EARLY.name)

    def test_shows_the_latest_run_s_files(
        self, state: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([str(state), "--run", "latest"])
        assert "  run.json" in capsys.readouterr().out.splitlines()[1]

    def test_stops_on_an_unknown_run_naming_it(self, state: Path) -> None:
        with pytest.raises(SystemExit, match="no run toy-typo"):
            main([str(state), "--run", "toy-typo"])

    def test_says_when_there_are_no_runs_yet(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([str(tmp_path)])
        assert "no baseline runs yet -- make baseline starts one" in capsys.readouterr().out


def test_the_newest_run_comes_first() -> None:
    assert runs_table([EARLY, LATE])[0].startswith(LATE.name)
