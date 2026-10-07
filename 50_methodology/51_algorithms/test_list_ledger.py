"""Facts about listing the run ledger, on hand-made entries and tmp_path folders."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from list_ledger import ledger_table, main, read_ledger

from shared.ceilings import Spread
from shared.harness import Run, Timing
from shared.runs import LedgerEntry, RunMetadata, entry_from_run, entry_name, entry_to_json
from shared.scorecard import Scorecard


def spread_at(p95: float) -> Spread:
    """A spread whose 95th percentile is p95."""
    return Spread(median=p95 / 2, p95=p95, max=2 * p95, mean=p95 / 2)


META = RunMetadata(
    id="0190a000-0000-7000-8000-000000000001",
    batch="0190a000-0000-7000-8000-000000000000",
    dataset="toy",
    target="mass",
    candidate="MLP",
    settings={"width": 8},
    commit="abc1234",
    dirty=False,
    started=datetime(2026, 10, 7, 13, 0, 0, tzinfo=UTC),
    epochs=3,
    best_epoch=2,
)
ENTRY = entry_from_run(
    Run(
        seed=1,
        test=Scorecard(zones={"test": spread_at(1e-2)}, decades={}),
        folds=(spread_at(1e-2), spread_at(1e-3)),
        timing=Timing(fit=2.0, predict_one=1e-3, predict_batch=2e-3),
        predictions=np.array([1.0]),
    ),
    META,
)


def later(entry: LedgerEntry, id: str, batch: str) -> LedgerEntry:
    """The entry under another id and batch."""
    return replace(entry, meta=replace(entry.meta, id=id, batch=batch))


def test_the_ledger_folder_is_read_back_entry_by_entry(tmp_path: Path) -> None:
    (tmp_path / entry_name(ENTRY)).write_text(entry_to_json(ENTRY))
    assert read_ledger(tmp_path) == [ENTRY]


def test_main_prints_the_ledger_of_the_state_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / entry_name(ENTRY)).write_text(entry_to_json(ENTRY))
    main([str(tmp_path)])
    assert capsys.readouterr().out.startswith("0190a000-0000  toy mass  MLP")


def test_main_says_when_the_ledger_has_no_entries_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main([str(tmp_path)])
    assert "no ledger entries yet -- make baseline writes them" in capsys.readouterr().out


def test_main_says_when_no_entry_is_of_the_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / entry_name(ENTRY)).write_text(entry_to_json(ENTRY))
    main([str(tmp_path), "--batch", "ffff"])
    assert "no ledger entries of batch ffff" in capsys.readouterr().out


def test_the_ledger_lists_the_newest_entry_first() -> None:
    newer = later(ENTRY, "0190b000-0000-7000-8000-000000000001", "0190b000-0000-7000-8000-0000")
    assert ledger_table([ENTRY, newer], batch=None)[0].startswith("0190b000-0000")


def test_the_ledger_keeps_only_the_batch_its_prefix_names() -> None:
    other = later(ENTRY, "0190b000-0000-7000-8000-000000000001", "0190b000-0000-7000-8000-0000")
    assert len(ledger_table([ENTRY, other], batch="0190b")) == 1


def test_a_ledger_line_gives_the_pair_candidate_seed_figures_and_seconds() -> None:
    assert ledger_table([ENTRY], batch=None) == [
        "0190a000-0000  toy mass  MLP  seed 1  figures folds 2.5 test 2.0  fit 2s  call 1.00e-03 s"
    ]
