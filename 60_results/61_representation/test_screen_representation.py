"""Facts about the representation screen behind Section 5.1, on tiny synthetic data."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pytest
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from screen_representation import (
    SECTION,
    Cell,
    Context,
    Job,
    Knobs,
    Score,
    Work,
    beats,
    cells,
    main,
    numbers,
    plan,
    representation_table,
    run_job,
    scores,
    screen_job,
)

from shared.batches import Batch
from shared.ceilings import Spread
from shared.config import PaperConfig
from shared.design import Design, DesignSpec
from shared.eda import make_curve_space
from shared.harness import Run, Timing
from shared.runs import LedgerEntry, RunMetadata, entry_from_run
from shared.scorecard import Scorecard
from shared.surrogate import Training
from shared.workers import run_jobs

# The writer of the toy NS and BH tables and their split, the fixture of 60_results/conftest.py.
ToyInputs = Callable[[Path], tuple[Path, Path, Path]]

# Ten curves keyed by p along x, all y = 1 + x^2; the test curve p = 2 is sampled between the
# grid points of the others, so only a whole-curve model rebuilds it exactly.
P = np.repeat(np.arange(10.0), 10)
X_ALONG = np.where(
    P == 2, np.tile(np.linspace(0.05, 0.95, 10), 10), np.tile(np.linspace(0, 1, 10), 10)
)
TOY = Design(
    X=np.column_stack([X_ALONG, P]).astype(np.float32),
    y=(1.0 + X_ALONG**2).astype(np.float32),
    groups=P.astype(int),
    test=P == 2,
    fold=np.where(P == 2, -1, P.astype(int) % 2),
    ablation=np.zeros(len(P), dtype=bool),
)
TOY_SPEC = DesignSpec("toy", make_curve_space({"x": "raw", "p": "raw"}, ("p",)))
# 20 neighbours span at least two toy curves: within one curve the RBF's linear part is singular.
KNOBS = Knobs(k=2, neighbours=20, knots=4)
SHORT = Training(
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


def spread_at(p95: float) -> Spread:
    """A spread whose 95th percentile is p95."""
    return Spread(median=p95 / 2, p95=p95, max=2 * p95, mean=p95 / 2)


def ledger_entry(
    representation: str, model: str, seed: int, test: float, folds: tuple[float, ...]
) -> LedgerEntry:
    """A toy mass entry of the screen with these test and fold p95 errors."""
    return entry_from_run(
        Run(
            seed=seed,
            test=Scorecard(zones={"test": spread_at(test)}, decades={}),
            folds=tuple(spread_at(p95) for p95 in folds),
            timing=Timing(fit=1.0, predict_one=1e-3, predict_batch=2e-3),
            predictions=np.array([1.0]),
        ),
        RunMetadata(
            id=f"0190a000-0000-7000-8000-00000000000{seed}",
            batch="0190a000-0000-7000-8000-000000000000",
            dataset="toy",
            target="mass",
            candidate=model,
            settings={"representation": representation, "model": model},
            commit="abc1234",
            dirty=False,
            started=datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC),
            epochs=None,
        ),
    )


def toy_job(representation: Any, unit: Any, model: Any) -> Job:
    """A job on the toy curves; the toy mass is positive, so it is scored as M."""
    return Job(Cell("toy", "mass", representation, TOY_SPEC, "mass", unit), model, 0)


@pytest.mark.model
def test_the_screen_plans_fourteen_representation_cells() -> None:
    assert len(cells()) == 14


@pytest.mark.model
def test_the_references_run_one_seed_and_the_pointwise_mlp_every_seed() -> None:
    assert len(plan(cells(), (1, 2, 3))) == 14 * 2 + 10 * 3


@pytest.mark.command
def test_a_job_records_its_representation_with_its_run() -> None:
    recorded: list[dict[str, Any]] = []

    def record(settings: dict[str, Any], run: Run) -> None:
        recorded.append(settings)

    run_job(toy_job("log10 M", "pointwise", "k-NN"), TOY, KNOBS, SHORT, record)
    assert recorded[0]["representation"] == "log10 M"


@pytest.mark.model
def test_a_curve_wise_job_rebuilds_a_curve_sampled_between_the_grid_points() -> None:
    run = run_job(toy_job("curve-wise", "curve-wise", "k-NN"), TOY, KNOBS, SHORT, lambda *_: None)
    assert run.test.zones["test"].p95 < 1e-5


@pytest.mark.command
def test_a_local_rbf_job_records_its_neighbours() -> None:
    recorded: list[dict[str, Any]] = []

    def record(settings: dict[str, Any], run: Run) -> None:
        recorded.append(settings)

    run_job(toy_job("baseline", "pointwise", "local RBF"), TOY, KNOBS, SHORT, record)
    assert recorded[0]["neighbours"] == KNOBS.neighbours


@pytest.mark.command
def test_an_mlp_job_hands_each_network_to_before_fit_on_every_fold_and_the_final_fit() -> None:
    built: list[Any] = []
    job = toy_job("baseline", "pointwise", "MLP")
    run_job(job, TOY, KNOBS, SHORT, lambda *_: None, before_fit=built.append)
    assert len(built) == 3


@pytest.mark.model
def test_an_mlp_job_on_a_curve_wise_cell_is_refused() -> None:
    job = toy_job("curve-wise", "curve-wise", "MLP")
    with pytest.raises(ValueError, match="curve-wise"):
        run_job(job, TOY, KNOBS, SHORT, lambda *_: None)


@pytest.mark.model
def test_the_seeds_of_one_model_on_one_representation_make_one_score() -> None:
    entries = [ledger_entry("baseline", "MLP", seed, 1e-2, (1e-2,)) for seed in (1, 2)]
    assert len(scores(entries)) == 1


@pytest.mark.model
def test_the_fold_figures_are_the_mean_over_every_fold_of_every_seed() -> None:
    entries = [
        ledger_entry("baseline", "MLP", 1, 1e-2, (1e-2, 1e-2)),
        ledger_entry("baseline", "MLP", 2, 1e-2, (1e-4, 1e-4)),
    ]
    assert scores(entries)[0].folds == pytest.approx(3.0)


@pytest.mark.model
def test_the_spread_is_the_standard_deviation_over_every_fold_of_every_seed() -> None:
    entries = [
        ledger_entry("baseline", "MLP", 1, 1e-2, (1e-2, 1e-2)),
        ledger_entry("baseline", "MLP", 2, 1e-2, (1e-4, 1e-4)),
    ]
    assert scores(entries)[0].spread == pytest.approx(1.0)


def score_at(folds: float, spread: float) -> Score:
    """A toy MLP score with these fold figures and spread."""
    return Score("toy", "mass", "log10 M", "MLP", test=folds, folds=folds, spread=spread)


@pytest.mark.model
def test_a_gain_beyond_both_spreads_beats_the_baseline() -> None:
    assert beats(score_at(3.0, 0.5), score_at(2.0, 0.5))


@pytest.mark.model
def test_a_gain_within_the_larger_spread_does_not_beat_the_baseline() -> None:
    assert not beats(score_at(3.0, 0.5), score_at(2.0, 1.5))


def ns_mass(representation: str, model: str, folds: float, spread: float) -> Score:
    """An NS mass score of the screen with these fold figures and spread."""
    return Score("neutron_stars", "mass", representation, model, folds, folds, spread)


def table_rows(table: str) -> list[str]:
    """The body rows of a booktabs table, between its midrule and bottomrule."""
    body = table.split("\\midrule\n")[1].split("\\bottomrule")[0]
    return [row for row in body.split("\\\\\n") if row.strip()]


@pytest.mark.view
def test_the_table_has_one_row_per_pair_and_representation() -> None:
    found = [ns_mass("baseline", "k-NN", 2.0, 0.1), ns_mass("baseline", "MLP", 2.5, 0.1)]
    assert len(table_rows(representation_table(found))) == 1


@pytest.mark.view
def test_a_fold_entry_that_beats_the_baseline_is_bold() -> None:
    found = [ns_mass("baseline", "MLP", 2.0, 0.1), ns_mass("log10 M", "MLP", 3.0, 0.1)]
    assert "\\mathbf{3.00_{\\pm 0.10}}" in table_rows(representation_table(found))[1]


@pytest.mark.view
def test_representation_numbers_name_each_score() -> None:
    found = [ns_mass("baseline", "MLP", 2.0, 0.1), ns_mass("raw rho_c", "MLP", 1.8, 0.1)]
    assert numbers(found) == {
        "reprNsMassBaselineMlpTest": "2.00",
        "reprNsMassBaselineMlpFolds": "2.00",
        "reprNsMassRawRhoMlpTest": "1.80",
        "reprNsMassRawRhoMlpFolds": "1.80",
        "reprNsMassRawRhoLossMlp": "0.20",
    }


@pytest.mark.view
def test_a_negative_test_figure_is_typeset_with_a_minus_sign() -> None:
    found = [ns_mass("baseline", "MLP", -1.04, 0.1)]
    assert "$-1.04$" in table_rows(representation_table(found))[0]


@pytest.mark.view
def test_aligned_caption_puts_the_deviation_in_parentheses() -> None:
    found = [ns_mass("baseline", "MLP", 2.0, 0.1)]
    caption = re.findall(r"\\caption\{(.*)\}\n", representation_table(found, "aligned"))[0]
    assert ("in parentheses" in caption, "subscript" in caption) == (True, False)


@pytest.mark.view
def test_representation_table_transposed_lists_the_models_as_rows() -> None:
    found = [ns_mass("baseline", "k-NN", 2.0, 0.1), ns_mass("log10 M", "MLP", 2.5, 0.1)]
    table = representation_table(found, shape="transposed")
    assert [row.split(" & ")[1].strip() for row in table_rows(table)] == ["$k$-NN", "MLP"]


@pytest.mark.view
def test_representation_table_split_writes_two_tables() -> None:
    bh = Score("black_holes", "mass", "baseline", "MLP", 3.0, 3.0, 0.1)
    table = representation_table([ns_mass("baseline", "MLP", 2.0, 0.1), bh], shape="split")
    labels = re.findall(r"\\label\{(tab:[^}]*)\}", table)
    assert labels == ["tab:representation", "tab:representation-bh"]


@pytest.mark.command
def test_a_job_logs_its_done_line_at_the_runs_level_in_a_fresh_process(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A worker started by forkserver keeps the root logger at WARNING, as pytest leaves it.
    batch = Batch(
        id="0190a000-0000-7000-8000-000000000000",
        ledger=tmp_path,
        journal=tmp_path / "journal.log",
        commit="abc1234",
        dirty=False,
        started=datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC),
    )
    context = Context(
        batch=batch,
        knobs=KNOBS,
        training=SHORT,
        log_level="INFO",
    )
    job = partial(screen_job, Work(toy_job("baseline", "pointwise", "k-NN"), TOY, context))
    run_jobs([job], workers=1, log_level=context.log_level)
    assert "job done" in caplog.text


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory, toy_inputs: ToyInputs) -> Path:
    """One main run on the toy tables (60_results/conftest.py), one seed and one worker, the
    commit and clock injected; returns its working folder (assets/ and state/ inside)."""
    folder = tmp_path_factory.mktemp("screen")
    ns, bh, split_file = toy_inputs(folder)
    network = {"width": 8, "depth": 1, "max_epochs": 2, "batch_size": 8, "workers": 1}
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "methodology": {"algorithms": network}}
    )
    paths = [str(ns), str(bh), str(split_file), str(folder / "assets"), str(folder / "state")]
    knobs = ["--k", "2", "--neighbours", "1000", "--knots", "2", "--seeds", "1"]
    main(
        [*paths, *knobs],
        config,
        commit=lambda: "abc1234",
        dirty=lambda: False,
        now=lambda: datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC),
    )
    return folder


@pytest.mark.functional
def test_main_writes_a_table_row_for_every_representation_cell(ran: Path) -> None:
    table = ran / "assets" / SECTION / f"{SECTION}_tab_representation.tex"
    assert len(table_rows(table.read_text())) == 14


def rebuild_from_batch(ran: Path, assets: Path) -> None:
    """Run main again on the toy inputs, rebuilding the table from the fixture run's batch."""
    entry = next((ran / "state" / "ledger").glob("*.json"))
    batch = json.loads(entry.read_text())["meta"]["batch"]
    paths = [str(ran / name) for name in ("ns.parquet", "bh.parquet", "split.parquet")]
    knobs = ["--k", "2", "--neighbours", "1000", "--knots", "2", "--seeds", "1"]
    main([*paths, str(assets), str(ran / "state"), *knobs, "--batch", batch])


@pytest.mark.functional
def test_rebuilding_the_table_from_a_batch_fits_nothing(ran: Path, tmp_path: Path) -> None:
    before = len(list((ran / "state" / "ledger").glob("*.json")))
    rebuild_from_batch(ran, tmp_path / "assets")
    assert len(list((ran / "state" / "ledger").glob("*.json"))) == before


@pytest.mark.functional
def test_main_runs_every_job_as_a_completed_trial_of_one_study(ran: Path) -> None:
    storage = JournalStorage(JournalFileBackend(str(ran / "state" / "optuna" / "journal.log")))
    (summary,) = optuna.get_all_study_summaries(storage)
    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    complete = study.get_trials(states=(TrialState.COMPLETE,))
    assert len(complete) == 14 * 2 + 10
