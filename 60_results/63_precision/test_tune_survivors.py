"""Facts about the tuning behind Section 5.3, on tiny synthetic data."""

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.distributions import CategoricalDistribution
from tune_survivors import (
    SECTION,
    Job,
    Tuned,
    Walled,
    jobs,
    main,
    outcomes_from_json,
    outcomes_to_json,
    run_job,
    screen_figures,
    tuning_table,
    walled,
)

from shared.candidates import Candidate
from shared.ceilings import Spread
from shared.config import PaperConfig
from shared.design import Design
from shared.harness import Run, Timing
from shared.runs import RunMetadata, entry_from_run
from shared.scorecard import Scorecard

# Eight curves of five rows, y = 1 + 0.1 p + x: curves 6 and 7 are test; the six training
# curves fall into two frozen folds.
P = np.repeat(np.arange(8.0), 5)
X_ALONG = np.tile(np.linspace(0.0, 1.0, 5), 8)
TOY = Design(
    X=np.column_stack([X_ALONG, P]),
    y=1.0 + 0.1 * P + X_ALONG,
    groups=P.astype(int),
    test=P >= 6,
    fold=np.where(P >= 6, -1, P.astype(int) % 2),
    ablation=P == 7,
)


def truth(x_fit, y_fit, x_valid, y_valid, seed):
    """A toy fitter predicting the true y."""
    return lambda x: 1.0 + 0.1 * x[:, 1] + x[:, 0]


def slow(x_fit, y_fit, x_valid, y_valid, seed):
    """A toy fitter that takes a second to fit."""
    time.sleep(1.0)
    return truth(x_fit, y_fit, x_valid, y_valid, seed)


SLOW_OR_NOT = {"slow": CategoricalDistribution((True, False))}
JOB = Job("toy", "mass", Candidate("k-NN", "pointwise"))


def slow_or_not(params):
    """A toy family: slow when the trial says so, else the truth."""
    return slow if params["slow"] else truth


def tuned_toy(first_slow: bool, then_slow: bool):
    """The outcome of a two-trial toy job whose trials are slow as given, walled at 0.05 s."""
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"slow": first_slow})
    study.enqueue_trial({"slow": then_slow})
    return run_job(JOB, TOY, slow_or_not, SLOW_OR_NOT, 2, 0, 0.34, 0.05, study, lambda run: "entry")


def test_a_trial_past_the_wall_fails_and_the_job_goes_on() -> None:
    assert cast(Tuned, tuned_toy(first_slow=True, then_slow=False)).walls == 1


def test_a_job_every_trial_of_which_meets_the_wall_is_walled() -> None:
    assert isinstance(tuned_toy(first_slow=True, then_slow=True), Walled)


def spread_at(p95: float) -> Spread:
    """A spread whose 95th percentile is p95."""
    return Spread(median=p95 / 2, p95=p95, max=2 * p95, mean=p95 / 2)


def screened(rows: int, p95: float, batch: str = "screen"):
    """A toy ledger entry of the k-NN pointwise on the toy pair at that many rows, every fold
    at that p95."""
    return entry_from_run(
        Run(
            seed=0,
            test=Scorecard(zones={"test": spread_at(p95)}, decades={}),
            folds=(spread_at(p95), spread_at(p95)),
            timing=Timing(fit=1.0, predict_one=1e-3, predict_batch=2e-3),
            predictions=np.array([1.0]),
        ),
        RunMetadata(
            id=f"{batch}-{rows}",
            batch=batch,
            dataset="toy",
            target="mass",
            candidate="k-NN",
            settings={"unit": "pointwise", "rows": rows},
            commit="abc1234",
            dirty=False,
            started=datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC),
            epochs=None,
            best_epoch=None,
        ),
    )


def test_a_screen_figure_is_the_one_on_the_most_rows_the_candidate_ran() -> None:
    figures = screen_figures([screened(100, 1e-2), screened(1000, 1e-3)], "screen")
    assert figures[JOB] == pytest.approx(3.0)


NS_MASS_RBF = Job("neutron_stars", "mass", Candidate("local RBF", "pointwise"))
NS_MASS_RBF_CW = Job("neutron_stars", "mass", Candidate("local RBF", "curve-wise"))


def test_the_tuning_table_holds_one_row_per_job() -> None:
    outcomes = [
        Tuned(NS_MASS_RBF, {"neighbours": 80}, 3.4, 3.3, 2, 0, "entry"),
        Walled(NS_MASS_RBF_CW, "all 2 trials past 30 min"),
    ]
    table = tuning_table(outcomes, {NS_MASS_RBF: 3.28})
    body = table.split("\\midrule\n")[1].split("\\bottomrule")[0]
    assert body.count("\\\\\n") == 2


def test_a_saved_batch_reads_back_as_its_outcomes() -> None:
    outcomes = (
        Tuned(NS_MASS_RBF, {"neighbours": 80, "weights": "uniform"}, 3.4, 3.3, 2, 1, "entry"),
        Walled(NS_MASS_RBF_CW, "all 2 trials past 30 min"),
    )
    assert outcomes_from_json(outcomes_to_json(outcomes)) == outcomes


CURVES = range(1, 11)
TOY_LABELS = ["test", "test", *["fold0"] * 4, *["fold1"] * 4]


def toy_inputs(folder: Path) -> tuple[Path, Path, Path]:
    """Tiny NS- and BH-shaped tables and one split for both: two curves of each in test, four in
    each of two folds."""
    ns_rows = [
        {"beta": float(i), "lambda": float(i * i), "rho_c": 10.0 ** (1 + j / 7)}
        | {"M": 1.0 + 0.1 * j + 0.01 * i, "D": 0.1 + 0.01 * j}
        for i in CURVES
        for j in range(8)
    ]
    bh_rows = [
        {"r_h": 4.0 + j, "beta": float(i), "M": 2.0 + 0.5 * j + 0.01 * i, "D": 0.3 - 0.02 * j}
        for i in CURVES
        for j in range(8)
    ]
    split = [
        {"dataset": "neutron_stars", "beta": float(i), "lambda": float(i * i)}
        | {"label": label, "ablation": False}
        for i, label in zip(CURVES, TOY_LABELS, strict=True)
    ] + [
        {"dataset": "black_holes", "beta": float(i), "label": label, "ablation": False}
        for i, label in zip(CURVES, TOY_LABELS, strict=True)
    ]
    ns, bh, split_file = folder / "ns.parquet", folder / "bh.parquet", folder / "split.parquet"
    pd.DataFrame(ns_rows).to_parquet(ns)
    pd.DataFrame(bh_rows).to_parquet(bh)
    pd.DataFrame(split).to_parquet(split_file)
    return ns, bh, split_file


TOY_CONFIG = PaperConfig.model_validate(
    {
        "plot": {"usetex": False},
        "methodology": {"algorithms": {"width": 8, "depth": 1, "max_epochs": 2, "batch_size": 8}},
    }
)
TOY_KNOBS = [
    *("--k", "2", "--neighbours", "4", "--knots", "4"),
    *("--trials", "2", "--minutes", "30", "--workers", "1"),
]


def toy_paths(folder: Path, assets: str = "assets") -> list[str]:
    """The toy run's inputs (written once) and its asset and state folders, as arguments."""
    ns, bh, split_file = (folder / name for name in ("ns.parquet", "bh.parquet", "split.parquet"))
    if not ns.exists():
        toy_inputs(folder)
    return [str(ns), str(bh), str(split_file), str(folder / assets), str(folder / "state")]


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One main run on the toy tables, two trials a job, one worker, the commit and clock
    injected; returns its working folder (assets/ and state/ inside)."""
    folder = tmp_path_factory.mktemp("tuning")
    main(
        [*toy_paths(folder), *TOY_KNOBS],
        TOY_CONFIG,
        commit=lambda: "abc1234",
        dirty=lambda: False,
        now=lambda: datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC),
    )
    return folder


def test_a_saved_batch_rebuilds_the_table(ran: Path) -> None:
    (saved,) = (ran / "state" / SECTION).glob("*.json")
    main([*toy_paths(ran, "rebuilt"), *TOY_KNOBS, "--batch", saved.stem], TOY_CONFIG)
    assert (ran / "rebuilt" / SECTION / f"{SECTION}_tab_tuning.tex").stat().st_size > 0


def test_main_writes_a_table_row_per_job(ran: Path) -> None:
    table = (ran / "assets" / SECTION / f"{SECTION}_tab_tuning.tex").read_text()
    body = table.split("\\midrule\n")[1].split("\\bottomrule")[0]
    assert body.count("\\\\\n") == 14


def test_one_job_per_pair_and_short_listed_candidate() -> None:
    assert len(set(jobs())) == 14


def test_a_fit_past_the_wall_predicts_nan() -> None:
    predict = walled(slow, seconds=0.05)(TOY.X, TOY.y, TOY.X, TOY.y, 0)
    assert np.isnan(predict(TOY.X)).all()


def test_a_fitter_past_its_wall_skips_its_later_fits() -> None:
    fit = walled(slow, seconds=0.05)
    fit(TOY.X, TOY.y, TOY.X, TOY.y, 0)
    started = time.monotonic()
    fit(TOY.X, TOY.y, TOY.X, TOY.y, 0)
    assert time.monotonic() - started < 0.05
