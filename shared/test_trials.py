"""Facts about the harness's fits as Optuna trials, on in-memory studies and a hand-made entry."""

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from skorch.history import History

from shared.ceilings import Spread
from shared.harness import Run, Timing
from shared.runs import RunMetadata, entry_from_run
from shared.scorecard import Scorecard
from shared.trials import ReportEpochs, open_study, tell_run


def spread_at(p95: float) -> Spread:
    """A spread whose 95th percentile is p95."""
    return Spread(median=p95 / 2, p95=p95, max=2 * p95, mean=p95 / 2)


ENTRY = entry_from_run(
    Run(
        seed=1,
        test=Scorecard(zones={"test": spread_at(1e-2)}, decades={}),
        folds=(spread_at(1e-2), spread_at(1e-3)),
        timing=Timing(fit=2.0, predict_one=1e-3, predict_batch=2e-3),
        predictions=np.array([1.0]),
    ),
    RunMetadata(
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
    ),
)


def net_at(epoch: int, valid_mare: float) -> SimpleNamespace:
    """A stand-in network whose history holds one epoch."""
    history = History()
    history.new_epoch()
    history.record("epoch", epoch)
    history.record("valid_mare", valid_mare)
    return SimpleNamespace(history=history)


def test_a_reported_epoch_is_an_intermediate_value_of_the_trial() -> None:
    study = open_study("toy", None)
    trial = study.ask()
    ReportEpochs(trial).on_epoch_end(net_at(3, 0.25))
    assert study.trials[0].intermediate_values == {3: 0.25}


def test_a_told_run_completes_the_trial_with_its_mean_fold_figures() -> None:
    study = open_study("toy", None)
    tell_run(study, study.ask(), ENTRY)
    assert study.trials[0].value == pytest.approx(2.5)


def test_a_told_trial_carries_its_ledger_entry_s_identity() -> None:
    study = open_study("toy", None)
    tell_run(study, study.ask(), ENTRY)
    assert study.trials[0].user_attrs == {
        "ledger_id": "0190a000-0000-7000-8000-000000000001",
        "batch": "0190a000-0000-7000-8000-000000000000",
        "dataset": "toy",
        "target": "mass",
        "candidate": "MLP",
        "seed": 1,
        "commit": "abc1234",
        "dirty": False,
        "test_figures": 2.0,
        "fit_s": 2.0,
    }
