"""The harness's fits as trials of an Optuna study (W-081).

One study per run of a make target (its batch), on a journal file every worker process writes,
so optuna-dashboard follows the running fits live. A network reports its valid_mare per epoch;
a finished fit tells the trial its mean fold significant figures -- never a test score -- and
names its W-077 ledger entry, which stays the scored record (scorecard, predictions, timing).
"""

from pathlib import Path
from typing import Any

import numpy as np
import optuna
from skorch.callbacks import Callback

from shared.runs import LedgerEntry
from shared.scorecard import significant_figures


def open_study(name: str, journal: Path | None) -> optuna.Study:
    """The study of that name on the journal file (created or joined), or in memory without
    one; it maximizes the fold significant figures and prunes nothing."""
    return optuna.create_study(study_name=name, direction="maximize")


class ReportEpochs(Callback):
    """Report a network's valid_mare to its trial after every epoch."""

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial

    def on_epoch_end(
        self, net: Any, dataset_train: Any = None, dataset_valid: Any = None, **kwargs: Any
    ) -> None:
        """Report this epoch's valid_mare at its epoch number."""
        self.trial.report(float(net.history[-1, "valid_mare"]), int(net.history[-1, "epoch"]))


def tell_run(study: optuna.Study, trial: optuna.Trial, entry: LedgerEntry) -> None:
    """Complete the trial with the entry's mean fold p95 significant figures, the entry's
    identity, test figures and fit seconds kept as its user attributes."""
    meta = entry.meta
    attributes: dict[str, Any] = {
        "ledger_id": meta.id,
        "batch": meta.batch,
        "dataset": meta.dataset,
        "target": meta.target,
        "candidate": meta.candidate,
        "seed": entry.seed,
        "commit": meta.commit,
        "dirty": meta.dirty,
        "test_figures": significant_figures(entry.test.zones["test"].p95),
        "fit_s": entry.timing.fit,
    }
    for key, value in attributes.items():
        trial.set_user_attr(key, value)
    folds = float(np.mean([significant_figures(fold.p95) for fold in entry.folds]))
    study.tell(trial, folds)
