"""The harness's fits as trials of an Optuna study (W-081).

One study per run of a make target (its batch), on a journal file every worker process writes,
so optuna-dashboard follows the running fits live. A network reports its valid_mare per epoch;
a finished fit tells the trial its mean fold significant figures -- never a test score -- and
names its W-077 ledger entry, which stays the scored record (scorecard, predictions, timing).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.exceptions import UpdateFinishedTrialError
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from skorch.callbacks import Callback

from shared.runs import LedgerEntry
from shared.scorecard import significant_figures

logger = logging.getLogger(__name__)


def open_study(name: str, journal: Path | None) -> optuna.Study:
    """The study of that name on the journal file (created or joined), or in memory without
    one; it maximizes the fold significant figures and prunes nothing. Optuna's INFO lines
    (a study made or joined, a trial told) are silenced in this process; its warnings show."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if journal is None:
        return optuna.create_study(study_name=name, direction="maximize")
    journal.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal)))
    return optuna.create_study(
        study_name=name, storage=storage, direction="maximize", load_if_exists=True
    )


class ReportEpochs(Callback):
    """Report a network's valid_mare to its trial after every epoch, at offset + epoch: one
    trial spans several fits (a harness run fits every fold, then all curves), each numbering
    its epochs from 1, so the offset carries the epochs of the trial's earlier fits."""

    def __init__(self, trial: optuna.Trial, offset: int = 0) -> None:
        self.trial = trial
        self.offset = offset

    def on_epoch_end(
        self, net: Any, dataset_train: Any = None, dataset_valid: Any = None, **kwargs: Any
    ) -> None:
        """Report this epoch's valid_mare at its epoch number."""
        step = self.offset + int(net.history[-1, "epoch"])
        try:
            self.trial.report(float(net.history[-1, "valid_mare"]), step)
        except UpdateFinishedTrialError:
            # The trial is only a live view: one finished elsewhere (a dashboard) loses its
            # reports, never the fit (W-085).
            logger.warning(
                "trial %d was finished elsewhere; epoch %d not reported", self.trial.number, step
            )


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
    folds = float(np.mean([significant_figures(fold.p95) for fold in entry.folds]))
    try:
        for key, value in attributes.items():
            trial.set_user_attr(key, value)
        study.tell(trial, folds)
    except UpdateFinishedTrialError:
        # Finished elsewhere (W-085): the ledger entry is the record, so the run goes on.
        logger.warning(
            "trial %d was finished elsewhere; ledger entry %s not told", trial.number, meta.id
        )
