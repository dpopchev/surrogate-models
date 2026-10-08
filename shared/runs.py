"""The record each surrogate run leaves beside its diagnostics (W-035), and its ledger entry.

A RunRecord names what was trained (dataset, target, the Training settings), from which commit
and when, how long it took and how it scored; it round-trips through JSON so local/state/<run>/
run.json can be read back without rerunning. A LedgerEntry (W-077) is the same for any
candidate scored through the harness, one per seed, with the fold spreads, the test scorecard,
the timing and the test predictions, so a search's progress can be rebuilt without refitting.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime

from shared.ceilings import Spread
from shared.harness import Run, Timing
from shared.scorecard import Scorecard
from shared.surrogate import Training

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """One fitted run: what, from where, when, how long and how well; `dirty` says the code
    held uncommitted changes, so `commit` alone does not reproduce it."""

    dataset: str
    target: str
    training: Training
    commit: str
    dirty: bool
    started: datetime
    seconds: float
    epochs: int
    best_epoch: int
    mare: float
    rmse: float


Setting = str | int | float | bool | None


@dataclass(frozen=True)
class RunMetadata:
    """What a harness Run does not know about itself: its ledger id (a uuid7, made by the
    shell, so ids sort by creation time), the batch
    (one make invocation) it belongs to, the pair, the candidate and its settings, the code it
    ran from, when it started, and the epochs where the candidate has them."""

    id: str
    batch: str
    dataset: str
    target: str
    candidate: str
    settings: dict[str, Setting]
    commit: str
    dirty: bool
    started: datetime
    epochs: int | None
    best_epoch: int | None


@dataclass(frozen=True)
class LedgerEntry:
    """One seed of one candidate on one pair, as the ledger keeps it."""

    meta: RunMetadata
    seed: int
    folds: tuple[Spread, ...]
    test: Scorecard
    timing: Timing
    predictions: tuple[float, ...]
    ripple: tuple[float, ...]


# --- pure functions ---------------------------------------------------------------------------


def run_stem(dataset: str, target: str, started: datetime) -> str:
    """<dataset>-<target>-<UTC start stamp>: a run's folder name, known before its fit and
    sorting by time."""
    return f"{dataset}-{target}-{started:%Y%m%dT%H%M%SZ}"


def run_name(record: RunRecord) -> str:
    """The folder name of a recorded run."""
    return run_stem(record.dataset, record.target, record.started)


def to_json(record: RunRecord) -> str:
    """The record as indented JSON, the start time in ISO 8601."""
    fields = asdict(record) | {"started": record.started.isoformat()}
    return json.dumps(fields, indent=2) + "\n"


def from_json(text: str) -> RunRecord:
    """The record a to_json text holds."""
    fields = json.loads(text)
    return RunRecord(
        **fields
        | {
            "training": Training(**fields["training"]),
            "started": datetime.fromisoformat(fields["started"]),
        }
    )


def entry_name(entry: LedgerEntry) -> str:
    """<id>.json: the entry's file in the ledger; with uuid7 ids, later entries sort later."""
    return f"{entry.meta.id}.json"


def entry_from_run(run: Run, meta: RunMetadata) -> LedgerEntry:
    """The ledger entry of a harness run."""
    predictions = tuple(float(p) for p in run.predictions)
    return LedgerEntry(meta, run.seed, run.folds, run.test, run.timing, predictions, run.ripple)


def entry_to_json(entry: LedgerEntry) -> str:
    """The entry as indented JSON, the start time in ISO 8601, the predictions as a list."""
    fields = asdict(entry)
    fields["meta"]["started"] = entry.meta.started.isoformat()
    return json.dumps(fields, indent=2) + "\n"


def entry_from_json(text: str) -> LedgerEntry:
    """The entry an entry_to_json text holds; JSON keys are strings, so the decades of D are
    read back as integers; an entry written before the ripple (T-170) reads back without one."""
    fields = json.loads(text)
    meta = fields["meta"] | {"started": datetime.fromisoformat(fields["meta"]["started"])}
    test = fields["test"]
    return LedgerEntry(
        meta=RunMetadata(**meta),
        seed=fields["seed"],
        folds=tuple(Spread(**s) for s in fields["folds"]),
        test=Scorecard(
            zones={name: Spread(**s) for name, s in test["zones"].items()},
            decades={int(d): Spread(**s) for d, s in test["decades"].items()},
        ),
        timing=Timing(**fields["timing"]),
        predictions=tuple(fields["predictions"]),
        ripple=tuple(fields.get("ripple", ())),
    )
