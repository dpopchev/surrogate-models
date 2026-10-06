"""The record each surrogate run leaves beside its diagnostics (W-035).

A RunRecord names what was trained (dataset, target, the Training settings), from which commit
and when, how long it took and how it scored; it round-trips through JSON so local/state/<run>/
run.json can be read back without rerunning.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime

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
