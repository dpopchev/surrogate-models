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
    """One fitted run: what, from where, when, how long and how well."""

    dataset: str
    target: str
    training: Training
    commit: str
    started: datetime
    seconds: float
    epochs: int
    best_epoch: int
    mare: float
    rmse: float


# --- pure functions ---------------------------------------------------------------------------


def run_name(record: RunRecord) -> str:
    """<dataset>-<target>-<UTC start stamp>: the run's folder name, sorting by time."""
    return f"{record.dataset}-{record.target}-{record.started:%Y%m%dT%H%M%SZ}"


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
