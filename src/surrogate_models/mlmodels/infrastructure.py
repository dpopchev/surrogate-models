"""Mlmodels infrastructure -- the imperative shell of the mlmodels context.

Provides the src implementation of the application's write port: save_trained_run
builds a run's model FROM its certified config and persists a checkpoint. It knows the
domain only enough to read a ConfiguredRun and hand back a Checkpoint -- it never
exposes a domain type outward.

Persistence is two artifacts per run: the Lightning ``{run_id}.ckpt`` (live weights)
and a ``{run_id}.json`` sidecar (metadata for the read side). RunManifest models that
sidecar and write_run_manifest / read_run_manifest round-trip it as JSON with no torch
import, so Find and Load stay cheap; the @safe port adapters that call these plain
helpers own the single I/O boundary. find_run_summary is the read adapter: it projects
the manifest into a RunSummaryDTO with no torch load, so a query can show a run without
materialising its weights.

This is the THIN training slice: the adapter persists the model's UNTRAINED weights
(no fit, no data, no Trainer yet) so the composition seat's happy path is reachable
with a real torch artifact on disk. The initial model is a minimal single-feature
linear regressor; the EXPAND slice widens it to the real feature count and trains it
over prepared data (replacing torch.save with a fitted Trainer checkpoint). The
context's multi-epoch training proof lives in tests, wiring the SAME application port
to a test-owned adapter that actually fits. Curried on ``checkpoint_dir`` at the
composition root, save_trained_run satisfies the application's SaveTrainedRunFn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning
import torch
from torch import nn

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
    RunSummaryDTO,
    TrainingConfig,
)
from surrogate_models.railway_adts import fmap_error, safe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The ``{run_id}.json`` sidecar persisted alongside a run's checkpoint.

    A boundary record (primitive fields only -- it is a row crossing an I/O boundary,
    not a domain type) holding the metadata the read side needs WITHOUT loading torch:
    the run id and the trained model's declared identity. Grows to carry the split,
    provenance, metrics, and structural fingerprint as their persistence slices land.
    A later mapper certifies these primitives back into domain value objects when
    hydrating the aggregate; the read-side Find projects them straight into a DTO.
    """

    run_id: str
    model_name: str
    model_version: str


def write_run_manifest(manifest_dir: Path, manifest: RunManifest) -> Path:
    """Write ``manifest`` to ``{run_id}.json`` under ``manifest_dir``; return the path.

    The sidecar writer the SaveTrainedRunFn adapter calls after checkpointing.
    Serialises the manifest's primitive fields as JSON so the read side can recover
    them with no torch import. Creates ``manifest_dir`` if missing. A plain helper
    (not ``@safe``): the port adapter that calls it owns the single ``@safe``
    boundary, so wrapping here would nest.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / f"{manifest.run_id}.json"
    target.write_text(json.dumps(asdict(manifest)))
    return target


def read_run_manifest(manifest_dir: Path, run_id: str) -> RunManifest:
    """Read ``{run_id}.json`` under ``manifest_dir`` back into a RunManifest.

    The read-side counterpart to write_run_manifest, recovering the persisted
    metadata with no torch import so Find and Load stay cheap. A plain helper (not
    ``@safe``): the port adapter that calls it owns the single ``@safe`` boundary.
    """
    payload = json.loads((manifest_dir / f"{run_id}.json").read_text())
    return RunManifest(
        run_id=payload["run_id"],
        model_name=payload["model_name"],
        model_version=payload["model_version"],
    )


@safe(OSError, fmap_error(lambda cause: cause, code="RUN_SUMMARY_READ_FAILED"))
def find_run_summary(manifest_dir: Path, run_id: str) -> RunSummaryDTO:
    """Project the ``{run_id}.json`` manifest into a RunSummaryDTO read model.

    The read side's lookup adapter. ``manifest_dir`` leads so the composition root binds
    it via ``partial`` to yield the application's ``FindRunSummaryFn``. It reads ONLY
    the manifest sidecar (via read_run_manifest -- json, NO torch load) and maps its
    primitives straight into a RunSummaryDTO, BYPASSING the aggregate and its
    certification (FIND a view to show it). ``@safe`` folds a missing or unreadable
    manifest (OSError) into an ErrorInfo cause (``RUN_SUMMARY_READ_FAILED``); a
    non-OSError (malformed json, a missing key) propagates as a corruption/logic fault,
    symmetric with the datasets find_dataset_frame read boundary.
    """
    logger.debug("reading run summary %s from %s", run_id, manifest_dir)
    manifest = read_run_manifest(manifest_dir, run_id)
    return RunSummaryDTO(manifest.run_id, manifest.model_name, manifest.model_version)


class SurrogateRegressor(lightning.LightningModule):
    """A minimal single-feature linear regressor built FROM a run's config.

    The mlmodels context's initial surrogate model -- one ``1->1`` linear layer that
    remembers the run's optimiser choice and learning rate so a later fit trains under
    the certified config. The thin slice persists this model UNTRAINED, so it holds
    only what serialising its ``state_dict`` needs; the EXPAND slice widens it to the
    real feature count and adds the forward pass, training step, and optimiser under
    its own tests. Not the final neutron-star architecture -- the honest starting point.

    Declares its own ``MODEL_NAME`` / ``MODEL_VERSION`` -- the identity a run records in
    its manifest so a reload can prove the class it loads weights into matches the one
    that trained. Identity is the model's property, so it lives ON the model; when fork
    1A injects a user-owned model, that model brings its own declared identity the same
    way and this adapter records whatever model it is handed.
    """

    MODEL_NAME = "surrogate-regressor"
    MODEL_VERSION = "0.1.0"

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.learning_rate = config.learning_rate
        self.optimizer_name = config.optimizer


@safe(Exception, fmap_error(lambda cause: cause, code="TRAINING_FAILED"))
def save_trained_run(checkpoint_dir: Path, run: ConfiguredRun) -> Checkpoint:
    """Build ``run``'s model and persist ``{run_id}.ckpt`` -- the src SaveTrainedRunFn.

    ``checkpoint_dir`` leads so the composition root binds it via ``partial`` to yield
    the application's ``SaveTrainedRunFn``. THIN slice: it builds a SurrogateRegressor
    from the run's certified config and writes its UNTRAINED weights
    (``torch.save(state_dict)``) to ``{run_id}.ckpt`` under the checkpoint dir. It ALSO
    writes the ``{run_id}.json`` manifest sidecar carrying the model's declared
    identity, so the two artifacts land together and the read side can recover the
    metadata with no torch import. Returns the Checkpoint pointing at the ckpt path --
    no fit, no data, no Trainer yet, so the seat's happy path is reachable on disk.
    ``@safe`` folds any I/O or serialisation failure into an ErrorInfo cause
    (``TRAINING_FAILED``). When the EXPAND slice trains for real, its fitted Trainer
    checkpoint replaces the weight write and every ring above stays unchanged.
    """
    logger.info("save_trained_run: run %s under %s", run.run_id, checkpoint_dir)
    model = SurrogateRegressor(run.config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"{run.run_id}.ckpt"
    torch.save(model.state_dict(), target)
    write_run_manifest(
        checkpoint_dir,
        RunManifest(str(run.run_id), model.MODEL_NAME, model.MODEL_VERSION),
    )
    return Checkpoint(str(target))
