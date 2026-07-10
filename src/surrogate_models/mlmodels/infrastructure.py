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
materialising its weights. load_trained_run is the write-side LOAD adapter: it reads the
same sidecar, re-certifies its primitives back through the domain smart constructors
(the trust boundary), and assembles a TrainedRun via configure_run -> complete_training
-- still torch-free, so the live nn is rebuilt later by materialize_model, on demand.

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

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning
import torch
from torch import nn

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
    InvalidRunID,
    ModelIdentityMismatch,
    RunSummaryDTO,
    TrainedRun,
    TrainingConfig,
    TrainingConfigError,
    complete_training,
    configure_run,
    make_runid,
    make_training_config,
    verify_fingerprint,
)
from surrogate_models.railway_adts import ErrorInfo, Result, fmap_error, safe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The ``{run_id}.json`` sidecar persisted alongside a run's checkpoint.

    A boundary record (primitive fields only -- it is a row crossing an I/O boundary,
    not a domain type) holding the metadata both projections need WITHOUT loading torch.
    A SUPERSET of every read: the run id and the trained model's declared identity
    (name, version) feed the read-side RunSummaryDTO, while the four training knobs
    (max_epochs, learning_rate, batch_size, optimizer) let the LOAD side rebuild the
    run's certified TrainingConfig -- the "optimizer/loss recorded as declared
    metadata" of fork 1A. The fingerprint is the model's structural digest at save
    time (see structural_fingerprint) -- the SHAPE half of the reload guard that
    materialize_model checks so a checkpoint cannot be loaded into a drifted
    architecture. Grows to carry the split, provenance, and metrics as their
    persistence slices land. load_trained_run certifies these primitives back into
    domain value objects when hydrating the aggregate; the read-side Find projects the
    identity fields straight into a DTO.
    """

    run_id: str
    model_name: str
    model_version: str
    max_epochs: int
    learning_rate: float
    batch_size: int
    optimizer: str
    fingerprint: str


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
        max_epochs=payload["max_epochs"],
        learning_rate=payload["learning_rate"],
        batch_size=payload["batch_size"],
        optimizer=payload["optimizer"],
        fingerprint=payload["fingerprint"],
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


@safe(OSError, fmap_error(lambda cause: cause, code="RUN_LOAD_FAILED"))
def _read_manifest(checkpoint_dir: Path, run_id: str) -> RunManifest:
    """Read the LOAD sidecar at the @safe I/O boundary.

    Separate from find's read boundary so LOAD carries its own code: ``@safe`` folds a
    missing or unreadable ``{run_id}.json`` (OSError) into ``RUN_LOAD_FAILED``, while a
    malformed json or missing key propagates as a corruption/logic fault -- symmetric
    with find_run_summary.
    """
    return read_run_manifest(checkpoint_dir, run_id)


def _load_corruption(error: InvalidRunID | TrainingConfigError) -> ErrorInfo:
    """Fold a manifest whose primitives no longer re-certify into a corruption cause.

    Our own save wrote this manifest, so a re-certification failure means the sidecar
    was corrupted or tampered with -- one uniform ``RUN_LOAD_CORRUPT`` code, the
    specific domain rejection preserved in the message. Typed against both
    re-certification rails so the same lift folds an invalid id and a bad knob alike.
    """
    return ErrorInfo(
        code="RUN_LOAD_CORRUPT",
        message=f"manifest failed re-certification: {error!r}",
    )


def _reconstitute_run(
    checkpoint_dir: Path, manifest: RunManifest
) -> Result[TrainedRun, ErrorInfo]:
    """Re-certify a read-back manifest's primitives and assemble the TrainedRun.

    The trust boundary for LOAD: the manifest's raw id and training knobs are run back
    through the same smart constructors that first built them (make_runid,
    make_training_config), each rejection folded to ``RUN_LOAD_CORRUPT`` BEFORE the
    chain continues so both rails share one error type; only an all-valid manifest
    assembles a run, via the sanctioned configure_run -> complete_training transition.
    The checkpoint path is DERIVED (``checkpoint_dir/{run_id}.ckpt`` -- deterministic,
    never persisted) and wrapped as an opaque handle; the ckpt itself is not read here.
    """
    checkpoint = Checkpoint(str(checkpoint_dir / f"{manifest.run_id}.ckpt"))
    return (
        make_runid(manifest.run_id)
        .fmap_err(_load_corruption)
        .and_then(
            lambda run_id: (
                make_training_config(
                    manifest.max_epochs,
                    manifest.learning_rate,
                    manifest.batch_size,
                    manifest.optimizer,
                )
                .fmap_err(_load_corruption)
                .fmap(
                    lambda config: complete_training(
                        configure_run(run_id, config), checkpoint
                    )
                )
            )
        )
    )


def load_trained_run(
    checkpoint_dir: Path, run_id: str
) -> Result[TrainedRun, ErrorInfo]:
    """Hydrate a re-certified TrainedRun from ``{run_id}.json`` (src LoadTrainedRunFn).

    The write side's LOAD counterpart of save_trained_run. ``checkpoint_dir`` leads so
    the composition root can bind it via ``partial`` to yield a ``RunID ->
    Result[TrainedRun]`` port. Torch-free: it reads ONLY the manifest sidecar (no ckpt
    load) and hands it to _reconstitute_run, which re-certifies the primitives (the
    trust boundary) and derives the checkpoint path. A missing sidecar folds into
    ``RUN_LOAD_FAILED``; a manifest that no longer certifies folds into
    ``RUN_LOAD_CORRUPT``. The LIVE nn is NOT rebuilt here -- materialize_model does that
    on demand from the checkpoint the hydrated aggregate points at.
    """
    logger.info("load_trained_run: run %s under %s", run_id, checkpoint_dir)
    return _read_manifest(checkpoint_dir, run_id).and_then(
        lambda manifest: _reconstitute_run(checkpoint_dir, manifest)
    )


def structural_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash a model's structure -- its sorted ``parameter name -> shape`` layout.

    The reload guard's SHAPE half: a digest of only the state_dict's keys and tensor
    shapes (never the weight VALUES), so it is identical for a trained and an untrained
    copy of the same architecture but differs the moment a layer's shape drifts. Save
    records it in the manifest; materialize_model recomputes it on the rebuilt nn and
    rejects a mismatch as a ModelIdentityMismatch, catching a checkpoint loaded into the
    wrong architecture before it silently corrupts predictions.
    """
    layout = sorted((key, tuple(tensor.shape)) for key, tensor in state_dict.items())
    return hashlib.sha256(repr(layout).encode()).hexdigest()


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
        RunManifest(
            str(run.run_id),
            model.MODEL_NAME,
            model.MODEL_VERSION,
            run.config.max_epochs,
            run.config.learning_rate,
            run.config.batch_size,
            run.config.optimizer,
            structural_fingerprint(model.state_dict()),
        ),
    )
    return Checkpoint(str(target))


@safe(Exception, fmap_error(lambda cause: cause, code="MODEL_MATERIALIZE_FAILED"))
def _load_live_model(
    factory: Callable[[], lightning.LightningModule], run: TrainedRun
) -> lightning.LightningModule:
    """Build a fresh model via the injected factory and load the ckpt weights into it.

    The torch I/O half of materialize: ``factory()`` (fork 1A's user-owned model) then
    load the state_dict with ``weights_only=True`` -- only weights travel, never a
    pickled class -- from the ckpt the hydrated aggregate points at. ``@safe`` folds a
    missing or unreadable ckpt into ``MODEL_MATERIALIZE_FAILED``.
    """
    logger.info("materialize run %s from %s", run.run_id, run.checkpoint.location)
    model = factory()
    model.load_state_dict(torch.load(run.checkpoint.location, weights_only=True))
    return model


def _reload_mismatch(error: ModelIdentityMismatch) -> ErrorInfo:
    """Fold a reload-guard mismatch into a coded boundary cause.

    The rebuilt model's structure no longer matches the manifest's recorded fingerprint;
    surface it as ``MODEL_IDENTITY_MISMATCH`` with the expected/got values preserved in
    the message, so a reload fails loudly instead of returning a silently wrong model.
    """
    return ErrorInfo(
        code="MODEL_IDENTITY_MISMATCH",
        message=f"reload identity mismatch: {error!r}",
    )


def _guard_reload(
    manifest_dir: Path, run: TrainedRun, model: lightning.LightningModule
) -> Result[lightning.LightningModule, ErrorInfo]:
    """Verify a rebuilt model against its manifest, returning the model if it matches.

    Reads the recorded fingerprint from the sidecar (reusing the @safe manifest read),
    recomputes the live model's structural_fingerprint, and runs the pure domain
    verify_fingerprint guard; a drift folds to ``MODEL_IDENTITY_MISMATCH`` while a match
    passes the model through. Sequential and_then over the @safe read (no nested
    @safe), mirroring load_trained_run.
    """
    got = structural_fingerprint(model.state_dict())
    return _read_manifest(manifest_dir, run.run_id).and_then(
        lambda manifest: (
            verify_fingerprint(run.run_id, manifest.fingerprint, got)
            .fmap_err(_reload_mismatch)
            .fmap(lambda _: model)
        )
    )


def materialize_model(
    factory: Callable[[], lightning.LightningModule],
    manifest_dir: Path,
    run: TrainedRun,
) -> Result[lightning.LightningModule, ErrorInfo]:
    """Rebuild ``run``'s LIVE nn on demand, guarded by its recorded fingerprint.

    The torch-bearing counterpart of load_trained_run: the metadata aggregate points at
    a checkpoint, and this turns it back into a usable model, so the live nn never has
    to cross into domain/application. ``factory`` and ``manifest_dir`` lead so the
    composition root binds them via ``partial`` to yield the design's ``TrainedRun ->
    Result[LightningModule]`` shape. Builds + loads the weights (_load_live_model), then
    verifies the rebuilt structure against the manifest (_guard_reload): a load failure
    is ``MODEL_MATERIALIZE_FAILED``, a structural drift is ``MODEL_IDENTITY_MISMATCH``.
    The declared name/version half of the guard lands in the next slice.
    """
    return _load_live_model(factory, run).and_then(
        lambda model: _guard_reload(manifest_dir, run, model)
    )
