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
    InvalidDatasetProvenance,
    InvalidHoldoutSpec,
    InvalidMetrics,
    InvalidRunID,
    ModelIdentity,
    ModelIdentityMismatch,
    RunSummaryDTO,
    TrainedRun,
    TrainingConfig,
    TrainingConfigError,
    complete_training,
    configure_run,
    make_dataset_provenance,
    make_holdout_spec,
    make_metrics,
    make_runid,
    make_training_config,
    verify_fingerprint,
    verify_identity,
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


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """The ``{run_id}.metrics.json`` sidecar persisted alongside a run's checkpoint.

    A boundary record (primitive fields only) holding a run's measured quality
    SEPARATELY from the RunManifest, because a run is evaluated AFTER it is saved and
    materialised -- the untrained ``save_trained_run`` toy has no metrics to write, so
    coupling metrics into the manifest would force it to invent them. Kept as its own
    sidecar so save_trained_run and RunManifest stay untouched; record_run_evaluation
    certifies the RMSE via the domain make_metrics gate before writing this row.
    """

    run_id: str
    val_rmse: float
    test_rmse: float


def write_evaluation_record(record_dir: Path, record: EvaluationRecord) -> Path:
    """Write ``record`` to ``{run_id}.metrics.json`` under ``record_dir``; return path.

    The evaluation-sidecar writer, mirroring write_run_manifest: serialises the record's
    primitive fields as JSON (no torch import) so a run's persisted quality reads back
    without materialising its weights. Creates ``record_dir`` if missing. A plain helper
    (not ``@safe``): the port adapter that calls it owns the single ``@safe`` boundary.
    """
    record_dir.mkdir(parents=True, exist_ok=True)
    target = record_dir / f"{record.run_id}.metrics.json"
    target.write_text(json.dumps(asdict(record)))
    return target


def read_evaluation_record(record_dir: Path, run_id: str) -> EvaluationRecord:
    """Read ``{run_id}.metrics.json`` under ``record_dir`` into an EvaluationRecord.

    The read-side counterpart to write_evaluation_record, recovering a run's persisted
    metrics with no torch import. A plain helper (not ``@safe``): the port adapter that
    calls it owns the single ``@safe`` boundary.
    """
    payload = json.loads((record_dir / f"{run_id}.metrics.json").read_text())
    return EvaluationRecord(
        run_id=payload["run_id"],
        val_rmse=payload["val_rmse"],
        test_rmse=payload["test_rmse"],
    )


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """The ``{run_id}.split.json`` sidecar recording how a run's data was divided.

    A boundary record (primitive fields only) holding the train/val/test fractions and
    the shuffle seed, so a persisted run's split is reproducible and comparable across
    experiments. Its own sidecar (like EvaluationRecord) not a RunManifest field: the
    untrained toy save path does no splitting, so coupling it into the manifest would
    force the toy to invent a split. record_run_split certifies these primitives via the
    domain make_holdout_spec gate before writing this row.
    """

    run_id: str
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int


def write_split_record(record_dir: Path, record: SplitRecord) -> Path:
    """Write ``record`` to ``{run_id}.split.json`` under ``record_dir``; return path.

    The split-sidecar writer, mirroring write_evaluation_record: serialises the record's
    primitive fields as JSON (no torch import). Creates ``record_dir`` if missing. A
    plain helper (not ``@safe``): the port adapter that calls it owns the ``@safe``
    boundary.
    """
    record_dir.mkdir(parents=True, exist_ok=True)
    target = record_dir / f"{record.run_id}.split.json"
    target.write_text(json.dumps(asdict(record)))
    return target


def read_split_record(record_dir: Path, run_id: str) -> SplitRecord:
    """Read ``{run_id}.split.json`` under ``record_dir`` into a SplitRecord.

    The read-side counterpart to write_split_record, recovering a run's persisted split
    with no torch import. A plain helper (not ``@safe``): the port adapter that calls it
    owns the ``@safe`` boundary.
    """
    payload = json.loads((record_dir / f"{run_id}.split.json").read_text())
    return SplitRecord(
        run_id=payload["run_id"],
        train_fraction=payload["train_fraction"],
        val_fraction=payload["val_fraction"],
        test_fraction=payload["test_fraction"],
        seed=payload["seed"],
    )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """The ``{run_id}.provenance.json`` sidecar recording a run's training data.

    A boundary record (primitive fields only) holding the dataset id, feature columns,
    target column, and row count, so a persisted run is self-describing about WHAT it
    trained on without shipping the data. Its own sidecar (like SplitRecord) not a
    RunManifest field: the untrained toy save path sources no data, so coupling it into
    the manifest would force the toy to invent provenance. record_run_provenance
    certifies these primitives via the domain make_dataset_provenance gate on write.
    """

    run_id: str
    dataset_id: str
    feature_columns: tuple[str, ...]
    target_column: str
    n_rows: int


def write_provenance_record(record_dir: Path, record: ProvenanceRecord) -> Path:
    """Write ``record`` to ``{run_id}.provenance.json`` under ``record_dir``.

    The provenance-sidecar writer, mirroring write_split_record: serialises the record's
    primitive fields as JSON (no torch import). Creates ``record_dir`` if missing. A
    plain helper (not ``@safe``): the port adapter that calls it owns the ``@safe``
    boundary.
    """
    record_dir.mkdir(parents=True, exist_ok=True)
    target = record_dir / f"{record.run_id}.provenance.json"
    target.write_text(json.dumps(asdict(record)))
    return target


def read_provenance_record(record_dir: Path, run_id: str) -> ProvenanceRecord:
    """Read ``{run_id}.provenance.json`` under ``record_dir`` into a ProvenanceRecord.

    The read-side counterpart to write_provenance_record. JSON has no tuple, so the
    feature columns come back as a list and are re-wrapped to a tuple, matching the
    frozen record's shape (and the domain DatasetProvenance) for equality. A plain
    helper (not ``@safe``): the port adapter that calls it owns the ``@safe`` boundary.
    """
    payload = json.loads((record_dir / f"{run_id}.provenance.json").read_text())
    return ProvenanceRecord(
        run_id=payload["run_id"],
        dataset_id=payload["dataset_id"],
        feature_columns=tuple(payload["feature_columns"]),
        target_column=payload["target_column"],
        n_rows=payload["n_rows"],
    )


def _invalid_metrics(error: InvalidMetrics) -> ErrorInfo:
    """Fold a domain metrics rejection into a coded boundary cause.

    The shell measured a nonsensical RMSE (below zero); surface it as
    ``INVALID_METRICS`` with the offending pair preserved in the message, so a bad
    evaluation is refused loudly instead of persisted.
    """
    return ErrorInfo(code="INVALID_METRICS", message=f"metrics rejected: {error!r}")


@safe(OSError, fmap_error(lambda cause: cause, code="EVALUATION_WRITE_FAILED"))
def _write_evaluation(record_dir: Path, record: EvaluationRecord) -> EvaluationRecord:
    """Write the evaluation sidecar at the @safe I/O boundary; return the record.

    Separates the fallible write (OSError -> ``EVALUATION_WRITE_FAILED``) from the pure
    metrics certification, so record_run_evaluation chains the two on the railway with
    no nested ``@safe`` -- the same shape as load_trained_run's read + reconstitute.
    """
    write_evaluation_record(record_dir, record)
    return record


def record_run_evaluation(
    record_dir: Path, run_id: str, val_rmse: float, test_rmse: float
) -> Result[EvaluationRecord, ErrorInfo]:
    """Certify a run's measured RMSE and persist it as the evaluation sidecar.

    The metrics write adapter: the shell scores a materialised run against its held-out
    loaders and hands the raw val/test RMSE here. The domain make_metrics gate certifies
    the pair (a negative RMSE folds to ``INVALID_METRICS`` and nothing is written), then
    ``_write_evaluation`` persists the ``{run_id}.metrics.json`` record. Kept apart from
    save_trained_run because a run is evaluated AFTER it is saved and materialised; the
    manifest and the untrained toy save path stay untouched.
    """
    logger.info("record_run_evaluation: run %s under %s", run_id, record_dir)
    return (
        make_metrics(val_rmse, test_rmse)
        .fmap_err(_invalid_metrics)
        .fmap(lambda m: EvaluationRecord(run_id, m.val_rmse, m.test_rmse))
        .and_then(lambda record: _write_evaluation(record_dir, record))
    )


def _invalid_split(error: InvalidHoldoutSpec) -> ErrorInfo:
    """Fold a domain holdout-split rejection into a coded boundary cause.

    The shell was handed fractions that do not form a valid split (out of range, not
    summing to one, or a negative seed); surface it as ``INVALID_SPLIT`` with the
    offending values preserved, so a nonsensical split is refused instead of persisted.
    """
    return ErrorInfo(code="INVALID_SPLIT", message=f"split rejected: {error!r}")


@safe(OSError, fmap_error(lambda cause: cause, code="SPLIT_WRITE_FAILED"))
def _write_split(record_dir: Path, record: SplitRecord) -> SplitRecord:
    """Write the split sidecar at the @safe I/O boundary; return the record.

    Separates the fallible write (OSError -> ``SPLIT_WRITE_FAILED``) from the pure split
    certification, so record_run_split chains the two on the railway with no nested
    ``@safe`` -- the same shape as record_run_evaluation.
    """
    write_split_record(record_dir, record)
    return record


def record_run_split(
    record_dir: Path,
    run_id: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Result[SplitRecord, ErrorInfo]:
    """Certify a run's train/val/test split and persist it as the split sidecar.

    The split write adapter: the notebook chooses the fractions + seed when it divides
    the data by hand and hands them here. The domain make_holdout_spec gate certifies
    them (an invalid split folds to ``INVALID_SPLIT`` and nothing is written), then
    ``_write_split`` persists the ``{run_id}.split.json`` record. Kept apart from
    save_trained_run because the untrained toy does no splitting; the manifest stays
    untouched.
    """
    logger.info("record_run_split: run %s under %s", run_id, record_dir)
    return (
        make_holdout_spec(train_fraction, val_fraction, test_fraction, seed)
        .fmap_err(_invalid_split)
        .fmap(
            lambda s: SplitRecord(
                run_id, s.train_fraction, s.val_fraction, s.test_fraction, s.seed
            )
        )
        .and_then(lambda record: _write_split(record_dir, record))
    )


def _invalid_provenance(error: InvalidDatasetProvenance) -> ErrorInfo:
    """Fold a domain provenance rejection into a coded boundary cause.

    The shell was handed data provenance that does not certify (blank id, no features, a
    target that is also a feature, or no rows); surface it as ``INVALID_PROVENANCE``,
    the offending values preserved, so unusable provenance is refused, not persisted.
    """
    return ErrorInfo(
        code="INVALID_PROVENANCE", message=f"provenance rejected: {error!r}"
    )


@safe(OSError, fmap_error(lambda cause: cause, code="PROVENANCE_WRITE_FAILED"))
def _write_provenance(record_dir: Path, record: ProvenanceRecord) -> ProvenanceRecord:
    """Write the provenance sidecar at the @safe I/O boundary; return the record.

    Separates the fallible write (OSError -> ``PROVENANCE_WRITE_FAILED``) from the pure
    provenance certification, so record_run_provenance chains the two on the railway
    with no nested ``@safe`` -- the same shape as record_run_split.
    """
    write_provenance_record(record_dir, record)
    return record


def record_run_provenance(
    record_dir: Path,
    run_id: str,
    dataset_id: str,
    feature_columns: tuple[str, ...],
    target_column: str,
    n_rows: int,
) -> Result[ProvenanceRecord, ErrorInfo]:
    """Certify a run's training-data provenance and persist it as the sidecar.

    The provenance write adapter: the notebook knows which dataset, columns, and row
    count it trained on and hands them here. The domain make_dataset_provenance gate
    certifies them (unusable provenance folds to ``INVALID_PROVENANCE`` and nothing is
    written), then ``_write_provenance`` persists the ``{run_id}.provenance.json``
    record. Kept apart from save_trained_run because the untrained toy sources no data;
    the manifest stays untouched.
    """
    logger.info("record_run_provenance: run %s under %s", run_id, record_dir)
    return (
        make_dataset_provenance(dataset_id, feature_columns, target_column, n_rows)
        .fmap_err(_invalid_provenance)
        .fmap(
            lambda p: ProvenanceRecord(
                run_id, p.dataset_id, p.feature_columns, p.target_column, p.n_rows
            )
        )
        .and_then(lambda record: _write_provenance(record_dir, record))
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
    pickled class -- from the ckpt the hydrated aggregate points at. Accepts a bare
    ``state_dict`` (what save_trained_run writes) OR a bundled Lightning checkpoint
    (``trainer.save_checkpoint`` nests weights under a ``state_dict`` key), so a
    resumable checkpoint reloads too. ``@safe`` folds a missing or unreadable ckpt
    into ``MODEL_MATERIALIZE_FAILED``.
    """
    logger.info("materialize run %s from %s", run.run_id, run.checkpoint.location)
    model = factory()
    checkpoint = torch.load(run.checkpoint.location, weights_only=True)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
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
    identity: ModelIdentity,
    manifest_dir: Path,
    run: TrainedRun,
    model: lightning.LightningModule,
) -> Result[lightning.LightningModule, ErrorInfo]:
    """Verify a rebuilt model against its manifest, returning the model if it matches.

    Reads the recorded fingerprint and declared identity from the sidecar (reusing the
    @safe manifest read) and runs both pure domain guards: verify_fingerprint on the
    live model's recomputed structural_fingerprint (shape drift), then verify_identity
    on the injected model's declared ``identity`` (name/version drift). Either drift
    folds to ``MODEL_IDENTITY_MISMATCH``; only a model clearing both passes through.
    Sequential and_then over the @safe read (no nested @safe), like load_trained_run.
    """
    got = structural_fingerprint(model.state_dict())
    return _read_manifest(manifest_dir, run.run_id).and_then(
        lambda manifest: (
            verify_fingerprint(run.run_id, manifest.fingerprint, got)
            .and_then(
                lambda _: verify_identity(
                    run.run_id, manifest.model_name, manifest.model_version, identity
                )
            )
            .fmap_err(_reload_mismatch)
            .fmap(lambda _: model)
        )
    )


def materialize_model(
    factory: Callable[[], lightning.LightningModule],
    identity: ModelIdentity,
    manifest_dir: Path,
    run: TrainedRun,
) -> Result[lightning.LightningModule, ErrorInfo]:
    """Rebuild ``run``'s LIVE nn on demand, guarded by its recorded identity.

    The torch-bearing counterpart of load_trained_run: the metadata aggregate points at
    a checkpoint, and this turns it back into a usable model, so the live nn never has
    to cross into domain/application. ``factory`` and its declared ``identity`` are the
    injected model seam (the composition root hands both, per fork 1A); ``manifest_dir``
    joins so ``partial`` yields the design's ``TrainedRun -> Result[LightningModule]``
    shape. Builds + loads the weights (_load_live_model), then verifies the rebuilt
    model against the manifest (_guard_reload): a load failure is
    ``MODEL_MATERIALIZE_FAILED``, a structural OR declared-identity drift is
    ``MODEL_IDENTITY_MISMATCH``.
    """
    return _load_live_model(factory, run).and_then(
        lambda model: _guard_reload(identity, manifest_dir, run, model)
    )
