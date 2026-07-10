"""Tests for the mlmodels src infrastructure -- the save_trained_run adapter.

The thin training slice: save_trained_run builds a minimal real SurrogateRegressor
FROM a run's certified config and persists its UNTRAINED weights as ``{run_id}.ckpt``
under the checkpoint dir, returning ``Ok(Checkpoint)`` -- so the composition seat's
happy path is reachable with a real torch artifact on disk. No training happens yet
(no fit, no data); the multi-epoch training proof stays in test_e2e against a
test-owned adapter, and the EXPAND slice grows this adapter to actually fit over real
data.

The manifest slice proves the ``{run_id}.json`` sidecar round-trips: a RunManifest
written by write_run_manifest reads back equal via read_run_manifest, over a minimal
synthetic manifest (generic run id and model identity -- never a real dataset's schema
or scale). One assert per test.
"""

from pathlib import Path

import torch
from torch import nn

from surrogate_models.mlmodels.domain import (
    Checkpoint,
    ConfiguredRun,
    RunID,
    RunSummaryDTO,
    TrainedRun,
    TrainingConfig,
    complete_training,
    configure_run,
)
from surrogate_models.mlmodels.infrastructure import (
    RunManifest,
    SurrogateRegressor,
    find_run_summary,
    load_trained_run,
    materialize_model,
    read_run_manifest,
    save_trained_run,
    structural_fingerprint,
    write_run_manifest,
)


def _configured_run() -> ConfiguredRun:
    """A certified ConfiguredRun, built directly for the adapter tests."""
    return configure_run(RunID("smoke"), TrainingConfig(1, 0.01, 2, "sgd"))


def test_save_trained_run_writes_the_named_checkpoint(tmp_path: Path) -> None:
    save_trained_run(tmp_path, _configured_run())
    assert (tmp_path / "smoke.ckpt").exists() is True


def test_save_trained_run_returns_the_written_checkpoint(tmp_path: Path) -> None:
    result = save_trained_run(tmp_path, _configured_run())
    assert result.unwrap() == Checkpoint(str(tmp_path / "smoke.ckpt"))


def test_run_manifest_round_trips_through_its_sidecar(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="demo",
        model_name="regressor",
        model_version="1.0.0",
        max_epochs=3,
        learning_rate=0.01,
        batch_size=2,
        optimizer="sgd",
        fingerprint="fp0",
    )
    write_run_manifest(tmp_path, manifest)
    assert read_run_manifest(tmp_path, "demo") == manifest


def test_save_trained_run_writes_a_manifest_naming_the_model(tmp_path: Path) -> None:
    save_trained_run(tmp_path, _configured_run())
    assert read_run_manifest(tmp_path, "smoke").model_name == "surrogate-regressor"


def test_save_trained_run_records_the_model_fingerprint(tmp_path: Path) -> None:
    run = _configured_run()
    save_trained_run(tmp_path, run)
    fingerprint = structural_fingerprint(SurrogateRegressor(run.config).state_dict())
    assert read_run_manifest(tmp_path, "smoke").fingerprint == fingerprint


def test_find_run_summary_projects_the_manifest_into_a_summary(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        RunManifest(
            run_id="demo",
            model_name="regressor",
            model_version="1.0.0",
            max_epochs=3,
            learning_rate=0.01,
            batch_size=2,
            optimizer="sgd",
            fingerprint="fp0",
        ),
    )
    result = find_run_summary(tmp_path, "demo")
    assert result.unwrap() == RunSummaryDTO(
        run_id="demo", model_name="regressor", model_version="1.0.0"
    )


def test_load_trained_run_hydrates_the_persisted_run(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        RunManifest(
            run_id="demo",
            model_name="regressor",
            model_version="1.0.0",
            max_epochs=3,
            learning_rate=0.01,
            batch_size=2,
            optimizer="sgd",
            fingerprint="fp0",
        ),
    )
    result = load_trained_run(tmp_path, "demo")
    assert result.unwrap() == TrainedRun(
        RunID("demo"),
        TrainingConfig(3, 0.01, 2, "sgd"),
        Checkpoint(str(tmp_path / "demo.ckpt")),
    )


def test_load_trained_run_missing_manifest_reports_read_failed(tmp_path: Path) -> None:
    result = load_trained_run(tmp_path, "absent")
    assert result.unwrap_err().code == "RUN_LOAD_FAILED"


def test_structural_fingerprint_differs_when_a_layer_shape_changes() -> None:
    narrow = nn.Linear(1, 1).state_dict()
    wide = nn.Linear(2, 1).state_dict()
    assert structural_fingerprint(narrow) != structural_fingerprint(wide)


def test_load_run_uncertifiable_manifest_is_corrupt(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        RunManifest(
            run_id="demo",
            model_name="regressor",
            model_version="1.0.0",
            max_epochs=3,
            learning_rate=0.01,
            batch_size=2,
            optimizer="rmsprop",
            fingerprint="fp0",
        ),
    )
    result = load_trained_run(tmp_path, "demo")
    assert result.unwrap_err().code == "RUN_LOAD_CORRUPT"


def test_materialize_model_loads_the_persisted_weights(tmp_path: Path) -> None:
    config = TrainingConfig(1, 0.01, 2, "sgd")
    reference = SurrogateRegressor(config)
    torch.save(reference.state_dict(), tmp_path / "smoke.ckpt")
    write_run_manifest(
        tmp_path,
        RunManifest(
            run_id="smoke",
            model_name="surrogate-regressor",
            model_version="0.1.0",
            max_epochs=1,
            learning_rate=0.01,
            batch_size=2,
            optimizer="sgd",
            fingerprint=structural_fingerprint(reference.state_dict()),
        ),
    )
    run = complete_training(
        configure_run(RunID("smoke"), config),
        Checkpoint(str(tmp_path / "smoke.ckpt")),
    )
    result = materialize_model(lambda: SurrogateRegressor(config), tmp_path, run)
    model = result.unwrap()
    assert torch.equal(
        model.state_dict()["linear.weight"], reference.state_dict()["linear.weight"]
    )


def test_materialize_model_rejects_a_fingerprint_mismatch(tmp_path: Path) -> None:
    config = TrainingConfig(1, 0.01, 2, "sgd")
    torch.save(SurrogateRegressor(config).state_dict(), tmp_path / "smoke.ckpt")
    write_run_manifest(
        tmp_path,
        RunManifest(
            run_id="smoke",
            model_name="surrogate-regressor",
            model_version="0.1.0",
            max_epochs=1,
            learning_rate=0.01,
            batch_size=2,
            optimizer="sgd",
            fingerprint="tampered",
        ),
    )
    run = complete_training(
        configure_run(RunID("smoke"), config),
        Checkpoint(str(tmp_path / "smoke.ckpt")),
    )
    result = materialize_model(lambda: SurrogateRegressor(config), tmp_path, run)
    assert result.unwrap_err().code == "MODEL_IDENTITY_MISMATCH"
