"""Tests for the datasets application -- the make/get dataset handlers.

handle_make_dataset (write) mints an id, certifies the frame via the domain
make_dataset, and persists through an injected SaveDatasetFn -- returning
Ok(DatasetID) on success, or Err(FailMadeDataset) folding either the schema
failure or the save failure into one ErrorInfo cause. handle_get_dataset (read)
binds an injected FindDatasetFn to return the stored frame as the read model DTO,
or Err(FailGetDataset) wrapping the read ErrorInfo -- never the Dataset aggregate.
Dependencies (new_id, save, find) are hand-written stubs; frames are hand-written.
One assert per test.
"""

import pandas as pd

from surrogate_models.datasets.application import (
    FailGetDataset,
    FailMadeDataset,
    FindDatasetFn,
    GetDataset,
    MakeDataset,
    SaveDatasetFn,
    handle_get_dataset,
    handle_make_dataset,
)
from surrogate_models.datasets.domain import Dataset, DatasetID
from surrogate_models.railway_adts import Err, ErrorInfo, Ok, Result

FIXED_ID = DatasetID("feed0001")


def _fixed_id() -> DatasetID:
    """Deterministic id source: always mints the same DatasetID."""
    return FIXED_ID


def _typed_frame() -> pd.DataFrame:
    """A frame whose columns all have explicit (non-object) dtypes -- certifies."""
    return pd.DataFrame({"i": [1, 2], "s": ["a", "b"], "f": [1.0, 2.0]})


def _object_frame() -> pd.DataFrame:
    """A frame with one ambiguous object-dtype column -- fails certification."""
    return pd.DataFrame({"i": [1, 2], "bad": pd.Series([1, "x"], dtype="object")})


def _save_ok(sink: list[Dataset]) -> SaveDatasetFn:
    """A save port that records the Dataset it received and reports success."""

    def save(dataset: Dataset) -> Result[None, ErrorInfo]:
        sink.append(dataset)
        return Ok(None)

    return save


def _save_err(error: ErrorInfo) -> SaveDatasetFn:
    """A save port that fails with a fixed ErrorInfo (a persistence boundary error)."""

    def save(dataset: Dataset) -> Result[None, ErrorInfo]:
        return Err(error)

    return save


# --- Ok rail: mint -> certify -> save ---


def test_handle_make_dataset_returns_minted_id() -> None:
    result = handle_make_dataset(
        save=_save_ok([]), new_id=_fixed_id, cmd=MakeDataset(_typed_frame())
    )
    assert result.unwrap() == FIXED_ID


def test_handle_make_dataset_saves_certified_dataset_id() -> None:
    saved: list[Dataset] = []
    handle_make_dataset(
        save=_save_ok(saved), new_id=_fixed_id, cmd=MakeDataset(_typed_frame())
    )
    assert saved[0].dataset_id == FIXED_ID


def test_handle_make_dataset_saves_certified_frame() -> None:
    saved: list[Dataset] = []
    frame = _typed_frame()
    handle_make_dataset(save=_save_ok(saved), new_id=_fixed_id, cmd=MakeDataset(frame))
    assert saved[0].frame is frame


# --- schema-fail rail: domain rejects the frame, save is never reached ---


def test_handle_make_dataset_err_on_object_frame() -> None:
    result = handle_make_dataset(
        save=_save_ok([]), new_id=_fixed_id, cmd=MakeDataset(_object_frame())
    )
    assert isinstance(result.unwrap_err(), FailMadeDataset)


def test_handle_make_dataset_schema_fail_cause_code() -> None:
    result = handle_make_dataset(
        save=_save_ok([]), new_id=_fixed_id, cmd=MakeDataset(_object_frame())
    )
    assert result.unwrap_err().cause.code == "DATASET_MISSING_SCHEMA"


def test_handle_make_dataset_schema_fail_does_not_save() -> None:
    saved: list[Dataset] = []
    handle_make_dataset(
        save=_save_ok(saved), new_id=_fixed_id, cmd=MakeDataset(_object_frame())
    )
    assert saved == []


# --- save-fail rail: the persistence ErrorInfo is wrapped verbatim as the cause ---


def test_handle_make_dataset_save_fail_wraps_errorinfo() -> None:
    cause = ErrorInfo(code="E_IO", message="disk full")
    result = handle_make_dataset(
        save=_save_err(cause), new_id=_fixed_id, cmd=MakeDataset(_typed_frame())
    )
    assert result.unwrap_err().cause == cause


# --- Queries: handle_get_dataset returns the stored frame as a read model ---


def _find_ok(frame: pd.DataFrame) -> FindDatasetFn:
    """A find port that reports the given frame for any id (a read hit)."""

    def find(dataset_id: DatasetID) -> Result[pd.DataFrame, ErrorInfo]:
        return Ok(frame)

    return find


def _find_recording(seen: list[DatasetID], frame: pd.DataFrame) -> FindDatasetFn:
    """A find port that records the id it was asked for, then reports the frame."""

    def find(dataset_id: DatasetID) -> Result[pd.DataFrame, ErrorInfo]:
        seen.append(dataset_id)
        return Ok(frame)

    return find


def _find_err(error: ErrorInfo) -> FindDatasetFn:
    """A find port that fails with a fixed ErrorInfo (a read boundary error)."""

    def find(dataset_id: DatasetID) -> Result[pd.DataFrame, ErrorInfo]:
        return Err(error)

    return find


def test_handle_get_dataset_returns_found_frame() -> None:
    frame = _typed_frame()
    result = handle_get_dataset(find=_find_ok(frame), query=GetDataset(FIXED_ID))
    assert result.unwrap() is frame


def test_handle_get_dataset_queries_by_requested_id() -> None:
    seen: list[DatasetID] = []
    handle_get_dataset(
        find=_find_recording(seen, _typed_frame()), query=GetDataset(FIXED_ID)
    )
    assert seen[0] == FIXED_ID


def test_handle_get_dataset_err_is_failgetdataset() -> None:
    result = handle_get_dataset(
        find=_find_err(ErrorInfo(code="DATASET_READ_FAILED", message="No such file")),
        query=GetDataset(FIXED_ID),
    )
    assert isinstance(result.unwrap_err(), FailGetDataset)


def test_handle_get_dataset_read_fail_wraps_errorinfo() -> None:
    cause = ErrorInfo(code="DATASET_READ_FAILED", message="No such file")
    result = handle_get_dataset(find=_find_err(cause), query=GetDataset(FIXED_ID))
    assert result.unwrap_err().cause == cause
