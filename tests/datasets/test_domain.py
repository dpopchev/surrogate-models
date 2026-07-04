"""Tests for the datasets domain -- the make_dataset smart constructor.

Schema rule under test: a frame's schema is "clear" iff it has at least one
column AND no column carries the ``object`` dtype (in pandas 3.0 ``object`` marks
a genuinely ambiguous, non-explicitly-typed column). Otherwise make_dataset
returns Err(DatasetMissingSchema). One assert per test.
"""

import pandas as pd

from surrogate_models.datasets.domain import (
    Dataset,
    DatasetID,
    DatasetMissingSchema,
    InvalidDatasetID,
    make_dataset,
    make_datasetid,
)

DATASET_ID = DatasetID("abc12345")


def _typed_frame() -> pd.DataFrame:
    """A frame whose columns all have explicit (non-object) dtypes."""
    return pd.DataFrame({"i": [1, 2], "s": ["a", "b"], "f": [1.0, 2.0]})


def _object_frame() -> pd.DataFrame:
    """A frame with one ambiguous object-dtype column."""
    return pd.DataFrame({"i": [1, 2], "bad": pd.Series([1, "x"], dtype="object")})


# --- acceptance: a schema-clear frame ---


def test_make_dataset_ok_on_typed_frame() -> None:
    assert make_dataset(DATASET_ID, _typed_frame()).is_ok() is True


def test_make_dataset_ok_returns_dataset() -> None:
    assert isinstance(make_dataset(DATASET_ID, _typed_frame()).unwrap(), Dataset)


def test_make_dataset_carries_the_dataset_id() -> None:
    assert make_dataset(DATASET_ID, _typed_frame()).unwrap().dataset_id == DATASET_ID


def test_make_dataset_carries_the_frame() -> None:
    frame = _typed_frame()
    assert make_dataset(DATASET_ID, frame).unwrap().frame is frame


def test_make_dataset_ok_on_typed_frame_without_rows() -> None:
    frame = pd.DataFrame({"i": pd.Series([], dtype="int64")})
    assert make_dataset(DATASET_ID, frame).is_ok() is True


# --- rejection: an unclear schema ---


def test_make_dataset_err_on_object_dtype_column() -> None:
    assert make_dataset(DATASET_ID, _object_frame()).is_err() is True


def test_make_dataset_err_is_missing_schema() -> None:
    error = make_dataset(DATASET_ID, _object_frame()).unwrap_err()
    assert isinstance(error, DatasetMissingSchema)


def test_make_dataset_err_carries_the_offending_frame() -> None:
    frame = _object_frame()
    assert make_dataset(DATASET_ID, frame).unwrap_err().frame is frame


def test_make_dataset_err_on_zero_column_frame() -> None:
    assert make_dataset(DATASET_ID, pd.DataFrame()).is_err() is True


# --- make_datasetid: the sole str -> DatasetID smart constructor ---
#
# Validity rule: a non-empty, filesystem-safe stem (``^[A-Za-z0-9._-]+$``), since
# the id becomes the ``{id}.parquet`` filename. Anything else -> Err(InvalidDatasetID).


def test_make_datasetid_ok_on_valid_id() -> None:
    assert make_datasetid("neutron-stars").is_ok() is True


def test_make_datasetid_ok_returns_the_id() -> None:
    assert make_datasetid("neutron-stars").unwrap() == DatasetID("neutron-stars")


def test_make_datasetid_ok_on_hex_mint_output() -> None:
    assert make_datasetid("abc12345").is_ok() is True


def test_make_datasetid_err_on_empty_string() -> None:
    assert make_datasetid("").is_err() is True


def test_make_datasetid_err_on_whitespace() -> None:
    assert make_datasetid("  ").is_err() is True


def test_make_datasetid_err_on_path_separator() -> None:
    assert make_datasetid("nested/id").is_err() is True


def test_make_datasetid_err_is_invalid_datasetid() -> None:
    assert isinstance(make_datasetid("").unwrap_err(), InvalidDatasetID)


def test_make_datasetid_err_carries_the_offending_string() -> None:
    assert make_datasetid("bad id").unwrap_err().dataset_id == "bad id"
