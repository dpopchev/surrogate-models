"""Tests for the datasets infrastructure -- the imperative shell.

factory_datasetid mints identity from a UUID, so it is nondeterministic and lives
in the shell, never the domain. save_dataset is the path-first persistence port:
it writes a certified Dataset to path/{dataset_id}.parquet and folds any OSError
onto the failure rail as a DATASET_SAVE_FAILED cause. find_dataset_frame is the
path-first read port: it reads path/{dataset_id}.parquet back into a bare frame
(the read model) and folds any OSError as a DATASET_READ_FAILED cause. One assert
per test.
"""

from pathlib import Path

import pandas as pd

from surrogate_models.datasets.domain import Dataset, DatasetID
from surrogate_models.datasets.infrastructure import (
    factory_datasetid,
    find_dataset_frame,
    save_dataset,
)
from surrogate_models.railway_adts import Ok


def test_factory_datasetid_is_eight_characters() -> None:
    assert len(factory_datasetid()) == 8


def test_factory_datasetid_is_lowercase_hex() -> None:
    assert all(char in "0123456789abcdef" for char in factory_datasetid())


def test_factory_datasetid_mints_distinct_ids() -> None:
    assert factory_datasetid() != factory_datasetid()


def test_save_dataset_returns_ok_on_success(tmp_path: Path) -> None:
    dataset = Dataset(DatasetID("abc12345"), pd.DataFrame({"x": [1, 2, 3]}))
    assert save_dataset(tmp_path, dataset) == Ok(None)


def test_save_dataset_writes_a_readable_parquet_file(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})
    save_dataset(tmp_path, Dataset(DatasetID("abc12345"), frame))
    assert pd.read_parquet(tmp_path / "abc12345.parquet").equals(frame)


def test_save_dataset_wraps_write_failure_as_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file standing where a directory is needed")
    dataset = Dataset(DatasetID("abc12345"), pd.DataFrame({"x": [1]}))
    assert save_dataset(blocker, dataset).unwrap_err().code == "DATASET_SAVE_FAILED"


def test_find_dataset_frame_reads_back_the_written_frame(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})
    frame.to_parquet(tmp_path / "abc12345.parquet")
    assert find_dataset_frame(tmp_path, DatasetID("abc12345")).unwrap().equals(frame)


def test_find_dataset_frame_wraps_missing_file_as_error(tmp_path: Path) -> None:
    missing = find_dataset_frame(tmp_path, DatasetID("missing0"))
    assert missing.unwrap_err().code == "DATASET_READ_FAILED"
