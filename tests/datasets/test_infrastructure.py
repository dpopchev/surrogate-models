"""Tests for the datasets infrastructure -- the imperative shell.

factory_datasetid mints identity from a UUID, so it is nondeterministic and lives
in the shell, never the domain. save_dataset is the path-first persistence port:
it writes a certified Dataset to path/{dataset_id}.parquet and folds any OSError
onto the failure rail as a DATASET_SAVE_FAILED cause. find_dataset_frame is the
path-first read port: it reads path/{dataset_id}.parquet back into a bare frame
(the read model) and folds any OSError as a DATASET_READ_FAILED cause. One assert
per test.

All ``.dat`` fixtures here are MINIMAL synthetic batches (generic ``f*`` columns, a
``Pc =`` comment -- the one parameter the reader actually extracts) exercising the
reader's contract; never the real dataset's schema or scale (see the tdd rule).
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from surrogate_models.datasets.domain import (
    Dataset,
    DatasetID,
    make_dataset,
    make_datasetid,
)
from surrogate_models.datasets.infrastructure import (
    factory_datasetid,
    find_dataset_frame,
    read_neutron_stars_frame,
    save_dataset,
)
from surrogate_models.railway_adts import Ok

ONE_BATCH = """\
# Pc = 5e-05
f1 f2 f3
1.0 2.0 3.0
4.0 5.0 6.0
"""

TWO_BATCH = """\
# Pc = 5e-05
f1 f2 f3
1.0 2.0 3.0
4.0 5.0 6.0
# Pc = 7e-05
f1 f2 f3
7.0 8.0 9.0
10.0 11.0 12.0
"""

# A comment-only batch: its `#` comment is immediately followed by the next batch's
# comment, with no header/data rows. Feeding pd.read_csv an empty body raises
# EmptyDataError, so the reader must SKIP such a batch and parse the rest.
EMPTY_BATCH_BETWEEN = """\
# Pc = 5e-05
f1 f2 f3
1.0 2.0 3.0
# tag = empty
# Pc = 7e-05
f1 f2 f3
7.0 8.0 9.0
"""

# A batch that emitted its header but ZERO data rows. pd.read_csv on a header-only
# body yields a 0-row, all-object frame; concatenated with the populated (float)
# batches it upcasts the shared columns to object -- which make_dataset rejects
# (DATASET_MISSING_SCHEMA). The reader must heal the dtype back to numeric.
HEADER_ONLY_BATCH_MIX = """\
# Pc = 5e-05
f1 f2 f3
1.0 2.0 3.0
4.0 5.0 6.0
# Pc = 7e-05
f1 f2 f3
"""


def test_factory_datasetid_is_eight_characters() -> None:
    assert len(factory_datasetid()) == 8


def test_factory_datasetid_is_lowercase_hex() -> None:
    assert all(char in "0123456789abcdef" for char in factory_datasetid())


def test_factory_datasetid_mints_distinct_ids() -> None:
    assert factory_datasetid() != factory_datasetid()


def test_factory_datasetid_output_certifies_via_make_datasetid() -> None:
    assert make_datasetid(factory_datasetid()).is_ok() is True


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


def test_read_neutron_stars_frame_appends_pc_init_from_comment(tmp_path: Path) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(ONE_BATCH)
    frame = read_neutron_stars_frame(source).unwrap()
    assert (frame["pc_init"] == 5e-05).all()


def test_read_neutron_stars_frame_concatenates_batches_with_per_batch_pc_init(
    tmp_path: Path,
) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(TWO_BATCH)
    frame = read_neutron_stars_frame(source).unwrap()
    assert list(frame["pc_init"]) == [5e-05, 5e-05, 7e-05, 7e-05]


def test_read_neutron_stars_frame_yields_certifiable_frame(tmp_path: Path) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(ONE_BATCH)
    frame = read_neutron_stars_frame(source).unwrap()
    assert make_dataset(DatasetID("nstars00"), frame).is_ok()


def test_read_neutron_stars_frame_wraps_missing_file_as_error(tmp_path: Path) -> None:
    missing = read_neutron_stars_frame(tmp_path / "absent.dat")
    assert missing.unwrap_err().code == "NEUTRON_STARS_READ_FAILED"


def test_read_neutron_stars_frame_wraps_malformed_batch_as_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text("# tag = 1\nf1 f2\n1.0 2.0\n")
    failure = read_neutron_stars_frame(source)
    assert failure.unwrap_err().code == "NEUTRON_STARS_READ_FAILED"


def test_read_neutron_stars_frame_skips_empty_comment_only_batch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(EMPTY_BATCH_BETWEEN)
    frame = read_neutron_stars_frame(source).unwrap()
    assert list(frame["pc_init"]) == [5e-05, 7e-05]


def test_read_neutron_stars_frame_warns_on_skipped_empty_batch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(EMPTY_BATCH_BETWEEN)
    with caplog.at_level(logging.WARNING):
        read_neutron_stars_frame(source)
    assert "tag = empty" in caplog.text


def test_read_neutron_stars_frame_heals_object_dtype_from_header_only_batch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(HEADER_ONLY_BATCH_MIX)
    frame = read_neutron_stars_frame(source).unwrap()
    assert all(str(dtype) != "object" for dtype in frame.dtypes)


def test_read_neutron_stars_frame_logs_ingest_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(ONE_BATCH)
    with caplog.at_level(logging.INFO):
        read_neutron_stars_frame(source)
    assert "ingested" in caplog.text
