"""Tests for the datasets infrastructure -- the imperative shell.

factory_datasetid mints identity from a UUID, so it is nondeterministic and lives
in the shell, never the domain. save_dataset is the path-first persistence port:
it writes a certified Dataset to path/{dataset_id}.parquet and folds any OSError
onto the failure rail as a DATASET_SAVE_FAILED cause. find_dataset_frame is the
path-first read port: it reads path/{dataset_id}.parquet back into a bare frame
(the read model) and folds any OSError as a DATASET_READ_FAILED cause. One assert
per test.
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
# x1 = 0.005, x2 = 1000.0, beta = 0.4, Lambda = 0.5, Pc = 5e-05, choice_theory = 31
rhoc Pc M
5.89e14 5.0e-05 0.56
6.06e14 5.5e-05 0.59
"""

TWO_BATCH = """\
# x1 = 0.005, x2 = 1000.0, beta = 0.4, Lambda = 0.5, Pc = 5e-05, choice_theory = 31
rhoc Pc M
5.89e14 5.0e-05 0.56
6.06e14 5.5e-05 0.59
# x1 = 0.005, x2 = 1000.0, beta = 6.4, Lambda = 0.7, Pc = 7e-05, choice_theory = 31
rhoc Pc M
7.10e14 7.0e-05 0.71
7.50e14 7.5e-05 0.75
"""

# A comment-only batch (a run that produced zero stars: its `#` parameter comment is
# immediately followed by the next batch's comment, with no header/data rows). The
# real neutron-stars.dat has 36 of these; each fed pd.read_csv an empty string ->
# EmptyDataError. The reader must SKIP such batches and parse the rest.
EMPTY_BATCH_BETWEEN = """\
# x1 = 0.005, x2 = 1000.0, beta = 0.4, Lambda = 0.5, Pc = 5e-05, choice_theory = 31
rhoc Pc M
5.89e14 5.0e-05 0.56
# x1 = 0.005, x2 = 1000.0, beta = 50.4, Lambda = 0.5, Pc = 5e-05, choice_theory = 31
# x1 = 0.005, x2 = 1000.0, beta = 6.4, Lambda = 0.7, Pc = 7e-05, choice_theory = 31
rhoc Pc M
7.10e14 7.0e-05 0.71
"""

# A batch that emitted its header but ZERO data rows (a run that printed the column
# header yet produced no stars). pd.read_csv on a header-only body yields a 0-row,
# all-object frame; concatenated with the populated (float) batches it upcasts the
# shared columns to object -- which make_dataset rejects (DATASET_MISSING_SCHEMA).
# The real neutron-stars.dat has 203 such batches. The reader must heal the dtype.
HEADER_ONLY_BATCH_MIX = """\
# x1 = 0.005, x2 = 1000.0, beta = 0.4, Lambda = 0.5, Pc = 5e-05, choice_theory = 31
rhoc Pc M
5.89e14 5.0e-05 0.56
6.06e14 5.5e-05 0.59
# x1 = 0.005, x2 = 1000.0, beta = 6.4, Lambda = 0.7, Pc = 7e-05, choice_theory = 31
rhoc Pc M
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
    source.write_text("# x1 = 0.005, beta = 0.4\nrhoc M\n5.89e14 0.56\n")
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
    assert "beta = 50.4" in caplog.text


def test_read_neutron_stars_frame_heals_object_dtype_from_header_only_batch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "neutron-stars.dat"
    source.write_text(HEADER_ONLY_BATCH_MIX)
    frame = read_neutron_stars_frame(source).unwrap()
    assert all(str(dtype) != "object" for dtype in frame.dtypes)
