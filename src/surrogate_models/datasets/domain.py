"""Datasets domain -- the pure functional core of the datasets bounded context.

Holds the Dataset aggregate, its identity (DatasetID), and the validation error
returned when a frame's schema is not explicit. No I/O, no uuid, no raise:
identity is supplied from the shell (see infrastructure.factory_datasetid) and
failures travel the railway as Result values.

The smart constructor make_dataset is added under TDD -- it is the only way to
build a certified Dataset, so every Dataset in the system has passed schema
certification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType

import pandas as pd

from surrogate_models.railway_adts import Err, Ok, Result

DatasetID = NewType("DatasetID", str)

_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def make_datasetid(dataset_id: str) -> Result[DatasetID, InvalidDatasetID]:
    """Certify a candidate ``dataset_id`` string into a DatasetID, or reject it.

    The SOLE way to build a DatasetID -- so every DatasetID in the system has
    passed this check. Pure and total: validate + wrap, no minting. A valid id is a
    non-empty, filesystem-safe stem (``^[A-Za-z0-9._-]+$``), because the id becomes
    the ``{id}.parquet`` filename downstream; anything else (blank, whitespace, a
    path separator) returns ``Err(InvalidDatasetID)`` carrying the offending string.

    The mint-if-absent fallback is deliberately NOT here: a read never mints, and
    generating a fresh id is nondeterministic (shell) work. The write handler owns
    that policy -- ``make_datasetid(cmd.dataset_id or new_id())``.
    """
    if _DATASET_ID_PATTERN.match(dataset_id):
        return Ok(DatasetID(dataset_id))
    return Err(InvalidDatasetID(dataset_id))


@dataclass(frozen=True, slots=True)
class InvalidDatasetID:
    """Validation failure: a candidate id is not a valid DatasetID.

    Carries the offending ``dataset_id`` string -- the value that failed
    certification -- per the boundary rule that a pure validation failure holds the
    offending value. Plain value equality is safe here (it wraps a ``str``, not a
    DataFrame), so the default ``eq`` stands.
    """

    dataset_id: str


@dataclass(frozen=True, slots=True, eq=False)
class DatasetMissingSchema:
    """Validation failure: the frame lacks a clear, explicit column schema.

    Carries the offending frame -- the value that failed certification -- per the
    boundary rule that a pure validation failure holds the offending value.
    ``eq=False`` keeps identity equality so comparing errors never triggers a
    DataFrame's ambiguous elementwise ``==``.
    """

    frame: pd.DataFrame


@dataclass(frozen=True, slots=True, eq=False)
class Dataset:
    """The datasets aggregate: a schema-certified frame under a stable identity.

    ``eq=False`` gives identity equality (an aggregate is identified by its
    ``dataset_id``, not by frame contents) and sidesteps a DataFrame's ambiguous
    elementwise ``==`` inside dataclass equality.
    """

    dataset_id: DatasetID
    frame: pd.DataFrame


def _has_clear_schema(frame: pd.DataFrame) -> bool:
    """Report whether every column carries an explicit (non-object) dtype.

    A clear schema needs at least one column, and none may be ``object`` -- in
    pandas 3.0 an ``object`` column is the ambiguous catch-all (plain strings are
    ``str`` dtype), so its presence means a column's type was never made explicit.
    Row count is irrelevant: a typed frame with zero rows still has a schema.
    """
    has_columns = len(frame.columns) > 0
    all_columns_explicit = all(str(dtype) != "object" for dtype in frame.dtypes)
    return has_columns and all_columns_explicit


def make_dataset(
    dataset_id: DatasetID, frame: pd.DataFrame
) -> Result[Dataset, DatasetMissingSchema]:
    """Certify ``frame`` under ``dataset_id`` into a Dataset, or reject it.

    The sole way to build a Dataset: returns ``Ok(Dataset)`` when the frame's
    schema is clear (see :func:`_has_clear_schema`), else ``Err`` carrying a
    ``DatasetMissingSchema`` that holds the offending frame. Pure and total -- no
    I/O, no raise; the caller composes the Result on the railway.
    """
    if _has_clear_schema(frame):
        return Ok(Dataset(dataset_id, frame))
    return Err(DatasetMissingSchema(frame))
