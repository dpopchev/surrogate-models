"""Datasets application -- CQRS handlers over the datasets domain.

Orchestrates the pure core via injected callables (DIP). Id generation is an
injected DatasetIdFn (mints a raw candidate string) and persistence an injected
SaveDatasetFn, both supplied by infrastructure at the composition root, so the
application certifies an edge id string via the domain make_datasetid and hands a
DatasetID to make_dataset without the core ever touching uuid or I/O. Command and
query handlers are added under TDD, kept as ordered CQRS sections.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from surrogate_models.datasets.domain import (
    Dataset,
    DatasetID,
    DatasetMissingSchema,
    InvalidDatasetID,
    make_dataset,
    make_datasetid,
)
from surrogate_models.railway_adts import ErrorInfo, Result

logger = logging.getLogger(__name__)

# --- (1) Commands ---

# Injected ports (DIP via callables): the shell supplies these at the composition
# root. DatasetIdFn mints a raw candidate id STRING (the domain make_datasetid wraps
# it -- the sole DatasetID constructor -- so the shell produces a primitive, never a
# domain type); the write side binds the aggregate PAIR SaveDatasetFn / LoadDatasetFn.
# SaveDatasetFn persists a certified Dataset (nothing on success, an ErrorInfo on
# failure); LoadDatasetFn HYDRATES the whole Dataset back from the store --
# re-certified on the way in (the trust boundary) -- so a command can load ->
# transition -> save the SAME aggregate. It returns domain objects, never a DTO.
DatasetIdFn = Callable[[], str]
SaveDatasetFn = Callable[[Dataset], Result[None, ErrorInfo]]
LoadDatasetFn = Callable[[DatasetID], Result[Dataset, ErrorInfo]]


@dataclass(frozen=True, slots=True, eq=False)
class MakeDataset:
    """Command: certify and persist ``frame`` as a Dataset.

    Carries only edge data -- the frame to certify and an optional requested
    ``dataset_id`` (a bare ``str``; empty means "mint one"). The handler wraps the
    id via the domain :func:`make_datasetid`. ``eq=False`` keeps identity equality
    so a command never triggers a DataFrame's ambiguous elementwise ``==``. The id
    source (for the mint fallback) and persistence are injected into the handler,
    not the command.
    """

    frame: pd.DataFrame
    dataset_id: str = ""


# The union of every datasets write command -- one member today, dispatched by
# match/case at the delivery edge. Grows as commands are added (no base class).
type DatasetCommand = MakeDataset


@dataclass(frozen=True, slots=True)
class FailMadeDataset:
    """Failure of handle_make_dataset, carrying its structured ``cause``.

    A single application error for both rails: a domain schema rejection and a
    save-boundary failure both fold into one ``ErrorInfo`` cause, so the failure
    crosses UP the ring as a serializable descriptor -- never the domain type or a
    lower-layer exception.
    """

    cause: ErrorInfo


def _fail_from_schema(_error: DatasetMissingSchema) -> FailMadeDataset:
    """Lift a domain schema rejection into the handler's own error.

    The offending frame stays in the domain error; the boundary reports a stable
    ``code`` plus message, never the domain type -- the railway analogue of
    ``raise FailMadeDataset from DatasetMissingSchema``. This lifter is also the
    boundary where the pure domain's schema decision becomes observable, so the
    rejection is logged here (the domain core stays free of logging).
    """
    logger.warning("dataset frame rejected: no clear column schema")
    return FailMadeDataset(
        ErrorInfo(
            code="DATASET_MISSING_SCHEMA",
            message="frame lacks a clear column schema",
        )
    )


def _fail_make_from_invalid_id(error: InvalidDatasetID) -> FailMadeDataset:
    """Lift a domain id-validation rejection into the write handler's own error.

    The offending id string stays in the domain error; the boundary reports a
    stable ``code`` plus message -- symmetric with the read side's
    :func:`_fail_get_from_invalid_id`. The rejection is logged here at the boundary
    (the domain core stays free of logging).
    """
    logger.warning(
        "dataset id rejected on write: %r is not a valid id", error.dataset_id
    )
    return FailMadeDataset(
        ErrorInfo(
            code="INVALID_DATASET_ID",
            message=f"invalid dataset id: {error.dataset_id!r}",
        )
    )


def handle_make_dataset(
    *, save: SaveDatasetFn, new_id: DatasetIdFn, cmd: MakeDataset
) -> Result[DatasetID, FailMadeDataset]:
    """Wrap the id, certify ``cmd.frame`` into a Dataset, and persist it.

    The id policy: a requested ``cmd.dataset_id`` wins; an empty one mints via the
    injected ``new_id`` -- ``cmd.dataset_id or new_id()`` -- and either way the
    candidate ``str`` is certified by the domain :func:`make_datasetid` (the sole
    ``DatasetID`` constructor). Returns ``Ok`` with that ``DatasetID`` once the frame
    certifies and the save succeeds. All three failure rails fold into
    ``FailMadeDataset``: an invalid id via :func:`_fail_make_from_invalid_id`, a
    schema rejection via :func:`_fail_from_schema`, a save ``ErrorInfo`` wrapped
    verbatim. Pure railway composition -- no branching on the ``Result``.
    """
    return (
        make_datasetid(cmd.dataset_id or new_id())
        .fmap_err(_fail_make_from_invalid_id)
        .and_then(
            lambda dataset_id: (
                make_dataset(dataset_id, cmd.frame)
                .fmap_err(_fail_from_schema)
                .and_then(lambda dataset: save(dataset).fmap_err(FailMadeDataset))
                .fmap(lambda _saved: dataset_id)
            )
        )
    )


# --- (2) Queries ---

# Read-side port (DIP via callables): FindDatasetFn returns the read model keyed by
# DatasetID -- a bare pandas DataFrame ready for projection -- BYPASSING the Dataset
# aggregate and its certification. FIND a view to show it (the read counterpart of
# LoadDatasetFn's LOAD-to-act-on-it). A bare frame is a legal query DTO at the I/O
# boundary; a lookup miss or read failure folds into an ErrorInfo cause. Never
# returns a domain aggregate -- that would be the read-model bypass smell (CQRS_004).
FindDatasetFn = Callable[[DatasetID], Result[pd.DataFrame, ErrorInfo]]


@dataclass(frozen=True, slots=True)
class GetDataset:
    """Query: fetch the stored frame for ``dataset_id`` as the read model.

    Carries only edge data -- the identity to look up, as a bare ``str`` (an edge
    DTO holds primitives; the handler re-wraps it into a ``DatasetID`` via the
    domain :func:`make_datasetid` before the read port is touched). Unlike a write
    command it holds no DataFrame, so default value equality is safe (no ambiguous
    elementwise ``==`` to sidestep). The read port is injected into the handler.
    """

    dataset_id: str


# The union of every datasets read query -- one member today, dispatched by
# match/case at the delivery edge. Grows as queries are added (no base class).
type DatasetQuery = GetDataset


@dataclass(frozen=True, slots=True)
class FailGetDataset:
    """Failure of handle_get_dataset, carrying its structured ``cause``.

    The read-boundary ``ErrorInfo`` folds into one application error, so the failure
    crosses UP the ring as a serializable descriptor -- never a lower-layer
    exception type.
    """

    cause: ErrorInfo


def _fail_get_from_invalid_id(error: InvalidDatasetID) -> FailGetDataset:
    """Lift a domain id-validation rejection into the query handler's own error.

    The offending id string stays in the domain error; the boundary reports a
    stable ``code`` plus message, never the domain type -- symmetric with
    :func:`_fail_from_schema` on the write side. The rejection is logged here at the
    boundary (the domain core stays free of logging).
    """
    logger.warning(
        "dataset id rejected on read: %r is not a valid id", error.dataset_id
    )
    return FailGetDataset(
        ErrorInfo(
            code="INVALID_DATASET_ID",
            message=f"invalid dataset id: {error.dataset_id!r}",
        )
    )


def handle_get_dataset(
    *, find: FindDatasetFn, query: GetDataset
) -> Result[pd.DataFrame, FailGetDataset]:
    """Fetch the stored frame for ``query.dataset_id`` as the read model.

    Re-wraps the edge ``str`` id into a ``DatasetID`` via the domain
    :func:`make_datasetid` (an invalid id short-circuits to ``FailGetDataset`` and
    the read port is never touched), then binds the injected ``FindDatasetFn`` and
    returns its bare DataFrame on the success rail -- the read model DTO, ready for
    projection -- never the Dataset aggregate (loading an aggregate only to show it
    is the read-model bypass smell, CQRS_004). A read-boundary ``ErrorInfo`` folds
    into ``FailGetDataset``. Pure railway composition -- no branching on the
    ``Result``.
    """
    return (
        make_datasetid(query.dataset_id)
        .fmap_err(_fail_get_from_invalid_id)
        .and_then(lambda dataset_id: find(dataset_id).fmap_err(FailGetDataset))
    )


# --- (3) Projections ---
