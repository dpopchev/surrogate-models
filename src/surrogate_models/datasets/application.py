"""Datasets application -- CQRS handlers over the datasets domain.

Orchestrates the pure core via injected callables (DIP). Id generation is an
injected DatasetIdFn and persistence an injected SaveDatasetFn, both supplied by
infrastructure at the composition root, so the application mints a DatasetID and
hands it to the domain's make_dataset without the core ever touching uuid or I/O.
Command/query handlers are added under TDD, kept as ordered CQRS sections.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from surrogate_models.datasets.domain import (
    Dataset,
    DatasetID,
    DatasetMissingSchema,
    make_dataset,
)
from surrogate_models.railway_adts import ErrorInfo, Result

# --- (1) Commands ---

# Injected ports (DIP via callables): the shell supplies these at the composition
# root. DatasetIdFn mints identity (Factory prefix dropped); SaveDatasetFn persists
# a certified Dataset, yielding nothing on success and an ErrorInfo on failure.
DatasetIdFn = Callable[[], DatasetID]
SaveDatasetFn = Callable[[Dataset], Result[None, ErrorInfo]]


@dataclass(frozen=True, slots=True, eq=False)
class MakeDataset:
    """Command: certify and persist ``frame`` as a Dataset.

    Carries only edge data -- the frame to certify. ``eq=False`` keeps identity
    equality so a command never triggers a DataFrame's ambiguous elementwise
    ``==``. The id source and persistence are injected into the handler, not the
    command.
    """

    frame: pd.DataFrame


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
    ``raise FailMadeDataset from DatasetMissingSchema``.
    """
    return FailMadeDataset(
        ErrorInfo(
            code="DATASET_MISSING_SCHEMA",
            message="frame lacks a clear column schema",
        )
    )


def handle_make_dataset(
    *, save: SaveDatasetFn, new_id: DatasetIdFn, cmd: MakeDataset
) -> Result[DatasetID, FailMadeDataset]:
    """Mint an id, certify ``cmd.frame`` into a Dataset, and persist it.

    Returns ``Ok`` with the minted ``DatasetID`` once the frame certifies and the
    save succeeds. Both failure rails fold into ``FailMadeDataset``: a schema
    rejection via :func:`_fail_from_schema`, a save ``ErrorInfo`` wrapped verbatim
    as the cause. Pure railway composition -- no branching on the ``Result``.
    """
    dataset_id = new_id()
    return (
        make_dataset(dataset_id, cmd.frame)
        .fmap_err(_fail_from_schema)
        .and_then(lambda dataset: save(dataset).fmap_err(FailMadeDataset))
        .fmap(lambda _saved: dataset_id)
    )


# --- (2) Queries ---
# --- (3) Projections ---
