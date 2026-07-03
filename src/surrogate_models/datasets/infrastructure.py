"""Datasets infrastructure -- the imperative shell of the datasets context.

Supplies the impure bits the pure core must not touch. factory_datasetid mints a
fresh DatasetID from a UUID; the application injects it as a FactoryDatasetIdFn so
the domain stays deterministic. Persistence and mappers are added as the context
grows.
"""

from __future__ import annotations

from uuid import uuid4

from surrogate_models.datasets.domain import DatasetID


def factory_datasetid() -> DatasetID:
    """Mint a fresh DatasetID from a UUID's first 8 hex characters.

    Nondeterministic, hence in the shell (never the domain). 8 hex characters =
    32 bits of identity -- ample for surrogate-model datasets.
    """
    return DatasetID(uuid4().hex[:8])
