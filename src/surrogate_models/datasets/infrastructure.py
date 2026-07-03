"""Datasets infrastructure -- the imperative shell of the datasets context.

Supplies the impure bits the pure core must not touch. factory_datasetid mints a
fresh DatasetID from a UUID; the application injects it as a DatasetIdFn so the
domain stays deterministic. save_dataset persists a certified Dataset and, curried
on its leading path, satisfies the application's SaveDatasetFn port. Mappers and
the real writer body are added as the context grows.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from surrogate_models.datasets.domain import Dataset, DatasetID
from surrogate_models.railway_adts import fmap_error, safe


def factory_datasetid() -> DatasetID:
    """Mint a fresh DatasetID from a UUID's first 8 hex characters.

    Nondeterministic, hence in the shell (never the domain). 8 hex characters =
    32 bits of identity -- ample for surrogate-model datasets.
    """
    return DatasetID(uuid4().hex[:8])


@safe(OSError, fmap_error(lambda cause: cause, code="DATASET_SAVE_FAILED"))
def save_dataset(path: Path, dataset: Dataset) -> None:
    """Persist ``dataset`` as ``path/{dataset_id}.parquet`` (parents created).

    The write side's persistence port. ``path`` leads so the composition root can
    bind it with ``partial(save_dataset, settings.datasets.path)`` to yield the
    application's ``SaveDatasetFn = Callable[[Dataset], Result[None, ErrorInfo]]``.

    ``@safe`` makes this the railway boundary: on success the wrapped call returns
    ``Ok(None)``; any ``OSError`` (missing/blocked path, permission, full disk) is
    caught and folded into ``Err(ErrorInfo(code="DATASET_SAVE_FAILED"))`` -- the
    structured cause crosses UP, never the raw exception. A non-``OSError`` (e.g. a
    missing parquet engine) is left to propagate: that is a boot/config fault, not
    a recoverable save failure.
    """
    target = path / f"{dataset.dataset_id}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(target)
