"""Datasets context root -- composition + the public load_neutron_stars facade.

The context's composition seat: it reads settings via get_settings() and binds the
infrastructure adapters to this context's OWN application handlers (DIP), never
touching the domain ring directly (a domain import here would be a seal leak -- the
id is passed as the bare string ``"neutron-stars"`` and the domain make_datasetid
wraps it inside the handler).

load_neutron_stars() is the notebook-friendly get-or-build entry point re-exported
from the top-level package: read the stored frame, else ingest -> persist -> re-read.
"""

from __future__ import annotations

import logging
from functools import partial

import pandas as pd

from surrogate_models.config import Settings, get_settings
from surrogate_models.datasets.application import (
    FailGetDataset,
    FailMadeDataset,
    GetDataset,
    MakeDataset,
    handle_get_dataset,
    handle_make_dataset,
)
from surrogate_models.datasets.infrastructure import (
    factory_datasetid,
    find_dataset_frame,
    read_neutron_stars_frame,
    save_dataset,
)
from surrogate_models.railway_adts import ErrorInfo, Result

logger = logging.getLogger(__name__)

_NEUTRON_STARS_ID = "neutron-stars"


def _cause(failure: FailGetDataset | FailMadeDataset) -> ErrorInfo:
    """Project a handler failure onto its serializable ``ErrorInfo`` cause.

    Both handler errors carry a ``cause``; folding to the one ``ErrorInfo`` type lets
    the read rail and the build rail compose under a single ``or_else`` and unwrap.
    """
    return failure.cause


def _read(settings: Settings) -> Result[pd.DataFrame, ErrorInfo]:
    """Fetch the stored ``neutron-stars`` frame as the read model.

    An absent parquet folds to ``Err(DATASET_READ_FAILED)`` -- the miss the caller's
    ``or_else`` recovers into a build.
    """
    logger.debug("reading stored neutron-stars frame from %s", settings.datasets.path)
    find = partial(find_dataset_frame, settings.datasets.path)
    query = GetDataset(dataset_id=_NEUTRON_STARS_ID)
    return handle_get_dataset(find=find, query=query).fmap_err(_cause)


def _build(settings: Settings) -> Result[pd.DataFrame, ErrorInfo]:
    """Ingest the raw source, persist the dataset, and re-read it (read-through).

    Digest ``datasets.neutron_stars_source`` into a frame, certify + persist it under
    the fixed ``neutron-stars`` id via the write handler, then read the just-saved
    frame back. Every failure (ingest, certify/save) folds to ``ErrorInfo``.
    """
    logger.info(
        "stored neutron-stars frame absent; building from source %s",
        settings.datasets.neutron_stars_source,
    )
    return (
        read_neutron_stars_frame(settings.datasets.neutron_stars_source)
        .and_then(
            lambda frame: handle_make_dataset(
                save=partial(save_dataset, settings.datasets.path),
                new_id=factory_datasetid,
                cmd=MakeDataset(frame, dataset_id=_NEUTRON_STARS_ID),
            ).fmap_err(_cause)
        )
        .and_then(lambda _dataset_id: _read(settings))
    )


def load_neutron_stars() -> pd.DataFrame:
    """Return the ``neutron-stars`` dataset frame, building it on first use.

    Get-or-build: read the stored frame from ``datasets.path``; on a miss, ingest
    ``datasets.neutron_stars_source``, persist the dataset, and return the re-read
    frame. A bare DataFrame is returned; ``unwrap`` raises at this outer edge if both
    the read and the build fail (e.g. a missing source file) -- carrying the
    ``ErrorInfo`` cause.
    """
    settings = get_settings()
    return _read(settings).or_else(lambda _miss: _build(settings)).unwrap()
