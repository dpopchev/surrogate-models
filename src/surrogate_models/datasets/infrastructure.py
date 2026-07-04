"""Datasets infrastructure -- the imperative shell of the datasets context.

Supplies the impure bits the pure core must not touch. factory_datasetid mints a
fresh DatasetID from a UUID; the application injects it as a DatasetIdFn so the
domain stays deterministic. save_dataset persists a certified Dataset and, curried
on its leading path, satisfies the application's SaveDatasetFn port;
find_dataset_frame is its read counterpart -- curried on path it satisfies the
FindDatasetFn read port, reading the stored frame back as the read model. Mappers
and further adapters are added as the context grows.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pandas as pd

from surrogate_models.datasets.domain import Dataset, DatasetID
from surrogate_models.railway_adts import fmap_error, safe

logger = logging.getLogger(__name__)

_PC_PATTERN = re.compile(r"\bPc\s*=\s*([0-9.eE+-]+)")


def factory_datasetid() -> str:
    """Mint a fresh candidate id STRING from a UUID's first 8 hex characters.

    Nondeterministic, hence in the shell (never the domain). Returns a bare ``str``
    -- a candidate the domain ``make_datasetid`` certifies and wraps into a
    ``DatasetID`` (the sole constructor); the shell never mints the domain type
    itself. 8 hex characters = 32 bits of identity -- ample for surrogate-model
    datasets, and always a valid id under ``make_datasetid``'s stem rule.
    """
    return uuid4().hex[:8]


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


@safe(OSError, fmap_error(lambda cause: cause, code="DATASET_READ_FAILED"))
def find_dataset_frame(path: Path, dataset_id: DatasetID) -> pd.DataFrame:
    """Read ``path/{dataset_id}.parquet`` back into a bare DataFrame (the read model).

    The read side's lookup port -- the counterpart of :func:`save_dataset`. ``path``
    leads so the composition root can bind it with
    ``partial(find_dataset_frame, settings.datasets.path)`` to yield the
    application's ``FindDatasetFn = Callable[[DatasetID], Result[pd.DataFrame,
    ErrorInfo]]``. It returns the raw frame -- FIND a view to show it -- never the
    Dataset aggregate, so the read path bypasses re-certification entirely.

    ``@safe`` makes this the railway boundary: on success the wrapped call returns
    ``Ok(frame)``; any ``OSError`` -- a missing file (``FileNotFoundError``), a
    permission error, an unreadable path -- is caught and folded into
    ``Err(ErrorInfo(code="DATASET_READ_FAILED"))`` (one uniform read-boundary code,
    with the caught reason preserved in the message). A non-``OSError`` (e.g. a
    missing parquet engine) propagates: that is a boot/config fault, not a
    recoverable read failure -- symmetric with :func:`save_dataset`.
    """
    return pd.read_parquet(path / f"{dataset_id}.parquet")


@safe(
    (OSError, ValueError),
    fmap_error(lambda cause: cause, code="NEUTRON_STARS_READ_FAILED"),
)
def read_neutron_stars_frame(path: Path) -> pd.DataFrame:
    """Digest the raw concatenated neutron-stars ``.dat`` at ``path`` into one frame.

    The ingest reader. The file is a run of batches, each ``[# comment, header row,
    data rows...]``: the comment carries the batch parameters (``x1``, ``x2``,
    ``beta``, ``Lambda``, ``Pc``, ``choice_theory``); the header names the columns;
    the rows are whitespace-separated numbers. Five of the six comment params are
    already reproduced verbatim as constant columns, so ONLY ``Pc`` is appended --
    as ``pc_init``, because the data's own ``Pc`` column is the RUNNING central
    pressure and the comment's ``Pc`` is its per-batch initial value.

    A run that produced zero stars appears as an EMPTY batch -- its ``#`` parameter
    comment immediately followed by the next batch's comment, with no header/data
    rows. Such a batch is SKIPPED (it contributes no rows) and logged at ``warning``
    naming the dropped comments, so the empty runs stay a visible data-quality signal
    rather than silently vanishing -- or crashing the ingest (feeding ``pd.read_csv``
    an empty body raises ``EmptyDataError``, which is exactly what an unfiltered empty
    batch used to do).

    A related run prints its column HEADER but no data rows; ``pd.read_csv`` types
    that 0-row batch's columns as ``object``, and ``pd.concat`` then upcasts the
    shared columns of the whole frame to ``object`` -- which the domain's schema
    certification rejects. The final ``pd.to_numeric`` pass HEALS this: every column
    is a number, so coercion restores the explicit ``float64``/``int64`` dtype
    losslessly. Its default ``errors="raise"`` keeps the rail honest -- a genuinely
    non-numeric column raises ``ValueError`` and folds onto the ingest failure below
    rather than being silently blanked to ``NaN``.

    ``@safe`` makes this the railway boundary: on success the wrapped call returns
    ``Ok(frame)``. Both a read fault (``OSError`` -- missing/unreadable file) and a
    parse fault (``ValueError`` -- a batch comment with no ``Pc``, an unparseable
    row, a non-numeric column, or a file with no populated batches at all) are caught
    and folded into an ``Err`` carrying code ``NEUTRON_STARS_READ_FAILED``. The
    ``ValueError`` arm is broader than the ``OSError``-only save/find precedent on
    purpose: a corrupt incoming file is a recoverable ingest failure, not a boot
    fault, so its structured cause must cross UP rather than crash the process.
    """
    batches = list(_iter_batches(path.read_text()))
    empty = [_batch_comment(batch) for batch in batches if not _has_body(batch)]
    if empty:
        logger.warning(
            "read_neutron_stars_frame skipped %d empty batch(es) "
            "(comment with no data rows): %s",
            len(empty),
            empty,
        )
    frames = [_read_batch(batch) for batch in batches if _has_body(batch)]
    combined = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(
        {name: pd.to_numeric(column) for name, column in combined.items()}
    )


def _has_body(batch: str) -> bool:
    """True iff the batch carries header/data rows below its ``#`` comment line.

    An empty (comment-only) batch has nothing after the first newline; feeding its
    empty body to ``pd.read_csv`` raises ``EmptyDataError``, so the reader drops it.
    """
    return bool(batch.partition("\n")[2].strip())


def _batch_comment(batch: str) -> str:
    """The batch's leading ``#`` comment line (its parameters), for diagnostics."""
    return batch.partition("\n")[0]


def _iter_batches(text: str) -> Iterator[str]:
    """Yield each batch's raw text -- the ``#`` comment line down to its last data row.

    Batches are delimited by ``#`` comment lines, so a new batch opens on every
    ``#``-prefixed line and the running group is flushed before it. Empty input
    yields nothing, so the caller's ``pd.concat`` fails loudly (ValueError) on a file
    with no batches -- folded onto the ingest rail like any other parse fault.
    """
    batch: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and batch:
            yield "\n".join(batch)
            batch = []
        batch.append(line)
    if batch:
        yield "\n".join(batch)


def _read_batch(batch: str) -> pd.DataFrame:
    """Parse one batch's ``[# comment, header row, data rows...]`` into a frame.

    The comment's ``Pc`` becomes the constant ``pc_init`` column (its per-batch
    initial central pressure); the header + data rows parse as whitespace-separated
    numbers. A comment with no ``Pc`` is malformed input -> ``ValueError``, which the
    caller's ``@safe`` folds onto the ingest rail.
    """
    comment, _, body = batch.partition("\n")
    match = _PC_PATTERN.search(comment)
    if match is None:
        raise ValueError(f"batch comment missing Pc: {comment!r}")
    frame = pd.read_csv(io.StringIO(body), sep=r"\s+")
    frame["pc_init"] = float(match.group(1))
    return frame
