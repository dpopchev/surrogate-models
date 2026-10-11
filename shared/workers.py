"""Independent jobs on k worker processes (W-081).

T-148 measured 65 fits per hour on one process and 206 on six one-thread workers. The harness
pins every fit to one torch thread and one BLAS thread (shared/harness.py), so k workers running
harness jobs use about k cores. A job is a picklable zero-argument callable (a module-level
function or a functools.partial of one).
"""

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def _configure(log_level: str | None, quiet: Sequence[str]) -> None:
    """Set up a worker's logging once, as it starts: a console handler where the root logger
    has none (a forkserver worker inherits neither), the root level, and warnings only from
    each quiet logger."""
    if log_level is None:
        return
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(message)s")
    logging.getLogger().setLevel(log_level)
    for name in quiet:
        logging.getLogger(name).setLevel(logging.WARNING)


def run_jobs[T](
    jobs: Sequence[Callable[[], T]],
    workers: int,
    log_level: str | None = None,
    quiet: Sequence[str] = (),
    on_done: Callable[[int, T], Any] = lambda index, result: None,
) -> list[T]:
    """The results of the jobs in job order, run on `workers` processes (1: in this process,
    in order), each worker's logging at log_level with the quiet loggers at warnings (None:
    left to the jobs); on_done is handed each job's index and result in this process as the
    job ends; a job that raises fails the call with its error."""
    if workers == 1:
        names = ["", *quiet]
        levels = [logging.getLogger(name).level for name in names]
        _configure(log_level, quiet)
        try:
            results = []
            for index, job in enumerate(jobs):
                results.append(job())
                on_done(index, results[-1])
            return results
        finally:
            for name, level in zip(names, levels, strict=True):
                logging.getLogger(name).setLevel(level)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_configure, initargs=(log_level, tuple(quiet))
    ) as pool:
        futures = {pool.submit(job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            on_done(futures[future], future.result())
        return [future.result() for future in futures]
