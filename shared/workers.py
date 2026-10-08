"""Independent jobs on k worker processes (W-081).

T-148 measured 65 fits per hour on one process and 206 on six one-thread workers. The harness
pins every fit to one torch thread and one BLAS thread (shared/harness.py), so k workers running
harness jobs use about k cores. A job is a picklable zero-argument callable (a module-level
function or a functools.partial of one).
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def run_jobs[T](
    jobs: Sequence[Callable[[], T]],
    workers: int,
    on_done: Callable[[int, T], Any] = lambda index, result: None,
) -> list[T]:
    """The results of the jobs in job order, run on `workers` processes (1: in this process,
    in order); on_done is handed each job's index and result in this process as the job
    ends; a job that raises fails the call with its error."""
    if workers == 1:
        results = []
        for index, job in enumerate(jobs):
            results.append(job())
            on_done(index, results[-1])
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            on_done(futures[future], future.result())
        return [future.result() for future in futures]
