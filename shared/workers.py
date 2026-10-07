"""Independent jobs on k worker processes (W-081).

T-148 measured 65 fits per hour on one process and 206 on six one-thread workers. The harness
pins every fit to one torch thread and one BLAS thread (shared/harness.py), so k workers running
harness jobs use about k cores. A job is a picklable zero-argument callable (a module-level
function or a functools.partial of one).
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor


def run_jobs[T](jobs: Sequence[Callable[[], T]], workers: int) -> list[T]:
    """The results of the jobs in job order, run on `workers` processes (1: in this process,
    in order); a job that raises fails the call with its error."""
    if workers == 1:
        return [job() for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job) for job in jobs]
        return [future.result() for future in futures]
