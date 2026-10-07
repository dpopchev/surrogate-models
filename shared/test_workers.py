"""Facts about running jobs on worker processes, on toy module-level jobs."""

import os
from functools import partial

import pytest

from shared.workers import run_jobs


def square(x: int) -> int:
    """A toy job."""
    return x * x


def test_results_come_back_in_job_order_from_two_workers() -> None:
    assert run_jobs([partial(square, n) for n in (1, 2, 3)], workers=2) == [1, 4, 9]


def broken() -> int:
    """A toy job that fails."""
    raise ValueError("job broke")


def test_a_raising_job_fails_the_call_with_its_error() -> None:
    with pytest.raises(ValueError, match="job broke"):
        run_jobs([partial(square, 1), broken], workers=2)


def test_one_worker_runs_the_jobs_in_this_process() -> None:
    assert run_jobs([os.getpid], workers=1) == [os.getpid()]
