"""Facts about the rung chain of make surrogates (W-172)."""

import pytest

from paperlib.tools.regenerate import Done, Rung, run

RUNGS = (
    Rung("alpha", ("alpha", "A_BATCH="), (), "A_BATCH"),
    Rung("beta", ("beta", "B_BATCH="), (("A_BATCH", "alpha"),), "B_BATCH"),
    Rung("gamma", ("gamma",), (("B_BATCH", "beta"),), None),
    Rung("delta", ("delta",), (), None),
)


@pytest.mark.command
def test_chain_skips_recorded_rungs_and_stops_on_a_failure() -> None:
    record = {"alpha": Done("a-old", 60.0)}
    calls: list[tuple[str, ...]] = []
    lines: list[str] = []
    seconds = iter([0.0, 5.0, 5.0, 9.0])

    def execute(argv: tuple[str, ...]) -> int:
        calls.append(argv)
        return 7 if argv[1] == "gamma" else 0

    status = run(
        RUNGS,
        record,
        execute,
        lambda: next(seconds),
        lambda rung, started: f"{rung.name}-fresh",
        lines.append,
    )
    assert (status, calls, lines, record) == (
        7,
        [("make", "beta", "B_BATCH=", "A_BATCH=a-old"), ("make", "gamma", "B_BATCH=beta-fresh")],
        [
            "alpha: a-old in 1m 0s (recorded, skipped)",
            "beta: beta-fresh in 5s",
            "gamma failed with exit status 7",
        ],
        {"alpha": Done("a-old", 60.0), "beta": Done("beta-fresh", 5.0)},
    )
