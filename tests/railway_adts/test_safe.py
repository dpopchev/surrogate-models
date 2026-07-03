"""Tests for the @safe / @safe_async exception-to-Result decorators.

The async cases drive the coroutine with ``asyncio.run`` so the suite needs no
async test plugin.
"""

import asyncio

import pytest

from surrogate_models.railway_adts.result import Err, Ok
from surrogate_models.railway_adts.safe import safe, safe_async

# --- safe (synchronous) ---


def test_safe_returns_ok_on_success() -> None:
    @safe(ZeroDivisionError, lambda exc: "div by zero")
    def reciprocal(n: int) -> float:
        return 1 / n

    assert reciprocal(4) == Ok(0.25)


def test_safe_maps_caught_exception_to_err() -> None:
    @safe(ZeroDivisionError, lambda exc: "div by zero")
    def reciprocal(n: int) -> float:
        return 1 / n

    assert reciprocal(0) == Err("div by zero")


def test_safe_passes_the_caught_exception_to_fmap_err() -> None:
    @safe(ValueError, lambda exc: type(exc).__name__)
    def parse(s: str) -> int:
        return int(s)

    assert parse("x") == Err("ValueError")


def test_safe_accepts_a_tuple_of_catch_types() -> None:
    @safe((ValueError, TypeError), lambda exc: type(exc).__name__)
    def parse(s: str) -> int:
        return int(s)

    assert parse("x") == Err("ValueError")


def test_safe_propagates_an_undeclared_exception() -> None:
    @safe(ValueError, str)
    def boom() -> None:
        raise KeyError("nope")

    with pytest.raises(KeyError):
        boom()


def test_safe_preserves_the_wrapped_function_name() -> None:
    @safe(ValueError, str)
    def named(x: int) -> int:
        return x

    assert named.__name__ == "named"


# --- safe_async (asynchronous) ---


def test_safe_async_returns_ok_on_success() -> None:
    @safe_async(ZeroDivisionError, lambda exc: "div by zero")
    async def reciprocal(n: int) -> float:
        return 1 / n

    assert asyncio.run(reciprocal(4)) == Ok(0.25)


def test_safe_async_maps_caught_exception_to_err() -> None:
    @safe_async(ZeroDivisionError, lambda exc: "div by zero")
    async def reciprocal(n: int) -> float:
        return 1 / n

    assert asyncio.run(reciprocal(0)) == Err("div by zero")


def test_safe_async_propagates_an_undeclared_exception() -> None:
    @safe_async(ValueError, str)
    async def boom() -> None:
        raise KeyError("nope")

    with pytest.raises(KeyError):
        asyncio.run(boom())


def test_safe_async_awaits_the_body_returning_a_value_not_a_coroutine() -> None:
    # Regression guard: plain @safe would yield Ok(<coroutine>); @safe_async runs
    # the body and returns Ok of its value.
    @safe_async(ValueError, str)
    async def answer() -> int:
        return 42

    assert asyncio.run(answer()) == Ok(42)
