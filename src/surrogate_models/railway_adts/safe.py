"""The ``@safe`` / ``@safe_async`` decorators -- turn exceptions into a ``Result``.

This is the bridge from Python's exception world onto the railway. Decorate a
side-effecting function and the wrapped call returns ``Ok(value)`` on success, or
``Err(fmap_err(exc))`` when it raises one of the declared ``catch`` exceptions;
any other exception propagates unchanged. Apply it at an I/O boundary (the
imperative shell) so the rest of the program composes ``Result`` values with
``.fmap``/``.and_then`` and never has to ``try``/``except``.

Use ``@safe`` for ordinary functions and ``@safe_async`` for coroutine functions
(``async def``): plain ``@safe`` would wrap the un-awaited coroutine in ``Ok``
without ever running its body, so an exception raised on ``await`` would escape
uncaught. ``@safe_async`` awaits the call inside the guard.

    >>> @safe(ZeroDivisionError, lambda exc: "div by zero")
    ... def reciprocal(n: int) -> float:
    ...     return 1 / n
    >>> reciprocal(4)
    Ok(value=0.25)
    >>> reciprocal(0)
    Err(error='div by zero')

An undeclared exception is not caught -- it still propagates:

    >>> reciprocal("oops")
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for /: 'int' and 'str'
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from surrogate_models.railway_adts.result import Err, Ok, Result


def safe[**P, T, E](
    catch: type[Exception] | tuple[type[Exception], ...],
    fmap_err: Callable[[Exception], E],
) -> Callable[[Callable[P, T]], Callable[P, Result[T, E]]]:
    """Build a decorator wrapping a synchronous side-effecting call into a ``Result``.

    ``catch`` is the exception type (or tuple of types) treated as a recoverable
    failure; ``fmap_err`` maps a caught exception onto the typed error ``E`` that
    travels the failure rail. For a structured, chainable error use
    :func:`~railway_adts.error.fmap_error` as ``fmap_err``. For ``async def``
    functions use :func:`safe_async` instead.
    """

    def decorate(fn: Callable[P, T]) -> Callable[P, Result[T, E]]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E]:
            try:
                return Ok(fn(*args, **kwargs))
            except catch as exc:
                return Err(fmap_err(exc))

        return wrapper

    return decorate


def safe_async[**P, T, E](
    catch: type[Exception] | tuple[type[Exception], ...],
    fmap_err: Callable[[Exception], E],
) -> Callable[
    [Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, Result[T, E]]]
]:
    """Build a decorator wrapping a coroutine (``async def``) into a ``Result``.

    The asynchronous counterpart of :func:`safe`: it awaits the wrapped call inside
    the guard, so the coroutine's body actually runs and a raised ``catch``
    exception is mapped onto ``Err``. Plain :func:`safe` cannot do this -- it would
    wrap the un-awaited coroutine in ``Ok`` and never run it, letting the exception
    escape on ``await``. ``catch`` and ``fmap_err`` behave exactly as in
    :func:`safe`.

    >>> import asyncio
    >>> @safe_async(ZeroDivisionError, lambda exc: "div by zero")
    ... async def reciprocal(n: int) -> float:
    ...     return 1 / n
    >>> asyncio.run(reciprocal(4))
    Ok(value=0.25)
    >>> asyncio.run(reciprocal(0))
    Err(error='div by zero')
    """

    def decorate(
        fn: Callable[P, Awaitable[T]],
    ) -> Callable[P, Coroutine[Any, Any, Result[T, E]]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E]:
            try:
                return Ok(await fn(*args, **kwargs))
            except catch as exc:
                return Err(fmap_err(exc))

        return wrapper

    return decorate
