"""Abstract data types for railway-oriented programming in Python.

Railway-oriented programming models a computation as a track with two rails: a
success rail and a failure rail. Each step either stays on the success rail or
switches the whole pipeline onto the failure rail, which then bypasses every
later step. This package provides the small set of total, immutable data types
that make that style ergonomic in Python:

- :class:`~railway_adts.result.Result` (``Ok`` | ``Err``) -- a value or an error;
  the core of railway-oriented error handling, replacing ``raise``/``except``.
- :class:`~railway_adts.option.Option` (``Some`` | ``Nothing``) -- a value that
  may be absent, replacing ``None`` sentinels.
- :class:`~railway_adts.error.ErrorInfo` and
  :func:`~railway_adts.error.fmap_error` -- a serializable failure descriptor and
  the helper that lifts a caught exception into a typed, chainable error.
- :func:`~railway_adts.safe.safe` (and :func:`~railway_adts.safe.safe_async` for
  ``async def``) -- a decorator that turns a chosen exception into an ``Err`` at an
  I/O boundary, so the rest of the program never sees a ``raise``.

The variant names follow Rust's ``std::result``/``std::option`` (``Ok``/``Err``,
``Some``), except absence is spelled ``Nothing`` (Haskell) to avoid clashing with
Python's built-in ``None``.

    >>> from surrogate_models.railway_adts import Ok, Err, Some, Nothing
    >>> Ok(2).fmap(lambda x: x + 1).and_then(lambda x: Ok(x * 10)).unwrap()
    30
    >>> Err("boom").fmap(lambda x: x + 1).unwrap_or(-1)
    -1
    >>> Some(2).filter(lambda x: x > 0).ok_or("absent")
    Ok(value=2)
    >>> Nothing().ok_or("absent")
    Err(error='absent')
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from surrogate_models.railway_adts.error import ErrorInfo, fmap_error
from surrogate_models.railway_adts.option import Nothing, Option, Some, from_optional
from surrogate_models.railway_adts.result import Err, Ok, Result, UnwrapError
from surrogate_models.railway_adts.safe import safe, safe_async

try:
    __version__ = version("railway-adts")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "Err",
    "ErrorInfo",
    "Nothing",
    "Ok",
    "Option",
    "Result",
    "Some",
    "UnwrapError",
    "__version__",
    "fmap_error",
    "from_optional",
    "safe",
    "safe_async",
]
