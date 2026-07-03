"""A structured, serializable failure descriptor and the helper that builds one.

``ErrorInfo`` is the shape a low-level error takes when a higher layer wraps it as
a ``cause``: it carries a stable, machine-readable ``code`` and a human-readable
``message`` without leaking the lower layer's own exception type. This keeps
layered failures chainable -- the railway-oriented analogue of ``raise X from Y``
-- so a higher layer can report its own error while still carrying the original
reason.

``fmap_error`` is the boundary adapter that turns a caught exception (or any
lower-layer error) into such a cause and lifts it into a layer-specific error
type. Pair it with :func:`~railway_adts.safe.safe` to translate exceptions into
typed errors at an I/O boundary.

    >>> err = ErrorInfo(code="E_NOT_FOUND", message="user 42 not found")
    >>> str(err)            # the message alone -- what an edge surfaces to a user
    'user 42 not found'
    >>> err                 # the full descriptor -- what logs and matching use
    ErrorInfo(code='E_NOT_FOUND', message='user 42 not found')
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """A failure's stable ``code`` plus a human-readable ``message``."""

    code: str
    message: str

    def __str__(self) -> str:
        """The human-readable reason -- what the edge surfaces to a user.

        ``code`` stays for machine/log use via ``repr``; ``str`` is the message
        alone, so an ``f"...: {error}"`` at the CLI edge reads cleanly.
        """
        return self.message


def fmap_error[E](
    into: Callable[[ErrorInfo], E], code: str, where: object = ""
) -> Callable[[object], E]:
    """Build a function that wraps any error into an ``ErrorInfo`` cause and lifts
    it to a layer-specific error of type ``E`` via ``into``.

    The caught failure becomes a chainable ``ErrorInfo`` -- its stable ``code`` plus
    a message (``str(error)``, optionally prefixed with ``where`` for context such
    as a file path) -- and ``into`` maps that cause into your own error type. A
    higher layer can thus report its own failure (e.g. "config load failed")
    WITHOUT naming the lower layer's exception type (``FileNotFoundError``,
    ``PermissionError``, ...). Use the returned function as the ``fmap_err`` of
    :func:`~railway_adts.safe.safe`, or to map the error rail of a ``Result`` at a
    layer boundary.

    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class LoadError:
    ...     cause: ErrorInfo
    >>> wrap = fmap_error(LoadError, code="E_IO", where="config.toml")
    >>> wrap(FileNotFoundError("No such file"))
    LoadError(cause=ErrorInfo(code='E_IO', message='config.toml: No such file'))

    Without ``where`` the message is the error rendered as-is:

    >>> to_cause = fmap_error(lambda c: c, code="E_PARSE")
    >>> to_cause(ValueError("unexpected token"))
    ErrorInfo(code='E_PARSE', message='unexpected token')
    """

    def wrap(error: object) -> E:
        message = f"{where}: {error}" if where else str(error)
        return into(ErrorInfo(code=code, message=message))

    return wrap
