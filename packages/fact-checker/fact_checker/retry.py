"""Which failures are worth repeating, and how long to wait between tries."""

import asyncio
import random
from collections.abc import Awaitable, Callable

import httpx
import openai

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0

_TRANSIENT_TYPES = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.TransportError,
)
_REJECTION_TYPES = (openai.AuthenticationError, openai.PermissionDeniedError)
_REJECTION_STATUSES = frozenset({401, 403})
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR_FLOOR = 500
_SERVER_ERROR_CEILING = 599


def _jitter() -> float:
    """Return the spread applied to one backoff delay, in `[0, 1)`."""
    return random.random()  # noqa: S311 — retry spread, with no bearing on security


def _status_of(exc: BaseException) -> int | None:
    """Return the HTTP status an exception reports, wherever it keeps it."""
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def is_authentication_failure(exc: BaseException) -> bool:
    """Report whether a credential was rejected, so the run has to end.

    The MCP client stack runs on task groups, so a rejection can reach here
    wrapped in an `ExceptionGroup`, at any depth.

    Args:
        exc: The failure to classify.

    Returns:
        True where the provider rejected the credential rather than the request,
        or where any failure a group holds was such a rejection.
    """
    if isinstance(exc, BaseExceptionGroup):
        return any(is_authentication_failure(held) for held in exc.exceptions)
    if isinstance(exc, _REJECTION_TYPES):
        return True
    return _status_of(exc) in _REJECTION_STATUSES


def is_transient(exc: BaseException) -> bool:
    """Report whether waiting and trying again could plausibly succeed.

    Args:
        exc: The failure to classify.

    Returns:
        True for a rate limit, a server-side error, or a broken connection,
        held directly or inside a group at any depth. A rejected credential is
        never transient: waiting does not make a rejected key valid, and a group
        holding one is not transient either.
    """
    if is_authentication_failure(exc):
        return False
    if isinstance(exc, BaseExceptionGroup):
        return any(is_transient(held) for held in exc.exceptions)
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    status = _status_of(exc)
    if status is None:
        return False
    return status == _TOO_MANY_REQUESTS or (
        _SERVER_ERROR_FLOOR <= status <= _SERVER_ERROR_CEILING
    )


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[], float] = _jitter,
) -> T:
    """Await `operation`, repeating it while a transient failure leaves attempts.

    Args:
        operation: The call to make, and to remake on a transient failure.
        attempts: How many times to call `operation` in total.
        base_delay: The first delay's length in seconds, before jitter.
        sleep: How to wait; injected so a test neither waits nor flakes.
        jitter: The spread applied to each delay; injected for the same reason.

    Returns:
        Whatever the first successful attempt returned.

    Raises:
        Exception: The last attempt's failure, or the first non-transient one.
    """
    for attempt in range(1, attempts):
        try:
            return await operation()
        except Exception as exc:
            if not is_transient(exc):
                raise
            await sleep(base_delay * 2 ** (attempt - 1) * (1 + jitter()))
    return await operation()
