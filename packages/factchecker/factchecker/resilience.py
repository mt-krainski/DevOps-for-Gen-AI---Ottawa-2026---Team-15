"""Which failures are worth another try, and how long to wait before making it."""

import random
from collections.abc import Awaitable, Callable
from typing import Literal

import httpx

from factchecker.errors import AuthenticationFailed

FailureKind = Literal["transient", "authentication", "permanent"]

_AUTHENTICATION_STATUSES = frozenset({401, 403})
_RATE_LIMITED = 429
_SERVER_ERROR_FLOOR = 500

# Read outermost first: one rejected credential ends the run, so it outranks a
# transient failure it happens to be grouped with.
_BY_URGENCY: tuple[FailureKind, ...] = ("authentication", "transient", "permanent")

_BASE_DELAY_SECONDS = 0.5
_JITTER_FRACTION = 0.25

_CREDENTIAL_REJECTED = "the Bright Data MCP server rejected the credential"


def classify(exc: BaseException) -> FailureKind:
    """Decide what kind of failure this is, and so whether another try is worth making.

    The matching is written against what the pinned stack raises rather than against
    the MCP protocol. `mcp` 1.29.0 calls `httpx.Response.raise_for_status` inside an
    `anyio` task group, so an HTTP error status reaches a caller of
    `langchain-mcp-adapters` 0.3.2 as an `ExceptionGroup` around an
    `httpx.HTTPStatusError`. `McpError` carries a JSON-RPC code rather than an HTTP
    one, so it holds no status to read and falls through as permanent.

    Args:
        exc: The failure a call raised.

    Returns:
        `transient` where the same call may yet succeed, `authentication` where a
        credential was rejected, and `permanent` for everything else.
    """
    if isinstance(exc, BaseExceptionGroup):
        kinds = {classify(member) for member in exc.exceptions}
        return next(kind for kind in _BY_URGENCY if kind in kinds)
    if isinstance(exc, httpx.HTTPStatusError):
        return _by_status(exc.response.status_code)
    if isinstance(exc, httpx.TransportError):
        return "transient"
    return "permanent"


async def with_retry(
    call: Callable[[], Awaitable[str]],
    attempts: int,
    sleep: Callable[[float], Awaitable[None]],
) -> str:
    """Make the call, and make it again while failing is worth repeating.

    Sleeping is an argument rather than `asyncio.sleep` so that a test asserts on the
    backoff schedule without waiting out a single second of it.

    Args:
        call: The call to make. It is made afresh on each attempt.
        attempts: How many times to make it at most. The loader rejects a count
            below one, so no guard here repeats that.
        sleep: Waits the number of seconds it is given.

    Returns:
        Whatever the call returned, on the first attempt that returns.

    Raises:
        AuthenticationFailed: A credential was rejected, which no retry will change.
        Exception: The last transient failure, once the attempts run out, or a
            permanent failure on the attempt that met it.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await call()
        except Exception as failure:
            kind = classify(failure)
            if kind == "authentication":
                # The cause is dropped rather than chained. It arrives from `httpx`,
                # whose message names the request URL, and the Bright Data token
                # travels inside that URL.
                raise AuthenticationFailed(_CREDENTIAL_REJECTED) from None
            if kind == "permanent" or attempt == attempts:
                raise
        await sleep(_delay(attempt))


def _by_status(status: int) -> FailureKind:
    """Read an HTTP status as a failure kind."""
    if status in _AUTHENTICATION_STATUSES:
        return "authentication"
    if status == _RATE_LIMITED or status >= _SERVER_ERROR_FLOOR:
        return "transient"
    return "permanent"


def _delay(attempt: int) -> float:
    """How long to wait after the given attempt: doubling, and jittered either way.

    The jitter matters because a run checks its statements at once. Without it, a
    rate limit refuses every statement together and every retry arrives together.
    """
    doubling = _BASE_DELAY_SECONDS * 2 ** (attempt - 1)
    spread = random.uniform(-_JITTER_FRACTION, _JITTER_FRACTION)  # noqa: S311 — jitter is not a secret
    return doubling * (1 + spread)
