"""Which failures are worth another try, how long to wait, and how one is reported.

Every match below reads `httpx` 0.28.1, which is the package
`mcp/client/streamable_http.py` imports. The `httpx2` 2.12.0 installed beside it in
the same environment is a separate distribution with exception classes of its own, and
a classifier written against those would match nothing a live call raises.
"""

import random
from collections.abc import Awaitable, Callable
from typing import Literal

import httpx

from factchecker.config import McpEndpoint
from factchecker.errors import AuthenticationFailed, McpCallError

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

    A tool-level error — the `CallToolResult(isError=True)` that `load_tools` asks the
    adapter to raise rather than answer with — falls through as permanent too. It
    arrives as a `ToolException` carrying the server's own prose and no status of any
    kind. Bright Data reports a rate limit, a blocked target and a rejected argument
    in that same prose, and no reading of it survives a wording change, so the
    statement fails once and plainly instead of spending the run's budget on a guess.

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


def describe_failure(endpoint: McpEndpoint, failure: BaseException) -> str:
    """Say what the server did, in words this package wrote itself.

    Nothing another library wrote reaches the returned text. An `httpx` message names
    the request URL and the Bright Data token travels inside it, so the endpoint is
    named through `McpEndpoint`, which prints itself redacted, and everything else is
    a status code. A status code is not a secret.

    Args:
        endpoint: The endpoint the failed call was made against.
        failure: The failure the call raised, wrapped however the transport wrapped it.

    Returns:
        One sentence, which a reader who writes no Python can act on.
    """
    found = _from_httpx(failure)
    if isinstance(found, httpx.HTTPStatusError):
        return f"the MCP server at {endpoint} returned {found.response.status_code}"
    if isinstance(found, httpx.TransportError):
        return f"the MCP server at {endpoint} could not be reached"
    return f"the MCP server at {endpoint} could not complete the request"


async def with_retry(
    call: Callable[[], Awaitable[str]],
    attempts: int,
    sleep: Callable[[float], Awaitable[None]],
    endpoint: McpEndpoint,
) -> str:
    """Make the call, and make it again while failing is worth repeating.

    Sleeping is an argument rather than `asyncio.sleep` so that a test asserts on the
    backoff schedule without waiting out a single second of it.

    Args:
        call: The call to make. It is made afresh on each attempt.
        attempts: How many times to make it at most.
        sleep: Waits the number of seconds it is given.
        endpoint: The endpoint the call is made against, named in what is raised.

    Returns:
        Whatever the call returned, on the first attempt that returns.

    Raises:
        AuthenticationFailed: A credential was rejected, which no retry will change.
        McpCallError: The call failed permanently, or it went on failing until the
            attempts ran out. The message names the endpoint redacted and the HTTP
            status where the failure carried one.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await call()
        except Exception as failure:  # noqa: BLE001 — classified, then re-raised as ours
            # Neither raise below chains its cause. The cause arrives from `httpx`,
            # whose message names the request URL, and the Bright Data token travels
            # inside that URL.
            kind = classify(failure)
            if kind == "authentication":
                raise AuthenticationFailed(_CREDENTIAL_REJECTED) from None
            if kind == "permanent" or attempt >= attempts:
                raise McpCallError(describe_failure(endpoint, failure)) from None
        await sleep(_delay(attempt))


def _from_httpx(failure: BaseException) -> httpx.HTTPError | None:
    """Find the `httpx` failure inside whatever the transport wrapped it in."""
    if isinstance(failure, httpx.HTTPError):
        return failure
    if isinstance(failure, BaseExceptionGroup):
        found = (_from_httpx(member) for member in failure.exceptions)
        return next((one for one in found if one is not None), None)
    return None


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
