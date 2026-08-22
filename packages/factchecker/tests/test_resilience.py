"""Tests for the failure classifier and the retry policy in `factchecker.resilience`.

Every failure here is synthetic, so a classifier that matches nothing real would pass
this suite and fail every live call. The shapes below were therefore read off the
pinned stack rather than recalled: `mcp` 1.29.0 calls `httpx.Response.raise_for_status`
inside an `anyio` task group, so what reaches a caller of `langchain-mcp-adapters`
0.3.2 is an `ExceptionGroup` around an `httpx` exception. `_status_failure` builds its
exception by calling `raise_for_status` rather than by hand, so the message this suite
asserts the token out of is the message a live call would produce.
"""

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from factchecker.errors import AuthenticationFailed
from factchecker.resilience import FailureKind, classify, with_retry

CREDENTIAL = "brd-4a7f2e91c0"
ENDPOINT = f"https://mcp.brightdata.com/mcp?token={CREDENTIAL}"

# One band per retry: the base delay doubles, and jitter moves it by a quarter either
# way. The bands do not overlap, so "the delay grows" is assertable without a seed.
BANDS = ((0.375, 0.625), (0.75, 1.25), (1.5, 2.5))


def _status_failure(status: int) -> httpx.HTTPStatusError:
    """The exception `httpx` raises for an error status, with its own message."""
    response = httpx.Response(status, request=httpx.Request("POST", ENDPOINT))
    with pytest.raises(httpx.HTTPStatusError) as raised:
        response.raise_for_status()
    return raised.value


def _from_the_transport(*failures: Exception) -> BaseException:
    """Wrap failures the way the MCP transport's task group hands them to a caller."""
    return ExceptionGroup("unhandled errors in a TaskGroup", list(failures))


class _Call:
    """A call that yields each outcome in turn: an exception raised, text returned."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.made = 0

    async def __call__(self) -> str:
        self.made += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)


class _Clock:
    """A sleep that records what it was asked to wait and waits for none of it."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _retry(
    call: Callable[[], Awaitable[str]], attempts: int, clock: _Clock
) -> str | BaseException:
    """Drive `with_retry` to completion, and hand back whatever it produced."""
    return asyncio.run(with_retry(call, attempts, clock))


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_rate_limit_and_a_server_error_are_transient(status: int) -> None:
    """The server is asking for another try, so the caller makes one."""
    assert classify(_status_failure(status)) == "transient"


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_credential_is_an_authentication_failure(status: int) -> None:
    """Retrying a rejected token only spends the budget rejecting it again."""
    assert classify(_status_failure(status)) == "authentication"


@pytest.mark.parametrize("status", [400, 404, 418, 422])
def test_every_other_status_is_permanent(status: int) -> None:
    """A malformed or absent request will be malformed or absent next time too."""
    assert classify(_status_failure(status)) == "permanent"


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("All connection attempts failed"),
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectTimeout("timed out"),
    ],
)
def test_a_dropped_or_refused_connection_is_transient(failure: Exception) -> None:
    """Nothing was answered, so nothing about the request is known to be wrong."""
    assert classify(failure) == "transient"


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("no tool by that name"),
        McpError(ErrorData(code=-32601, message="Method not found")),
    ],
)
def test_a_failure_carrying_no_http_status_is_permanent(failure: Exception) -> None:
    """`McpError` carries a JSON-RPC code, not an HTTP one, so it reads as permanent."""
    assert classify(failure) == "permanent"


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        (_status_failure(429), "transient"),
        (_status_failure(401), "authentication"),
        (_status_failure(404), "permanent"),
    ],
)
def test_the_transport_s_wrapping_does_not_hide_the_failure(
    failure: httpx.HTTPStatusError, kind: FailureKind
) -> None:
    """This is the shape a live call raises, so the classifier has to see through it."""
    assert classify(_from_the_transport(failure)) == kind


def test_a_group_holding_a_rejected_credential_reads_as_authentication() -> None:
    """One rejected credential ends the run, whatever else failed alongside it."""
    group = _from_the_transport(_status_failure(503), _status_failure(401))

    assert classify(group) == "authentication"


def test_a_call_that_succeeds_is_made_once_and_waits_for_nothing() -> None:
    """The retry policy costs nothing on the path that does not fail."""
    call, clock = _Call("the page"), _Clock()

    assert _retry(call, 3, clock) == "the page"
    assert call.made == 1
    assert clock.delays == []


def test_a_transient_failure_is_retried_until_it_succeeds() -> None:
    """Two refusals and an answer is an answer."""
    failure = _from_the_transport(_status_failure(503))
    call, clock = _Call(failure, failure, "the page"), _Clock()

    assert _retry(call, 3, clock) == "the page"
    assert call.made == 3
    assert len(clock.delays) == 2


def test_a_transient_failure_that_never_clears_raises_what_it_last_saw() -> None:
    """The attempts run out and the caller sees the failure, not a substitute."""
    failure = _from_the_transport(_status_failure(503))
    call, clock = _Call(failure, failure, failure), _Clock()

    with pytest.raises(ExceptionGroup) as raised:
        _retry(call, 3, clock)

    assert raised.value is failure
    assert call.made == 3


def test_the_delay_between_attempts_grows_and_is_jittered() -> None:
    """Backoff spreads a burst of retries out instead of hammering in step."""
    failure = _from_the_transport(_status_failure(503))
    call, clock = _Call(*[failure] * 4), _Clock()

    with pytest.raises(ExceptionGroup):
        _retry(call, 4, clock)

    assert len(clock.delays) == len(BANDS)
    for delay, (lowest, highest) in zip(clock.delays, BANDS, strict=True):
        assert lowest <= delay <= highest
    assert clock.delays == sorted(clock.delays)


@pytest.mark.parametrize("attempts", [1, 2, 5])
def test_the_number_of_attempts_is_the_one_the_caller_passes(attempts: int) -> None:
    """The count is configuration, so nothing here may hold a literal of its own."""
    failure = _from_the_transport(_status_failure(503))
    call, clock = _Call(*[failure] * attempts), _Clock()

    with pytest.raises(ExceptionGroup):
        _retry(call, attempts, clock)

    assert call.made == attempts
    assert len(clock.delays) == attempts - 1


def test_a_permanent_failure_raises_at_once() -> None:
    """Nothing is gained by asking a second time, so nothing waits."""
    failure = _from_the_transport(_status_failure(404))
    call, clock = _Call(failure, "unreached"), _Clock()

    with pytest.raises(ExceptionGroup) as raised:
        _retry(call, 3, clock)

    assert raised.value is failure
    assert call.made == 1
    assert clock.delays == []


def test_a_rejected_credential_becomes_this_package_s_own_failure() -> None:
    """`run_check` ends a run on `AuthenticationFailed`, and this is where it starts."""
    rejection = _from_the_transport(_status_failure(401))
    call, clock = _Call(rejection, "unreached"), _Clock()

    with pytest.raises(AuthenticationFailed):
        _retry(call, 3, clock)

    assert call.made == 1
    assert clock.delays == []


def test_the_rejected_credential_carries_no_token_out_of_the_retry() -> None:
    """`httpx` names the request URL in its message, and that URL is the credential."""
    original = _from_the_transport(_status_failure(401))
    assert CREDENTIAL in str(original.exceptions[0])

    with pytest.raises(AuthenticationFailed) as raised:
        _retry(_Call(original), 3, _Clock())

    assert CREDENTIAL not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is original


def test_a_cancelled_call_is_not_retried() -> None:
    """A per-statement timeout has to end the call, not start it again."""
    call, clock = _Call(asyncio.CancelledError(), "unreached"), _Clock()

    with pytest.raises(asyncio.CancelledError):
        _retry(call, 3, clock)

    assert call.made == 1
    assert clock.delays == []
