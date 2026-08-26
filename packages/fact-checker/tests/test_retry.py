"""Classifying a failure, and retrying the kind that waiting can fix."""

import httpx
import openai
import pytest

from fact_checker.retry import is_authentication_failure, is_transient, with_retry
from tests.conftest import (
    ResponseStatusError,
    StatusAttributeError,
    StatusCodeError,
    openai_connection_error,
    openai_status_error,
)


class RecordedSleep:
    """A `sleep` that records what it was asked to wait, and waits nothing."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        """Record the delay asked for, and return at once."""
        self.delays.append(delay)


def a_rate_limit() -> Exception:
    """Return a transient failure with no bearing on the assertion under test."""
    return openai_status_error(openai.RateLimitError, 429)


def grouped(*failures: Exception) -> ExceptionGroup[Exception]:
    """Wrap failures the way an anyio task group reports what its tasks raised."""
    return ExceptionGroup("unhandled errors in a TaskGroup", list(failures))


async def test_a_transient_failure_is_retried_and_then_succeeds() -> None:
    """One rate limit, then the second attempt's result."""
    outcomes: list[object] = [a_rate_limit(), "ruling"]

    async def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)

    result = await with_retry(operation, sleep=RecordedSleep())

    assert result == "ruling"
    assert outcomes == []


async def test_the_last_attempts_failure_propagates() -> None:
    """Three transient failures means three attempts and the third raised."""
    attempts: list[Exception] = []

    async def operation() -> str:
        failure = a_rate_limit()
        attempts.append(failure)
        raise failure

    with pytest.raises(openai.RateLimitError) as raised:
        await with_retry(operation, sleep=RecordedSleep())

    assert len(attempts) == 3
    assert raised.value is attempts[-1]


async def test_a_non_transient_failure_raises_on_the_first_attempt() -> None:
    """A bad request is not worth repeating, so the operation runs once."""
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise openai_status_error(openai.BadRequestError, 400)

    with pytest.raises(openai.BadRequestError):
        await with_retry(operation, sleep=RecordedSleep())

    assert calls == 1


async def test_an_authentication_failure_is_not_retried() -> None:
    """A rejected key stays rejected however long the wrapper waits."""
    calls = 0
    sleep = RecordedSleep()

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise openai_status_error(openai.AuthenticationError, 401)

    with pytest.raises(openai.AuthenticationError):
        await with_retry(operation, sleep=sleep)

    assert calls == 1
    assert sleep.delays == []


async def test_the_delays_grow_by_the_backoff_formula() -> None:
    """`base_delay * 2 ** (attempt - 1) * (1 + jitter())`, for a fixed jitter."""
    sleep = RecordedSleep()

    async def operation() -> str:
        raise a_rate_limit()

    with pytest.raises(openai.RateLimitError):
        await with_retry(
            operation, attempts=4, base_delay=2.0, sleep=sleep, jitter=lambda: 0.5
        )

    assert sleep.delays == [3.0, 6.0, 12.0]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(openai_status_error(openai.RateLimitError, 429), id="rate-limit"),
        pytest.param(
            openai_status_error(openai.InternalServerError, 500), id="server-error"
        ),
        pytest.param(
            openai_connection_error(openai.APIConnectionError), id="connection"
        ),
        pytest.param(openai_connection_error(openai.APITimeoutError), id="timeout"),
        pytest.param(httpx.ConnectError("refused"), id="httpx-transport"),
        pytest.param(StatusCodeError(429), id="status-code-429"),
        pytest.param(StatusCodeError(503), id="status-code-503"),
        pytest.param(StatusAttributeError(503), id="status-503"),
        pytest.param(ResponseStatusError(503), id="response-status-503"),
        pytest.param(grouped(StatusCodeError(503)), id="grouped-503"),
        pytest.param(grouped(grouped(StatusCodeError(503))), id="nested-503"),
        pytest.param(
            grouped(ValueError("unrelated"), StatusCodeError(503)), id="grouped-partly"
        ),
    ],
)
def test_transient_failures_are_recognised(failure: Exception) -> None:
    """Every kind the wrapper is meant to retry."""
    assert is_transient(failure) is True


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(
            openai_status_error(openai.BadRequestError, 400), id="bad-request"
        ),
        pytest.param(StatusCodeError(400), id="status-code-400"),
        pytest.param(
            openai_status_error(openai.AuthenticationError, 401), id="rejected-key"
        ),
        pytest.param(
            openai_status_error(openai.PermissionDeniedError, 403), id="forbidden"
        ),
        pytest.param(StatusCodeError(401), id="status-code-401"),
        pytest.param(ValueError("nothing to do with a status"), id="unrelated"),
        pytest.param(grouped(StatusCodeError(401)), id="grouped-401"),
        pytest.param(grouped(ValueError("unrelated")), id="grouped-unrelated"),
        pytest.param(
            grouped(StatusCodeError(503), StatusCodeError(401)), id="grouped-with-401"
        ),
    ],
)
def test_other_failures_are_not_transient(failure: Exception) -> None:
    """Repeating any of these changes nothing, so none of them is transient."""
    assert is_transient(failure) is False


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(
            openai_status_error(openai.AuthenticationError, 401), id="rejected-key"
        ),
        pytest.param(
            openai_status_error(openai.PermissionDeniedError, 403), id="forbidden"
        ),
        pytest.param(StatusCodeError(401), id="status-code-401"),
        pytest.param(StatusAttributeError(403), id="status-403"),
        pytest.param(ResponseStatusError(401), id="response-status-401"),
        pytest.param(grouped(StatusCodeError(401)), id="grouped-401"),
        pytest.param(grouped(grouped(StatusCodeError(403))), id="nested-403"),
        pytest.param(
            grouped(StatusCodeError(503), StatusCodeError(401)), id="grouped-with-401"
        ),
    ],
)
def test_rejected_credentials_are_recognised(failure: Exception) -> None:
    """Every shape a rejection arrives in, across the three status attributes."""
    assert is_authentication_failure(failure) is True


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(openai_status_error(openai.RateLimitError, 429), id="rate-limit"),
        pytest.param(StatusCodeError(500), id="status-code-500"),
        pytest.param(ValueError("nothing to do with a status"), id="unrelated"),
        pytest.param(grouped(StatusCodeError(500)), id="grouped-500"),
        pytest.param(grouped(grouped(a_rate_limit())), id="nested-rate-limit"),
    ],
)
def test_other_failures_are_not_rejections(failure: Exception) -> None:
    """The two predicates stay disjoint, and neither claims a plain failure."""
    assert is_authentication_failure(failure) is False
