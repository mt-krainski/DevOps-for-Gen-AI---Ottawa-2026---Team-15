"""Tests for the failure vocabulary in `fact_checker.errors`."""

import pytest

from fact_checker.errors import (
    AuthenticationFailure,
    CheckError,
    ErrorCode,
    StatementFailure,
)


def test_a_run_level_failure_carries_its_code_and_message() -> None:
    """`CheckError` is what the command line turns into an exit code."""
    with pytest.raises(CheckError) as raised:
        raise CheckError(ErrorCode.IO_ERROR, "the output path is not writable")

    assert raised.value.code is ErrorCode.IO_ERROR
    assert raised.value.message == "the output path is not writable"
    assert str(raised.value) == "the output path is not writable"


def test_a_statement_failure_carries_the_code_its_error_field_takes() -> None:
    """One statement's failure travels to that statement's `error` field."""
    failure = StatementFailure(ErrorCode.TOOL_ERROR, "the search tool returned 500")

    assert failure.code is ErrorCode.TOOL_ERROR
    assert failure.message == "the search tool returned 500"


def test_an_authentication_failure_is_always_an_auth_error() -> None:
    """A rejected credential has one code, so the caller never chooses it."""
    failure = AuthenticationFailure("OpenRouter rejected the key")

    assert failure.code is ErrorCode.AUTH_ERROR
    assert failure.message == "OpenRouter rejected the key"


def test_every_code_serializes_as_its_own_name() -> None:
    """The wire carries the member name, because `ErrorCode` is a `StrEnum`."""
    assert [code.value for code in ErrorCode] == [
        "INVALID_INPUT",
        "MISSING_CREDENTIAL",
        "AUTH_ERROR",
        "AGENT_ERROR",
        "TOOL_ERROR",
        "TIMEOUT",
        "PARSE_ERROR",
        "IO_ERROR",
    ]
