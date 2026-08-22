"""Tests for the exceptions in `factchecker.errors`."""

from factchecker.errors import (
    AuthenticationFailed,
    InputValidationError,
    McpCallError,
)


def test_input_validation_error_carries_its_message() -> None:
    """The rejection reaches the caller with the detail that caused it."""
    error = InputValidationError("statement 3 has no surroundingContext")

    assert isinstance(error, Exception)
    assert str(error) == "statement 3 has no surroundingContext"


def test_authentication_failed_carries_its_message() -> None:
    """A rejected credential reaches the caller with the service that rejected it."""
    error = AuthenticationFailed("openrouter rejected the api key")

    assert isinstance(error, Exception)
    assert str(error) == "openrouter rejected the api key"


def test_authentication_failed_is_not_an_input_validation_error() -> None:
    """A rejected credential ends the run; a rejected payload is a separate path."""
    assert not isinstance(AuthenticationFailed("rejected"), InputValidationError)
    assert not isinstance(InputValidationError("bad payload"), AuthenticationFailed)


def test_mcp_call_error_carries_its_message() -> None:
    """The message reaches the output payload, so a person reads it."""
    error = McpCallError("the MCP server at https://example.test returned 422")

    assert isinstance(error, Exception)
    assert str(error) == "the MCP server at https://example.test returned 422"


def test_mcp_call_error_is_not_an_authentication_failure() -> None:
    """One failed tool call fails its own statement; it does not end the run."""
    assert not isinstance(McpCallError("returned 422"), AuthenticationFailed)
