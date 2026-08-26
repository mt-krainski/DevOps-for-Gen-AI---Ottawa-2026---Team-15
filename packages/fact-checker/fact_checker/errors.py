"""The failures this package raises."""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Every code this package writes, run-level and per-statement alike."""

    INVALID_INPUT = "INVALID_INPUT"
    MISSING_CREDENTIAL = "MISSING_CREDENTIAL"
    AUTH_ERROR = "AUTH_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    TIMEOUT = "TIMEOUT"
    PARSE_ERROR = "PARSE_ERROR"
    IO_ERROR = "IO_ERROR"


class CheckError(Exception):
    """Raised for run-level failures: nothing partial is returned."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Carry the code and the message the caller reports."""
        super().__init__(message)
        self.code = code
        self.message = message


# The two below name a role rather than a category, so neither takes the `Error`
# suffix N818 asks for: each is a control-flow signal between the agent and the
# service, and the pair reads as one distinction at the call site.
class StatementFailure(Exception):  # noqa: N818 — see the note above
    """A per-statement failure: isolated onto that statement's `error` field."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Carry the code and the message that statement's `error` field takes."""
        super().__init__(message)
        self.code = code
        self.message = message


class AuthenticationFailure(Exception):  # noqa: N818 — see the note above
    """Invalid credentials: ends the run, because every statement would fail alike."""

    def __init__(self, message: str) -> None:
        """Carry the message, under the one code a rejected credential can have."""
        super().__init__(message)
        self.code = ErrorCode.AUTH_ERROR
        self.message = message
