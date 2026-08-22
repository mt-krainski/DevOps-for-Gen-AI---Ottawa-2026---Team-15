"""The failures this package raises."""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Every code this package writes, batch-level and per-statement alike."""

    INVALID_INPUT = "INVALID_INPUT"
    MISSING_API_KEY = "MISSING_API_KEY"
    AUTH_ERROR = "AUTH_ERROR"
    LLM_ERROR = "LLM_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    PARSE_ERROR = "PARSE_ERROR"
    IO_ERROR = "IO_ERROR"


class ClassifierError(Exception):
    """Raised for batch-level failures: nothing partial is returned."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Carry the code and the message the caller reports."""
        super().__init__(message)
        self.code = code
        self.message = message


# The two below name a role rather than a category, so neither takes the `Error`
# suffix N818 asks for: each is a control-flow signal between the classifier and
# the service, and the pair reads as one distinction at the call site.
class ClassificationFailure(Exception):  # noqa: N818 — see the note above
    """A per-statement failure: isolated onto that statement's `error` field."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Carry the code and the message that statement's `error` field takes."""
        super().__init__(message)
        self.code = code
        self.message = message


class AuthenticationFailure(Exception):  # noqa: N818 — see the note above
    """Invalid credentials: aborts the whole batch, not just this statement."""

    def __init__(self, message: str) -> None:
        """Carry the message, under the one code a rejected credential can have."""
        super().__init__(message)
        self.code = ErrorCode.AUTH_ERROR
        self.message = message
