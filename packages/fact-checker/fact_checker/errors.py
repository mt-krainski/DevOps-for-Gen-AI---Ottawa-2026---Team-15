"""The failures this package raises, and the bound on what a message quotes."""

from enum import StrEnum

# A failure message quotes what came back, and that message becomes the
# statement's published `error`. What came back can be a base64 image block or a
# whole page, so it is cut to this many characters first.
MAX_REPR_CHARACTERS = 300


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
    """A per-statement failure: isolated onto that statement's `error` field.

    `tool_calls_used` is what the statement had spent when it failed, which the
    run reports beside the outcome. A site that raises without holding the
    running count leaves it `None`, and the checking loop fills it in on the way
    out. `None` therefore means unknown, and never zero.
    """

    def __init__(
        self, code: ErrorCode, message: str, tool_calls_used: int | None = None
    ) -> None:
        """Carry the code, the message, and the calls spent where they are known."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.tool_calls_used = tool_calls_used


class AuthenticationFailure(Exception):  # noqa: N818 — see the note above
    """Invalid credentials: ends the run, because every statement would fail alike."""

    def __init__(self, message: str) -> None:
        """Carry the message, under the one code a rejected credential can have."""
        super().__init__(message)
        self.code = ErrorCode.AUTH_ERROR
        self.message = message


def bounded_repr(value: object) -> str:
    """Return `value`'s `repr`, cut to `MAX_REPR_CHARACTERS` and marked where cut."""
    shown = repr(value)
    if len(shown) <= MAX_REPR_CHARACTERS:
        return shown
    return f"{shown[:MAX_REPR_CHARACTERS]}..."
