from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_API_KEY = "MISSING_API_KEY"
    AUTH_ERROR = "AUTH_ERROR"
    LLM_ERROR = "LLM_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    PARSE_ERROR = "PARSE_ERROR"


class ClassifierError(Exception):
    """Raised for batch-level failures: nothing partial is returned."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
