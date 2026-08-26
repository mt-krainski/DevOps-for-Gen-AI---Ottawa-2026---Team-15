"""Environment to configuration: what one run holds, read once at start-up."""

import os
from dataclasses import dataclass

from fact_checker.errors import CheckError, ErrorCode

# The statement timeout guards against a hang, never against normal work. This
# is the time one tool call may reasonably take, so the budget multiplied by it
# is the floor the timeout has to clear.
MIN_SECONDS_PER_TOOL_CALL = 24

DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MCP_ENDPOINT = "https://mcp.brightdata.com/mcp"
DEFAULT_CONCURRENCY = 8
DEFAULT_TOOL_CALL_BUDGET = 10
DEFAULT_STATEMENT_TIMEOUT_SECONDS = 240
DEFAULT_SCRAPE_CHAR_LIMIT = 100000


@dataclass(frozen=True, repr=False)
class BrightDataConfig:
    """Where the MCP server is, and the token that opens it."""

    api_token: str
    base_endpoint: str

    def endpoint_url(self) -> str:
        """Return the endpoint the MCP client connects to, token included."""
        return self._endpoint_with(self.api_token)

    def redacted_endpoint_url(self) -> str:
        """Return the endpoint with `***` in the token's place, for anything logged."""
        return self._endpoint_with("***")

    def __repr__(self) -> str:
        """Show the redacted endpoint, so no stray `repr` can carry the token."""
        return f"BrightDataConfig(endpoint={self.redacted_endpoint_url()!r})"

    def _endpoint_with(self, token: str) -> str:
        return f"{self.base_endpoint}?token={token}"


@dataclass(frozen=True, repr=False)
class CheckerConfig:
    """One run's settings: the model, the tools, and the limits on both."""

    api_key: str
    model: str
    base_url: str
    bright_data: BrightDataConfig
    concurrency: int
    tool_call_budget: int
    statement_timeout_seconds: int
    scrape_char_limit: int

    def __repr__(self) -> str:
        """Show every setting except the OpenRouter key."""
        return (
            f"CheckerConfig(model={self.model!r}, base_url={self.base_url!r}, "
            f"bright_data={self.bright_data!r}, concurrency={self.concurrency}, "
            f"tool_call_budget={self.tool_call_budget}, "
            f"statement_timeout_seconds={self.statement_timeout_seconds}, "
            f"scrape_char_limit={self.scrape_char_limit})"
        )


def load_config() -> CheckerConfig:
    """Read one run's configuration from the environment.

    Returns:
        The settings the run holds, with a default in place of every variable
        the environment leaves unset.

    Raises:
        CheckError: A credential is unset or blank, a number does not parse, or
            a limit is one the run cannot work under.
    """
    tool_call_budget = _read_positive_int(
        "FACT_CHECKER_TOOL_CALL_BUDGET", DEFAULT_TOOL_CALL_BUDGET
    )
    statement_timeout_seconds = _read_int(
        "FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS", DEFAULT_STATEMENT_TIMEOUT_SECONDS
    )
    _reject_a_timeout_the_budget_outruns(tool_call_budget, statement_timeout_seconds)

    return CheckerConfig(
        api_key=_read_credential("OPENROUTER_API_KEY"),
        model=_read_text("OPENROUTER_MODEL", DEFAULT_MODEL),
        base_url=_read_text("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        bright_data=BrightDataConfig(
            api_token=_read_credential("BRIGHTDATA_API_TOKEN"),
            base_endpoint=_read_text("BRIGHTDATA_MCP_ENDPOINT", DEFAULT_MCP_ENDPOINT),
        ),
        concurrency=_read_positive_int("FACT_CHECKER_CONCURRENCY", DEFAULT_CONCURRENCY),
        tool_call_budget=tool_call_budget,
        statement_timeout_seconds=statement_timeout_seconds,
        scrape_char_limit=_read_int(
            "FACT_CHECKER_SCRAPE_CHAR_LIMIT", DEFAULT_SCRAPE_CHAR_LIMIT
        ),
    )


def _read_set_value(name: str) -> str | None:
    # `.env.example` ships every name with an empty value, so a part-filled
    # `.env` reaches here with blanks that mean "unset".
    raw = os.environ.get(name, "").strip()
    return raw or None


def _read_credential(name: str) -> str:
    value = _read_set_value(name)
    if value is None:
        raise CheckError(
            ErrorCode.MISSING_CREDENTIAL,
            f"{name} is unset or blank; set it in the environment or in .env",
        )
    return value


def _read_text(name: str, default: str) -> str:
    return _read_set_value(name) or default


def _read_int(name: str, default: int) -> int:
    raw = _read_set_value(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise CheckError(
            ErrorCode.INVALID_INPUT,
            f"{name} must be a whole number, got {raw!r}",
        ) from exc


def _read_positive_int(name: str, default: int) -> int:
    value = _read_int(name, default)
    if value < 1:
        raise CheckError(
            ErrorCode.INVALID_INPUT,
            f"{name} must be at least 1, got {value}",
        )
    return value


def _reject_a_timeout_the_budget_outruns(budget: int, timeout_seconds: int) -> None:
    floor = budget * MIN_SECONDS_PER_TOOL_CALL
    if timeout_seconds < floor:
        raise CheckError(
            ErrorCode.INVALID_INPUT,
            f"FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS is {timeout_seconds}, under the "
            f"{floor} seconds a budget of {budget} tool calls can spend at "
            f"{MIN_SECONDS_PER_TOOL_CALL} seconds each; raise the timeout or lower "
            f"the budget",
        )
