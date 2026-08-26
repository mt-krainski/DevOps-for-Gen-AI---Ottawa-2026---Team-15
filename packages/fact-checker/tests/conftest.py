"""The fakes and the fixtures the test files in this package share."""

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from fact_checker.config import (
    DEFAULT_SCRAPE_CHAR_LIMIT,
    BrightDataConfig,
    CheckerConfig,
)

CONFIGURED_VARIABLES = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
    "BRIGHTDATA_API_TOKEN",
    "BRIGHTDATA_MCP_ENDPOINT",
    "FACT_CHECKER_CONCURRENCY",
    "FACT_CHECKER_TOOL_CALL_BUDGET",
    "FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS",
    "FACT_CHECKER_SCRAPE_CHAR_LIMIT",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Unset every variable this package reads, so no test inherits a real one."""
    for name in CONFIGURED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class StatusCodeError(Exception):
    """An error that reports its HTTP status the way most SDK errors do."""

    def __init__(self, status_code: int) -> None:
        """Carry the status under the `status_code` attribute."""
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class StatusAttributeError(Exception):
    """An error that spells its status `status` rather than `status_code`."""

    def __init__(self, status: int) -> None:
        """Carry the status under the `status` attribute."""
        super().__init__(f"HTTP {status}")
        self.status = status


class ResponseStatusError(Exception):
    """An error that reports its status only through a nested response."""

    def __init__(self, status_code: int) -> None:
        """Carry the status under `response.status_code`."""
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


def openai_status_error(kind: type[openai.APIStatusError], status: int) -> Exception:
    """Build an OpenAI status error without touching the network."""
    request = httpx.Request("POST", "https://openrouter.invalid/api/v1/chat")
    return kind("boom", response=httpx.Response(status, request=request), body=None)


def openai_connection_error(kind: type[openai.APIConnectionError]) -> Exception:
    """Build an OpenAI connection error without touching the network."""
    request = httpx.Request("POST", "https://openrouter.invalid/api/v1/chat")
    if kind is openai.APITimeoutError:
        return kind(request=request)
    return kind(message="boom", request=request)


class FakeTool:
    """Stands in for one Bright Data `BaseTool`, recording what it was asked."""

    def __init__(self, name: str, respond: Callable[[dict[str, Any]], object]) -> None:
        """Take the tool's name and the function deciding each invocation."""
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self._respond = respond

    async def ainvoke(self, arguments: dict[str, Any]) -> object:
        """Record the arguments, then return or raise what `respond` decided."""
        self.calls.append(arguments)
        outcome = self._respond(arguments)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeMCPClient:
    """Stands in for `MultiServerMCPClient`: one fixed tool list, no session."""

    def __init__(
        self, tools: Sequence[FakeTool], *, failure: Exception | None = None
    ) -> None:
        """Take the tools the server offers, or the failure `get_tools` raises."""
        self._tools = list(tools)
        self._failure = failure

    async def get_tools(self) -> list[FakeTool]:
        """Return the offered tools, or raise the connection failure."""
        if self._failure is not None:
            raise self._failure
        return list(self._tools)


def always(value: object) -> Callable[[dict[str, Any]], object]:
    """Return a `FakeTool` responder giving the same outcome every time."""

    def respond(_arguments: dict[str, Any]) -> object:
        return value

    return respond


BRIGHT_DATA_CREDENTIAL = "bd-must-never-be-logged"
BRIGHT_DATA_ENDPOINT = "https://mcp.brightdata.invalid/mcp"


def make_config(*, scrape_char_limit: int = DEFAULT_SCRAPE_CHAR_LIMIT) -> CheckerConfig:
    """Build one run's configuration without reading the environment."""
    return CheckerConfig(
        api_key="sk-openrouter-fake",
        model="google/gemma-4-31b-it",
        base_url="https://openrouter.invalid/api/v1",
        bright_data=BrightDataConfig(
            api_token=BRIGHT_DATA_CREDENTIAL, base_endpoint=BRIGHT_DATA_ENDPOINT
        ),
        concurrency=8,
        tool_call_budget=10,
        statement_timeout_seconds=240,
        scrape_char_limit=scrape_char_limit,
    )
