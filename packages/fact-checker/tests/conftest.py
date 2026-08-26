"""Fixtures the test files in this package share."""

import pytest

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
