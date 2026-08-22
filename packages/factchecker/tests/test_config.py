"""Tests for the settings loader and the redacting endpoint in `factchecker.config`."""

import logging

import pytest
from langchain_core.language_models import BaseChatModel

from factchecker.config import (
    BRIGHTDATA_MCP_URL,
    OPENROUTER_BASE_URL,
    ConfigurationError,
    McpEndpoint,
    Settings,
    build_model,
    load_settings,
)

OPENROUTER_CREDENTIAL = "sk-or-v1-0123456789abcdef"
BRIGHTDATA_CREDENTIAL = "brd-4a7f2e91c0"

REQUIRED_ONLY = {
    "OPENROUTER_API_KEY": OPENROUTER_CREDENTIAL,
    "BRIGHTDATA_API_TOKEN": BRIGHTDATA_CREDENTIAL,
}

OVERRIDDEN = REQUIRED_ONLY | {
    "FACTCHECKER_MODEL": "anthropic/claude-sonnet-5",
    "FACTCHECKER_TOOL_CALL_BUDGET": "4",
    "FACTCHECKER_PAGE_CHARACTER_CEILING": "12000",
    "FACTCHECKER_CONCURRENCY": "2",
    "FACTCHECKER_STATEMENT_TIMEOUT_SECONDS": "30.5",
    "FACTCHECKER_RETRY_ATTEMPTS": "1",
}

OVERRIDES = [name for name in OVERRIDDEN if name.startswith("FACTCHECKER_")]
NUMERIC_OVERRIDES = [name for name in OVERRIDES if name != "FACTCHECKER_MODEL"]

DEFAULT_TUNABLES = {
    "model": "google/gemma-4-31b-it",
    "tool_call_budget": 10,
    "page_character_ceiling": 100000,
    "concurrency": 8,
    "statement_timeout_seconds": 240.0,
    "retry_attempts": 3,
}


def _tunables(settings: Settings) -> dict[str, object]:
    """The six settings an environment variable may override, by field name."""
    return {
        "model": settings.model,
        "tool_call_budget": settings.tool_call_budget,
        "page_character_ceiling": settings.page_character_ceiling,
        "concurrency": settings.concurrency,
        "statement_timeout_seconds": settings.statement_timeout_seconds,
        "retry_attempts": settings.retry_attempts,
    }


def test_the_pinned_hosts_are_the_ones_this_package_documents() -> None:
    """Every other host assertion is built on these two, so pin them to literals."""
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert BRIGHTDATA_MCP_URL == "https://mcp.brightdata.com/mcp"


def test_the_required_variables_alone_produce_the_documented_defaults() -> None:
    """Two variables are enough, and every other setting falls back."""
    settings = load_settings(REQUIRED_ONLY)

    assert settings.openrouter_api_key == OPENROUTER_CREDENTIAL
    assert (
        settings.mcp_endpoint.unredacted_url()
        == f"{BRIGHTDATA_MCP_URL}?token={BRIGHTDATA_CREDENTIAL}"
    )
    assert _tunables(settings) == DEFAULT_TUNABLES


def test_every_override_is_read_and_parsed_as_its_type() -> None:
    """Each `FACTCHECKER_` variable reaches its field, as a number where it is one."""
    settings = load_settings(OVERRIDDEN)

    assert _tunables(settings) == {
        "model": "anthropic/claude-sonnet-5",
        "tool_call_budget": 4,
        "page_character_ceiling": 12000,
        "concurrency": 2,
        "statement_timeout_seconds": 30.5,
        "retry_attempts": 1,
    }


@pytest.mark.parametrize("name", list(REQUIRED_ONLY))
def test_a_missing_required_variable_is_named(name: str) -> None:
    """The rejection says which variable to go and set."""
    env = {
        written: value for written, value in REQUIRED_ONLY.items() if written != name
    }

    with pytest.raises(ConfigurationError, match=name):
        load_settings(env)


@pytest.mark.parametrize("name", list(REQUIRED_ONLY))
def test_an_empty_required_variable_is_named(name: str) -> None:
    """A copied `.env.example` left unfilled is rejected as plainly as an absent one."""
    with pytest.raises(ConfigurationError, match=name):
        load_settings(REQUIRED_ONLY | {name: ""})


@pytest.mark.parametrize("name", NUMERIC_OVERRIDES)
def test_an_override_that_will_not_parse_is_named(name: str) -> None:
    """A number written as prose is rejected, and the rejection names the variable."""
    with pytest.raises(ConfigurationError, match=name):
        load_settings(REQUIRED_ONLY | {name: "as many as it takes"})


@pytest.mark.parametrize("written", ["0", "-1"])
@pytest.mark.parametrize("name", NUMERIC_OVERRIDES)
def test_a_numeric_override_at_or_below_zero_is_named(name: str, written: str) -> None:
    """A concurrency of none hangs a run in silence, so no tunable may sink there."""
    with pytest.raises(ConfigurationError, match=name):
        load_settings(REQUIRED_ONLY | {name: written})


def test_an_empty_override_takes_its_default() -> None:
    """A variable written with no value after it counts as unset."""
    settings = load_settings(REQUIRED_ONLY | dict.fromkeys(OVERRIDES, ""))

    assert _tunables(settings) == DEFAULT_TUNABLES


def test_the_process_environment_is_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the mapping decides, so a test never has to touch process state."""
    monkeypatch.setenv("FACTCHECKER_CONCURRENCY", "99")
    monkeypatch.setenv("FACTCHECKER_MODEL", "openai/gpt-4o")

    assert _tunables(load_settings(REQUIRED_ONLY)) == DEFAULT_TUNABLES


def test_the_endpoint_hides_the_token_in_both_its_printed_forms() -> None:
    """`str` and `repr` alike return the URL with the token replaced."""
    endpoint = McpEndpoint(BRIGHTDATA_CREDENTIAL)

    assert str(endpoint) == f"{BRIGHTDATA_MCP_URL}?token=REDACTED"
    assert repr(endpoint) == str(endpoint)
    assert BRIGHTDATA_CREDENTIAL not in str(endpoint)
    assert BRIGHTDATA_CREDENTIAL not in repr(endpoint)


def test_the_endpoint_exposes_the_whole_url_through_its_named_accessor() -> None:
    """The one way to the token is a call a reader can see."""
    endpoint = McpEndpoint(BRIGHTDATA_CREDENTIAL)

    expected = f"{BRIGHTDATA_MCP_URL}?token={BRIGHTDATA_CREDENTIAL}"

    assert endpoint.unredacted_url() == expected


def test_endpoints_compare_by_the_token_they_hold() -> None:
    """Two endpoints around one token are one value, and they hash alike."""
    endpoint = McpEndpoint(BRIGHTDATA_CREDENTIAL)

    assert endpoint == McpEndpoint(BRIGHTDATA_CREDENTIAL)
    assert hash(endpoint) == hash(McpEndpoint(BRIGHTDATA_CREDENTIAL))
    assert endpoint != McpEndpoint("brd-0000000000")
    assert endpoint != BRIGHTDATA_MCP_URL


def test_two_settings_read_from_one_environment_are_equal() -> None:
    """Loading twice gives equal values, now the endpoint compares by token."""
    assert load_settings(OVERRIDDEN) == load_settings(OVERRIDDEN)


def test_no_log_record_carries_the_token(caplog: pytest.LogCaptureFixture) -> None:
    """Formatting an endpoint into a record, by either conversion, redacts it."""
    endpoint = McpEndpoint(BRIGHTDATA_CREDENTIAL)

    with caplog.at_level(logging.DEBUG, logger="factchecker.test"):
        logging.getLogger("factchecker.test").debug("%s then %r", endpoint, endpoint)

    assert BRIGHTDATA_CREDENTIAL not in caplog.text
    assert caplog.text.count("token=REDACTED") == 2


def test_a_settings_representation_carries_neither_credential() -> None:
    """`pytest --showlocals` prints this object, so it must hold nothing to rotate."""
    written = repr(load_settings(REQUIRED_ONLY))

    assert OPENROUTER_CREDENTIAL not in written
    assert BRIGHTDATA_CREDENTIAL not in written


def test_build_model_returns_a_chat_model_pointed_at_openrouter() -> None:
    """The client carries the configured slug, the OpenRouter host, and the key."""
    model = build_model(load_settings(OVERRIDDEN))

    assert isinstance(model, BaseChatModel)
    assert model.model_name == "anthropic/claude-sonnet-5"
    assert model.openai_api_base == OPENROUTER_BASE_URL
    assert model.openai_api_key.get_secret_value() == OPENROUTER_CREDENTIAL
