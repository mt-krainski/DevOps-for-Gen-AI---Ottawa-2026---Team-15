"""Tests for the environment-to-configuration step in `fact_checker.config`."""

from dataclasses import FrozenInstanceError

import pytest

from fact_checker.config import (
    MIN_SECONDS_PER_TOOL_CALL,
    BrightDataConfig,
    CheckerConfig,
    load_config,
)
from fact_checker.errors import CheckError, ErrorCode

OPENROUTER_KEY = "sk-or-v1-not-a-real-key"
BRIGHT_DATA_TOKEN = "bd-not-a-real-token"  # noqa: S105 — a fixture, not a credential


def set_credentials(env: pytest.MonkeyPatch) -> None:
    """Set the two required credentials and nothing else."""
    env.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    env.setenv("BRIGHTDATA_API_TOKEN", BRIGHT_DATA_TOKEN)


def test_defaults_apply_when_only_the_credentials_are_set(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A run needs two credentials; every other value has a working default."""
    set_credentials(clean_env)

    config = load_config()

    assert config.api_key == OPENROUTER_KEY
    assert config.model == "google/gemma-4-31b-it"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.bright_data.api_token == BRIGHT_DATA_TOKEN
    assert config.bright_data.base_endpoint == "https://mcp.brightdata.com/mcp"
    assert config.concurrency == 8
    assert config.tool_call_budget == 10
    assert config.statement_timeout_seconds == 240
    assert config.scrape_char_limit == 100000


def test_the_environment_overrides_every_default(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """Each variable in the contract reaches the field it configures."""
    set_credentials(clean_env)
    clean_env.setenv("OPENROUTER_MODEL", "openai/gpt-5")
    clean_env.setenv("OPENROUTER_BASE_URL", "https://proxy.test/v1")
    clean_env.setenv("BRIGHTDATA_MCP_ENDPOINT", "https://proxy.test/mcp")
    clean_env.setenv("FACT_CHECKER_CONCURRENCY", "2")
    clean_env.setenv("FACT_CHECKER_TOOL_CALL_BUDGET", "4")
    clean_env.setenv("FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS", "300")
    clean_env.setenv("FACT_CHECKER_SCRAPE_CHAR_LIMIT", "500")

    config = load_config()

    assert config.model == "openai/gpt-5"
    assert config.base_url == "https://proxy.test/v1"
    assert config.bright_data.base_endpoint == "https://proxy.test/mcp"
    assert config.concurrency == 2
    assert config.tool_call_budget == 4
    assert config.statement_timeout_seconds == 300
    assert config.scrape_char_limit == 500


def test_a_missing_openrouter_key_names_the_variable(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The run refuses to start, and says which variable to set."""
    clean_env.setenv("BRIGHTDATA_API_TOKEN", BRIGHT_DATA_TOKEN)

    with pytest.raises(CheckError) as raised:
        load_config()

    assert raised.value.code is ErrorCode.MISSING_CREDENTIAL
    assert "OPENROUTER_API_KEY" in raised.value.message


def test_a_missing_bright_data_token_names_the_variable(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The second credential is required the same way the first is."""
    clean_env.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)

    with pytest.raises(CheckError) as raised:
        load_config()

    assert raised.value.code is ErrorCode.MISSING_CREDENTIAL
    assert "BRIGHTDATA_API_TOKEN" in raised.value.message


def test_a_blank_credential_counts_as_missing(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A variable set to whitespace is an unset variable that looks set."""
    set_credentials(clean_env)
    clean_env.setenv("OPENROUTER_API_KEY", "   ")

    with pytest.raises(CheckError) as raised:
        load_config()

    assert raised.value.code is ErrorCode.MISSING_CREDENTIAL
    assert "OPENROUTER_API_KEY" in raised.value.message


def test_a_number_that_does_not_parse_names_the_variable_and_the_value(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A typo in a numeric variable stops the run before any statement."""
    set_credentials(clean_env)
    clean_env.setenv("FACT_CHECKER_CONCURRENCY", "eight")

    with pytest.raises(CheckError) as raised:
        load_config()

    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert "FACT_CHECKER_CONCURRENCY" in raised.value.message
    assert "eight" in raised.value.message


def test_a_concurrency_of_zero_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    """A bound of zero would run nothing at all."""
    set_credentials(clean_env)
    clean_env.setenv("FACT_CHECKER_CONCURRENCY", "0")

    with pytest.raises(CheckError) as raised:
        load_config()

    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert "FACT_CHECKER_CONCURRENCY" in raised.value.message


def test_a_tool_call_budget_of_zero_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    """A budget of zero leaves the agent no way to look anything up."""
    set_credentials(clean_env)
    clean_env.setenv("FACT_CHECKER_TOOL_CALL_BUDGET", "0")

    with pytest.raises(CheckError) as raised:
        load_config()

    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert "FACT_CHECKER_TOOL_CALL_BUDGET" in raised.value.message


def test_a_timeout_below_the_budget_floor_names_all_three_numbers(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The timeout catches a hang, so it must sit above what the budget can spend."""
    set_credentials(clean_env)
    clean_env.setenv("FACT_CHECKER_TOOL_CALL_BUDGET", "10")
    clean_env.setenv("FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS", "239")

    with pytest.raises(CheckError) as raised:
        load_config()

    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert "10" in raised.value.message
    assert "239" in raised.value.message
    assert str(10 * MIN_SECONDS_PER_TOOL_CALL) in raised.value.message


def test_a_timeout_exactly_at_the_budget_floor_is_accepted(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The floor is what the budget can consume, not one second more."""
    set_credentials(clean_env)
    clean_env.setenv("FACT_CHECKER_TOOL_CALL_BUDGET", "5")
    clean_env.setenv(
        "FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS", str(5 * MIN_SECONDS_PER_TOOL_CALL)
    )

    config = load_config()

    assert config.statement_timeout_seconds == 5 * MIN_SECONDS_PER_TOOL_CALL


def test_the_endpoint_url_carries_the_token() -> None:
    """The MCP server reads the token from the query string, so the URL holds it."""
    bright_data = BrightDataConfig(
        api_token=BRIGHT_DATA_TOKEN,
        base_endpoint="https://mcp.brightdata.com/mcp",
    )

    assert bright_data.endpoint_url() == (
        f"https://mcp.brightdata.com/mcp?token={BRIGHT_DATA_TOKEN}"
    )


def test_the_redacted_url_and_the_repr_withhold_the_token() -> None:
    """Every form that could reach a log carries `***` in the token's place."""
    bright_data = BrightDataConfig(
        api_token=BRIGHT_DATA_TOKEN,
        base_endpoint="https://mcp.brightdata.com/mcp",
    )

    assert bright_data.redacted_endpoint_url() == (
        "https://mcp.brightdata.com/mcp?token=***"
    )
    assert BRIGHT_DATA_TOKEN not in repr(bright_data)
    assert "***" in repr(bright_data)


def test_the_checker_config_repr_withholds_both_credentials(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A configuration dumped into a log names its settings, never its secrets."""
    set_credentials(clean_env)

    config = load_config()

    assert OPENROUTER_KEY not in repr(config)
    assert BRIGHT_DATA_TOKEN not in repr(config)
    assert "google/gemma-4-31b-it" in repr(config)


def test_the_checker_config_is_frozen(clean_env: pytest.MonkeyPatch) -> None:
    """Configuration is read once at start-up and never edited under a run."""
    set_credentials(clean_env)

    config = load_config()

    with pytest.raises(FrozenInstanceError):
        config.concurrency = 2


def test_the_checker_config_can_be_built_without_the_environment() -> None:
    """The dataclass is the unit the rest of the package takes, however it is built."""
    config = CheckerConfig(
        api_key=OPENROUTER_KEY,
        model="openai/gpt-5",
        base_url="https://proxy.test/v1",
        bright_data=BrightDataConfig(
            api_token=BRIGHT_DATA_TOKEN,
            base_endpoint="https://proxy.test/mcp",
        ),
        concurrency=1,
        tool_call_budget=1,
        statement_timeout_seconds=24,
        scrape_char_limit=100,
    )

    assert config.bright_data.endpoint_url().endswith(BRIGHT_DATA_TOKEN)
