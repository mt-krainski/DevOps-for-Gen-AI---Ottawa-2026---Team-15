"""What a run reads from the environment, and how it holds the two credentials."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import quote

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
BRIGHTDATA_MCP_URL = "https://mcp.brightdata.com/mcp"


class ConfigurationError(Exception):
    """A required setting is absent, or a setting will not parse as its type."""


class McpEndpoint:
    """The Bright Data MCP endpoint, printed with its token replaced.

    Bright Data authenticates by a query parameter rather than by a header, so the
    endpoint URL is itself a credential. Holding the token here, and building the
    URL only where a caller asks for it by name, makes redaction a property of the
    value: no log record and no exception message can leak the token by carrying
    the endpoint, however the endpoint reaches it.
    """

    def __init__(self, token: str) -> None:
        """Hold the token the endpoint URL is built around.

        Args:
            token: The Bright Data API token.
        """
        self._token = token

    def unredacted_url(self) -> str:
        """Return the endpoint URL with the token in it.

        Returns:
            The URL to open the MCP session against. Nothing may log this value.
        """
        return self._url(quote(self._token, safe=""))

    def __str__(self) -> str:
        """Return the endpoint URL, with `REDACTED` where the token would be."""
        return self._url("REDACTED")

    def __repr__(self) -> str:
        """Return what `__str__` returns, so no conversion reveals more than another."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Return whether `other` is an endpoint around the same token.

        Comparison reads the token itself rather than a built URL, so equality
        gives the token no way out: an assertion on two unequal endpoints prints
        the redacted form of each.
        """
        if not isinstance(other, McpEndpoint):
            return NotImplemented
        return self._token == other._token

    def __hash__(self) -> int:
        """Return a hash of the token, so equal endpoints hash alike."""
        return hash(self._token)

    def _url(self, token: str) -> str:
        """Build the endpoint URL around whatever stands in the token's place."""
        return f"{BRIGHTDATA_MCP_URL}?token={token}"


@dataclass(frozen=True)
class Settings:
    """Everything one run reads from the environment.

    The OpenRouter key is kept out of the generated `repr`. The test task runs
    `pytest --showlocals`, which prints any `Settings` sitting in a failed test's
    frame, and a key printed into a continuous-integration log is a key to rotate.
    """

    openrouter_api_key: str = field(repr=False)
    model: str
    mcp_endpoint: McpEndpoint
    tool_call_budget: int
    page_character_ceiling: int
    concurrency: int
    statement_timeout_seconds: float
    retry_attempts: int


def load_settings(env: Mapping[str, str]) -> Settings:
    """Read one run's settings out of a mapping of environment variables.

    The environment arrives as an argument rather than being read from the process,
    so a caller decides what this sees and a test supplies its own.

    The two required variables carry no prefix, because they name a vendor's
    credential. The six overrides carry `FACTCHECKER_`, because this package shares
    a repository with others and a bare `CONCURRENCY` would collide.

    Args:
        env: The environment to read.

    Returns:
        The settings, with a default in place of every override the environment
        leaves unset or empty.

    Raises:
        ConfigurationError: A required variable is unset or empty, an override
            will not parse as its type, or a numeric override is not above zero.
    """
    return Settings(
        openrouter_api_key=_required(env, "OPENROUTER_API_KEY"),
        model=_parsed(env, "FACTCHECKER_MODEL", str, "google/gemma-4-31b-it"),
        mcp_endpoint=McpEndpoint(_required(env, "BRIGHTDATA_API_TOKEN")),
        tool_call_budget=_positive(env, "FACTCHECKER_TOOL_CALL_BUDGET", int, 10),
        page_character_ceiling=_positive(
            env, "FACTCHECKER_PAGE_CHARACTER_CEILING", int, 100000
        ),
        concurrency=_positive(env, "FACTCHECKER_CONCURRENCY", int, 8),
        statement_timeout_seconds=_positive(
            env, "FACTCHECKER_STATEMENT_TIMEOUT_SECONDS", float, 240.0
        ),
        retry_attempts=_positive(env, "FACTCHECKER_RETRY_ATTEMPTS", int, 3),
    )


def build_model(settings: Settings) -> BaseChatModel:
    """Build the chat client this run reaches OpenRouter through.

    OpenRouter speaks the OpenAI API, so the OpenAI client reaches it by base URL
    alone. It also accepts `HTTP-Referer` and `X-OpenRouter-Title` headers for its
    public rankings; both are optional and this package sends neither.

    Args:
        settings: The settings whose model slug and key the client carries.

    Returns:
        A chat model pointed at OpenRouter.
    """
    return ChatOpenAI(
        model=settings.model,
        base_url=OPENROUTER_BASE_URL,
        api_key=SecretStr(settings.openrouter_api_key),
    )


def _required(env: Mapping[str, str], name: str) -> str:
    """Read a variable that has no default.

    Args:
        env: The environment to read.
        name: The variable to read.

    Returns:
        What the variable is set to.

    Raises:
        ConfigurationError: The variable is unset or empty.
    """
    written = env.get(name, "")
    if not written:
        raise ConfigurationError(f"{name} is not set, and it has no default")
    return written


def _parsed[T](env: Mapping[str, str], name: str, kind: type[T], default: T) -> T:
    """Read an override and parse it, or fall back where it is unset.

    Args:
        env: The environment to read.
        name: The variable to read.
        kind: The type the written value is parsed as.
        default: What applies where the variable is unset or empty.

    Returns:
        The parsed value, or the default.

    Raises:
        ConfigurationError: The written value will not parse as `kind`.
    """
    written = env.get(name, "")
    if not written:
        return default
    try:
        return kind(written)
    except ValueError as failure:
        raise ConfigurationError(
            f"{name} is set to {written!r}, which will not parse as {kind.__name__}"
        ) from failure


def _positive[T: (int, float)](
    env: Mapping[str, str], name: str, kind: type[T], default: T
) -> T:
    """Read a numeric override, and reject a value at or below zero.

    A concurrency of 0 is a semaphore no worker passes, and a timeout below zero
    expires before the work starts. Each fails without a diagnostic, so each is
    refused here, where the rejection can name the variable.

    Args:
        env: The environment to read.
        name: The variable to read.
        kind: The type the written value is parsed as.
        default: What applies where the variable is unset or empty.

    Returns:
        The parsed value, or the default.

    Raises:
        ConfigurationError: The written value will not parse as `kind`, or it
            parses to a value at or below zero.
    """
    value = _parsed(env, name, kind, default)
    if value <= 0:
        raise ConfigurationError(f"{name} is set to {value!r}, which is not above zero")
    return value
