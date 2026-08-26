"""The Bright Data MCP connection, the two tools, and the layers round a call."""

import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol

from langchain_core.tools import BaseTool, ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient

from fact_checker.cache import RunCache
from fact_checker.config import CheckerConfig
from fact_checker.errors import (
    AuthenticationFailure,
    CheckError,
    ErrorCode,
    StatementFailure,
)
from fact_checker.retry import is_authentication_failure, with_retry

SEARCH_ENGINE = "search_engine"
SCRAPE_AS_MARKDOWN = "scrape_as_markdown"
SELECTED_TOOL_NAMES = (SEARCH_ENGINE, SCRAPE_AS_MARKDOWN)

# What a tool returned is shown in the failure saying it held no text, and that
# message becomes the statement's published `error`. A content-block list can
# carry base64 image data, so the whole blob cannot go in it.
MAX_REPR_CHARACTERS = 300

_SERVER_NAME = "brightdata"
_TARGET_ARGUMENT = {SEARCH_ENGINE: "query", SCRAPE_AS_MARKDOWN: "url"}

logger = logging.getLogger(__name__)


class MCPClient(Protocol):
    """The part of `MultiServerMCPClient` this package uses."""

    async def get_tools(self) -> list[BaseTool]:
        """Return every tool the connected servers offer."""
        ...


def build_bright_data_client(endpoint_url: str) -> MultiServerMCPClient:
    """Build the MCP client for the Bright Data hosted server.

    Args:
        endpoint_url: The server's URL, with the token in its query string.

    Returns:
        A client that raises a `ToolException` where a tool reports a failure.
        Under the adapter's default that failure would instead come back as
        content, and every layer above would read it as evidence.
    """
    return MultiServerMCPClient(
        {_SERVER_NAME: {"transport": "http", "url": endpoint_url}},
        handle_tool_errors=False,
    )


class Toolkit:
    """The two Bright Data tools, and what one run has already fetched."""

    def __init__(
        self, tools: Sequence[BaseTool], config: CheckerConfig, cache: RunCache
    ) -> None:
        """Take the selected tools, the limits they run under, and the cache."""
        self.bound_tools = list(tools)
        self.searches = 0
        self._by_name = {tool.name: tool for tool in tools}
        self._config = config
        self._cache = cache

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and return what it fetched, as text the agent can read.

        Args:
            name: One of the two selected tool names.
            arguments: What the model filled in from that tool's own schema.

        Returns:
            The fetched text, cut to the scrape ceiling where it is a page.

        Raises:
            StatementFailure: The tool is not one of the two, the arguments are
                unusable, the server reported a failure, or what came back holds
                no text.
            AuthenticationFailure: The Bright Data token was rejected.
        """
        tool = self._by_name.get(name)
        if tool is None:
            raise StatementFailure(
                ErrorCode.TOOL_ERROR,
                f"{name!r} is not a tool this run offers; it has "
                f"{', '.join(self._by_name)}",
            )

        reached_the_server = False

        async def fetch() -> str:
            nonlocal reached_the_server
            reached_the_server = True
            return await with_retry(lambda: self._invoke(tool, name, arguments))

        result = await self._cache.get_or_call(_cache_key(name, arguments), fetch)
        _log_call(name, arguments, result, cached=not reached_the_server)
        return result

    async def _invoke(
        self, tool: BaseTool, name: str, arguments: dict[str, Any]
    ) -> str:
        if name == SEARCH_ENGINE:
            self.searches += 1
        token = self._config.bright_data.api_token
        try:
            returned = await tool.ainvoke(arguments)
        except ToolException as exc:
            reported = _without_the_token(str(exc), token)
            raise StatementFailure(ErrorCode.TOOL_ERROR, reported) from exc
        except Exception as exc:
            if is_authentication_failure(exc):
                rejection = _without_the_token(str(exc), token)
                raise AuthenticationFailure(rejection) from exc
            raise
        text = _as_text(returned, name)
        if name == SCRAPE_AS_MARKDOWN:
            return self._within_the_ceiling(text)
        return text

    def _within_the_ceiling(self, page: str) -> str:
        limit = self._config.scrape_char_limit
        if len(page) <= limit:
            return page
        return f"{page[:limit]}\n\n[truncated at {limit} characters]"


@asynccontextmanager
async def open_toolkit(
    config: CheckerConfig,
    cache: RunCache,
    *,
    client_factory: Callable[[str], MCPClient] = build_bright_data_client,
) -> AsyncIterator[Toolkit]:
    """Connect to the Bright Data server and hand back the two tools it offers.

    Args:
        config: The run's settings, holding the endpoint and the scrape ceiling.
        cache: What this run has already fetched.
        client_factory: How to build the MCP client from the endpoint URL.

    Yields:
        The toolkit the agent calls.

    Raises:
        CheckError: The connection failed, the token was rejected, or the server
            does not offer one of the two tools.
    """
    client = client_factory(config.bright_data.endpoint_url())
    try:
        offered = await client.get_tools()
    except Exception as exc:
        code = (
            ErrorCode.AUTH_ERROR
            if is_authentication_failure(exc)
            else ErrorCode.TOOL_ERROR
        )
        raise CheckError(
            code,
            _without_the_token(
                f"could not load the tools at "
                f"{config.bright_data.redacted_endpoint_url()}: {exc}",
                config.bright_data.api_token,
            ),
        ) from exc
    yield Toolkit(_select(offered), config, cache)


def _select(offered: Sequence[BaseTool]) -> list[BaseTool]:
    by_name = {tool.name: tool for tool in offered}
    missing = [name for name in SELECTED_TOOL_NAMES if name not in by_name]
    if missing:
        raise CheckError(
            ErrorCode.TOOL_ERROR,
            f"the Bright Data server does not offer {', '.join(missing)}; "
            f"it offers {', '.join(by_name) or 'no tools at all'}",
        )
    return [by_name[name] for name in SELECTED_TOOL_NAMES]


def _cache_key(name: str, arguments: dict[str, Any]) -> str:
    if name == SEARCH_ENGINE:
        return f"search:{json.dumps(arguments, sort_keys=True)}"
    url = arguments.get("url")
    if not isinstance(url, str):
        raise StatementFailure(
            ErrorCode.TOOL_ERROR,
            f"{SCRAPE_AS_MARKDOWN} needs a url; it was given "
            f"{', '.join(arguments) or 'no arguments at all'}",
        )
    return f"scrape:{url}"


def _without_the_token(text: str, token: str) -> str:
    # An upstream failure quotes the request URL it was given, and the Bright
    # Data token rides in that URL's query string. A blank token is left alone,
    # because replacing an empty string would match between every character.
    if not token:
        return text
    return text.replace(token, "***")


def _as_text(returned: object, name: str) -> str:
    if isinstance(returned, str):
        return returned
    if isinstance(returned, list):
        text = "\n\n".join(
            block["text"]
            for block in returned
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if text:
            return text
    raise StatementFailure(
        ErrorCode.TOOL_ERROR,
        f"{name} returned no text to read: {type(returned).__name__} "
        f"{_bounded_repr(returned)}",
    )


def _bounded_repr(returned: object) -> str:
    shown = repr(returned)
    if len(shown) <= MAX_REPR_CHARACTERS:
        return shown
    return f"{shown[:MAX_REPR_CHARACTERS]}..."


def _log_call(
    name: str, arguments: dict[str, Any], result: str, *, cached: bool
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    target = arguments.get(_TARGET_ARGUMENT[name])
    if cached:
        logger.debug("cache hit: %s %r returned %s", name, target, _size_of(result))
    else:
        logger.debug("%s %r returned %s", name, target, _size_of(result))


def _size_of(result: str) -> str:
    try:
        parsed = json.loads(result)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        return f"{len(parsed)} entries"
    return f"{len(result)} characters"
