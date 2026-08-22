"""The two Bright Data tools a run spends its budget on, and the guards around them."""

import logging
from collections.abc import Awaitable, Callable, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from factchecker.cache import RunCache
from factchecker.config import ConfigurationError, McpEndpoint, Settings
from factchecker.resilience import with_retry

logger = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_engine"
PAGE_TOOL_NAME = "scrape_as_markdown"
TRUNCATION_MARKER = "\n\n[truncated: page exceeded the character ceiling]"

_WANTED = (SEARCH_TOOL_NAME, PAGE_TOOL_NAME)
_SERVER_NAME = "brightdata"
_BLOCK_SEPARATOR = "\n\n"


async def load_tools(
    endpoint: McpEndpoint,
) -> tuple[list[BaseTool], Callable[[], Awaitable[None]]]:
    """Open the Bright Data MCP server and take the two tools this package uses.

    The release callable comes back beside the tools because a caller cannot close a
    list. What it releases is described on `_release`.

    Args:
        endpoint: The Bright Data endpoint, which is itself the credential.

    Returns:
        The search tool and the page tool, in that order, and the callable that
        releases what the connection holds.

    Raises:
        ConfigurationError: The server could not be reached, or its catalogue is
            missing one of the two tools. Either way the connection is released
            before this is raised, and the message names the endpoint redacted.
    """
    client = MultiServerMCPClient(
        {
            _SERVER_NAME: {
                "transport": "streamable_http",
                "url": endpoint.unredacted_url(),
            }
        }
    )
    try:
        offered = {tool.name: tool for tool in await client.get_tools()}
    except Exception as failure:  # noqa: BLE001 — nothing here may escape unredacted
        await _release()
        # The cause is dropped rather than chained. It arrives from `httpx`, whose
        # message names the request URL, and the token travels inside that URL. This
        # is the one thing `McpEndpoint` cannot keep out of another library's text.
        raise ConfigurationError(
            f"the MCP server at {endpoint} could not be reached: "
            f"{type(failure).__name__}"
        ) from None
    missing = [name for name in _WANTED if name not in offered]
    if missing:
        await _release()
        raise ConfigurationError(
            f"the MCP server at {endpoint} offers no {' and no '.join(missing)}; "
            f"it offers {', '.join(offered) or 'nothing at all'}"
        )
    return [offered[name] for name in _WANTED], _release


def instrument(
    tools: Sequence[BaseTool],
    cache: RunCache,
    settings: Settings,
    sleep: Callable[[float], Awaitable[None]],
) -> list[BaseTool]:
    """Give each tool the run's cache, its retry policy, and the page ceiling.

    The wrappers hold no per-statement state, which is what lets one instrumented set
    of tools serve every statement of a run at once. Nothing here counts anything: a
    count kept here would be shared by every statement running at that moment, and a
    statement's search count is its own.

    Args:
        tools: The tools `load_tools` returned.
        cache: The run's store of fetched material.
        settings: Read for `retry_attempts` and `page_character_ceiling`.
        sleep: Waits the number of seconds it is given, between retries.

    Returns:
        One instrumented tool per tool given, keeping each tool's name, description
        and argument schema so that the agent chooses between them as before.
    """
    return [_instrumented(tool, cache, settings, sleep) for tool in tools]


async def _release() -> None:
    """Release what the MCP connection holds, which in this version is nothing.

    Read from `langchain-mcp-adapters` 0.3.2 rather than recalled. Its
    `MultiServerMCPClient` refuses to be a context manager — `__aenter__` raises
    `NotImplementedError` — and offers no close of any kind. `get_tools` opens a
    session, lists the catalogue and closes it again, and each tool it returns opens
    a fresh session for every call and closes that too. So nothing is held between
    calls and nothing is left to release. This exists so that a version which does
    hold something can be released without any caller changing shape.
    """


def _instrumented(
    tool: BaseTool,
    cache: RunCache,
    settings: Settings,
    sleep: Callable[[float], Awaitable[None]],
) -> BaseTool:
    """Rebuild one tool around the wrapper that guards it."""
    run = (
        _page_reader(tool, cache, settings, sleep)
        if tool.name == PAGE_TOOL_NAME
        else _searcher(tool, cache, settings, sleep)
    )
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=run,
    )


def _searcher(
    tool: BaseTool,
    cache: RunCache,
    settings: Settings,
    sleep: Callable[[float], Awaitable[None]],
) -> Callable[..., Awaitable[str]]:
    """Build the search tool's wrapper: the cache in front, the retry policy behind.

    The cache is keyed on the query alone. Any further argument the model supplies is
    passed to the server but does not distinguish one cached search from another.
    """

    async def search(query: str, **rest: object) -> str:
        served = cache.search(query)
        source = "cache hit"
        if served is None:
            served = await _answer(tool, {"query": query, **rest}, settings, sleep)
            cache.record_search(query, served)
            source = "cache miss"
        logger.debug(
            "%s %r: %d results (%s)", tool.name, query, _results(served), source
        )
        return served

    return search


def _page_reader(
    tool: BaseTool,
    cache: RunCache,
    settings: Settings,
    sleep: Callable[[float], Awaitable[None]],
) -> Callable[..., Awaitable[str]]:
    """Build the page tool's wrapper, which also holds the character ceiling.

    The cache holds the page as fetched and the ceiling is applied on the way out,
    so a hit and the miss that filled it hand the agent the same text.
    """

    async def read(url: str, **rest: object) -> str:
        fetched = cache.page(url)
        source = "cache hit"
        if fetched is None:
            fetched = await _answer(tool, {"url": url, **rest}, settings, sleep)
            cache.record_page(url, fetched)
            source = "cache miss"
        served = _capped(fetched, settings.page_character_ceiling)
        logger.debug("%s %s: %d characters (%s)", tool.name, url, len(served), source)
        return served

    return read


async def _answer(
    tool: BaseTool,
    arguments: dict[str, object],
    settings: Settings,
    sleep: Callable[[float], Awaitable[None]],
) -> str:
    """Call the tool under the retry policy, and read its answer as text."""

    async def call() -> str:
        return _as_text(await tool.ainvoke(arguments))

    return await with_retry(call, settings.retry_attempts, sleep)


def _as_text(answered: object) -> str:
    """Read what an MCP tool returned as the text a language model reads.

    `langchain-mcp-adapters` 0.3.2 builds every tool with
    `response_format="content_and_artifact"`, so invoking one returns the content: a
    list of LangChain content blocks. Only the text blocks carry anything a model can
    read, so an image or a file block is dropped rather than described. A tool that
    answers with plain text instead — which `BaseTool` allows — is passed through.
    """
    if isinstance(answered, list):
        return _BLOCK_SEPARATOR.join(
            block["text"] for block in answered if block.get("type") == "text"
        )
    return str(answered)


def _results(text: str) -> int:
    """Count the results in a search's text.

    A search arrives as content blocks that `_as_text` joins with a blank line, so
    counting the text's non-empty blank-line-separated sections counts them back. A
    cache hit and the miss that filled it therefore report the same number.
    """
    return len([section for section in text.split(_BLOCK_SEPARATOR) if section.strip()])


def _capped(page: str, ceiling: int) -> str:
    """Cut a page at the ceiling, and say where it was cut.

    The ceiling guards against the page that would overflow the model's context. It
    is not a summarising step, so a page within it is passed through whole.
    """
    if len(page) <= ceiling:
        return page
    return page[:ceiling] + TRUNCATION_MARKER
