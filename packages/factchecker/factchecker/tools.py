"""The two Bright Data tools a run spends its budget on, and the guards around them."""

import logging
import re
import traceback
from collections.abc import Awaitable, Callable, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from factchecker.cache import RunCache
from factchecker.config import ConfigurationError, McpEndpoint, Settings
from factchecker.resilience import describe_failure, with_retry

logger = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_engine"
PAGE_TOOL_NAME = "scrape_as_markdown"
TRUNCATION_MARKER = "\n\n[truncated: page exceeded the character ceiling]"

_SEARCH_ARGUMENT = "query"
_PAGE_ARGUMENT = "url"

_WANTED = (SEARCH_TOOL_NAME, PAGE_TOOL_NAME)
_SERVER_NAME = "brightdata"
_BLOCK_SEPARATOR = "\n\n"

# The loggers of the pinned stack that write a request URL, named one by one because a
# filter runs only for the records its own logger emits: a record from a child reaches
# an ancestor's handlers without ever meeting that ancestor's filters.
_LOGGERS_THAT_WRITE_A_URL = ("httpx", "mcp.client.streamable_http")

_TOKEN_PARAMETER = re.compile(r"([?&]token=)[^&\s'\"<>]+")
_REDACTED_PARAMETER = r"\1REDACTED"


async def load_tools(
    endpoint: McpEndpoint,
) -> tuple[list[BaseTool], Callable[[], Awaitable[None]]]:
    """Open the Bright Data MCP server and take the two tools this package uses.

    The release callable comes back beside the tools because a caller cannot close a
    list. What it releases is described on `_release`.

    The adapter is told to handle no tool error. Left to itself,
    `langchain-mcp-adapters` 0.3.2 answers a `CallToolResult(isError=True)` with the
    server's error text as though that text were a result, and the wrappers
    `instrument` adds would store it as fetched material and serve it to every later
    statement of the run.

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
    _redact_third_party_records()
    client = MultiServerMCPClient(
        {
            _SERVER_NAME: {
                "transport": "streamable_http",
                "url": endpoint.unredacted_url(),
            }
        },
        handle_tool_errors=False,
    )
    try:
        offered = {tool.name: tool for tool in await client.get_tools()}
    except Exception as failure:  # noqa: BLE001 — nothing here may escape unredacted
        await _release()
        # The cause is dropped rather than chained. It arrives from `httpx`, whose
        # message names the request URL, and the token travels inside that URL. This
        # is the one thing `McpEndpoint` cannot keep out of another library's text.
        raise ConfigurationError(describe_failure(endpoint, failure)) from None
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
        settings: Read for `retry_attempts`, `page_character_ceiling` and the endpoint
            a failure names.
        sleep: Waits the number of seconds it is given, between retries.

    Returns:
        One instrumented tool per tool given, keeping each tool's name and
        description so that the agent chooses between them as before, and offering
        the one argument the run cache keys on.
    """
    return [_instrumented(tool, cache, settings, sleep) for tool in tools]


class _TokenRedaction(logging.Filter):
    """Take the Bright Data token out of a record before any handler prints it."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record in place, and keep it.

        The message is settled here and its arguments dropped, so that a later
        `getMessage` cannot rebuild the unredacted text. A traceback is rendered here
        for the same reason: `logging.Formatter` renders one only where `exc_text` is
        still empty, so filling it in is what decides what every handler prints.

        Args:
            record: The record a logger is about to hand to its handlers.

        Returns:
            `True`, always. This filter redacts rather than discards, because a
            record naming a failed request is worth reading.
        """
        record.msg = _without_the_token(record.getMessage())
        record.args = None
        if record.exc_info is not None and record.exc_text is None:
            rendered = traceback.format_exception(*record.exc_info)
            record.exc_text = _without_the_token("".join(rendered))
        return True


def _redact_third_party_records() -> None:
    """Stop another library's record from printing the endpoint URL whole.

    `configure_logging` attaches a handler to the `factchecker` logger alone, so a
    record from `mcp` or `httpx` finds no handler anywhere above it and
    `logging.lastResort` prints it to stderr. `mcp` 1.29.0 renders a whole traceback
    that way when a notification POST fails, and the `httpx` message inside it names
    the request URL, which is the credential.

    Attaching the filter twice would redact twice, which is harmless, and would grow
    the list on every call, which is not. So a logger already carrying one is left
    alone.
    """
    for name in _LOGGERS_THAT_WRITE_A_URL:
        guarded = logging.getLogger(name)
        attached = (isinstance(one, _TokenRedaction) for one in guarded.filters)
        if not any(attached):
            guarded.addFilter(_TokenRedaction())


def _without_the_token(text: str) -> str:
    """Replace the value of every `token` query parameter with `REDACTED`."""
    return _TOKEN_PARAMETER.sub(_REDACTED_PARAMETER, text)


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
    """Rebuild one tool around the wrapper that guards it.

    Name, description and metadata all come across, because the agent reads them to
    choose between tools and the adapter puts the server's tool annotations in the
    last of them. `response_format` does not come across: the original answers with
    content beside an artifact, and the wrapper answers with the text it read out of
    that content. The argument schema is narrowed rather than copied, by `_narrowed`.
    """
    reads_a_page = tool.name == PAGE_TOOL_NAME
    run = (
        _page_reader(tool, cache, settings, sleep)
        if reads_a_page
        else _searcher(tool, cache, settings, sleep)
    )
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=_narrowed(
            tool.args_schema, _PAGE_ARGUMENT if reads_a_page else _SEARCH_ARGUMENT
        ),
        metadata=tool.metadata,
        coroutine=run,
    )


def _narrowed(published: dict[str, object], kept: str) -> dict[str, object]:
    """Offer the agent the one argument the run cache keys on, and no other.

    The cache keys a search on its query and a page on its URL, and forwards every
    further argument unkeyed. Two calls differing only in one of those would return
    the first one's cached answer, so the agent must never send them.

    Narrowing here rather than asking for it in the system prompt makes the
    constraint structural: the model is not told to leave `country` alone, it is
    never offered a `country`. It also keeps the binding honest. A tool bound beside
    a `response_format` is converted strictly, and that conversion rewrites
    `required` to list every property — a rewrite that adds nothing once the only
    property is the required one.

    The kept property comes across as the server wrote it, description and all, and
    the server's own schema is left as it was.

    Args:
        published: The argument schema the server published for this tool.
        kept: The one argument the agent may send.

    Returns:
        A schema offering that argument alone, and requiring it.
    """
    return {
        "type": "object",
        "properties": {kept: published["properties"][kept]},
        "required": [kept],
    }


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

    return await with_retry(call, settings.retry_attempts, sleep, settings.mcp_endpoint)


def _as_text(answered: object) -> str:
    """Read what an MCP tool returned as the text a language model reads.

    `langchain-mcp-adapters` 0.3.2 builds every tool with
    `response_format="content_and_artifact"`, so invoking one returns the content: a
    list of LangChain content blocks. Only the text blocks carry anything a model can
    read, so an image or a file block is dropped rather than described. A tool that
    answers with plain text instead — which `BaseTool` allows — is passed through.

    A block that is not a mapping, and a text block with no text in it, are both read
    as nothing. Neither shape is one the pinned adapter builds, and reading them as
    nothing costs a line where letting them raise would cost a statement its check
    and say only that a key was missing.
    """
    if isinstance(answered, list):
        return _BLOCK_SEPARATOR.join(
            block.get("text", "")
            for block in answered
            if isinstance(block, dict) and block.get("type") == "text"
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
