"""Tests for the Bright Data tool layer in `factchecker.tools`.

No test here opens a connection. `load_tools` takes only an endpoint, so it has no
injection seam of its own: these tests replace `MultiServerMCPClient` where the module
holds it, which is the seam the command-line tests use for their collaborators too.
The tools that stand in for the server's own are real `StructuredTool` objects built
the way `langchain-mcp-adapters` 0.3.2 builds them: a JSON-schema `dict` for
`args_schema`, and an answer returned as a list of LangChain content blocks.
"""

import asyncio
import logging
import traceback
from collections.abc import Awaitable, Callable

import httpx
import pytest
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.tools import _MCPToolExecutionError

from factchecker import tools
from factchecker.cache import RunCache
from factchecker.config import ConfigurationError, McpEndpoint, Settings
from factchecker.errors import AuthenticationFailed, McpCallError
from factchecker.tools import (
    PAGE_TOOL_NAME,
    SEARCH_TOOL_NAME,
    TRUNCATION_MARKER,
    instrument,
    load_tools,
)

CREDENTIAL = "brd-4a7f2e91c0"
ENDPOINT = McpEndpoint(CREDENTIAL)

QUERY = "does water boil at 100 C"
URL = "https://example.test/Boiling"

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
PAGE_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string"}},
    "required": ["url"],
}
SCHEMAS = {SEARCH_TOOL_NAME: SEARCH_SCHEMA, PAGE_TOOL_NAME: PAGE_SCHEMA}

# The two catalogues as Bright Data publishes them, with the arguments beyond the
# one this package allows. `country` and the rest are what the run cache cannot key
# on, so an instrumented tool must not offer them.
WIDE_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "what to ask the search engine"},
        "country": {"type": "string"},
        "num_results": {"type": "integer"},
    },
    "required": ["query"],
}
WIDE_PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "the page to read"},
        "data_format": {"type": "string"},
    },
    "required": ["url"],
}
WIDE_SCHEMAS = {SEARCH_TOOL_NAME: WIDE_SEARCH_SCHEMA, PAGE_TOOL_NAME: WIDE_PAGE_SCHEMA}


def _blocks(*texts: str) -> list[dict[str, str]]:
    """An answer in the shape the MCP adapter returns one: LangChain content blocks."""
    return [{"type": "text", "text": text} for text in texts]


def _status_failure(status: int) -> BaseException:
    """The failure a live call raises for an HTTP status: a group around `httpx`.

    The endpoint URL, token and all, is in the `httpx` message, exactly as it is on
    the wire. That is what makes the redaction assertions below mean anything.
    """
    request = httpx.Request("POST", ENDPOINT.unredacted_url())
    response = httpx.Response(status, request=request)
    with pytest.raises(httpx.HTTPStatusError) as raised:
        response.raise_for_status()
    return ExceptionGroup("unhandled errors in a TaskGroup", [raised.value])


class _Server:
    """One tool as the server implements it: it records calls and replies in turn.

    The reply for a call is the one at its position, and the last reply stands for
    every call after it, so a tool that always fails is written with one failure.
    """

    def __init__(self, *replies: object) -> None:
        self.replies = replies
        self.calls: list[dict[str, object]] = []

    async def __call__(self, **arguments: object) -> object:
        self.calls.append(arguments)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _Clock:
    """A sleep that records what it was asked to wait and waits for none of it."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _tool(
    name: str,
    server: _Server,
    metadata: dict[str, object] | None = None,
    schema: dict[str, object] | None = None,
) -> BaseTool:
    """A tool of the given name, answered by the given server.

    A real adapter tool raises rather than answers when the server reports a tool
    error, because `load_tools` builds the client with `handle_tool_errors=False`.
    A `StructuredTool` does the same with an unset `handle_tool_error`, so a `_Server`
    that raises a `_MCPToolExecutionError` reproduces that path exactly.
    """

    async def call(**arguments: object) -> object:
        return await server(**arguments)

    return StructuredTool(
        name=name,
        description=f"the {name} tool",
        args_schema=schema or SCHEMAS.get(name, SEARCH_SCHEMA),
        metadata=metadata,
        coroutine=call,
    )


def _client_offering(
    *offered: BaseTool, failing: BaseException | None = None
) -> type[object]:
    """A stand-in for `MultiServerMCPClient` that offers the given tools.

    The class records every connection configuration it is constructed with and every
    option it was built with, so a test can assert on the URL `load_tools` opened
    without that URL being logged.
    """

    class _FakeClient:
        opened: list[dict[str, dict[str, str]]] = []
        options: list[dict[str, object]] = []

        def __init__(
            self, connections: dict[str, dict[str, str]], **options: object
        ) -> None:
            _FakeClient.opened.append(connections)
            _FakeClient.options.append(options)

        async def get_tools(self) -> list[BaseTool]:
            if failing is not None:
                raise failing
            return list(offered)

    return _FakeClient


def _offering_neither() -> type[object]:
    """A client whose server offers neither of the two tools this package needs."""
    return _client_offering(_tool("session_stats", _Server()))


def _offering_only(name: str) -> type[object]:
    """A client whose server offers one of the two tools and not the other."""
    return _client_offering(_tool(name, _Server()))


def _tool_error(
    text: str = "target site blocked the request",
) -> _MCPToolExecutionError:
    """The exception the adapter raises for `CallToolResult(isError=True)`."""
    return _MCPToolExecutionError([{"type": "text", "text": text}])


def _settings(**overrides: object) -> Settings:
    """Settings whose tunables a test can move one at a time."""
    fields: dict[str, object] = {
        "openrouter_api_key": "sk-or-v1-unused",
        "model": "google/gemma-4-31b-it",
        "mcp_endpoint": ENDPOINT,
        "tool_call_budget": 10,
        "page_character_ceiling": 100,
        "concurrency": 8,
        "statement_timeout_seconds": 240.0,
        "retry_attempts": 3,
    }
    return Settings(**(fields | overrides))


def _instrumented(
    *servers: tuple[str, _Server],
    cache: RunCache | None = None,
    settings: Settings | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> dict[str, BaseTool]:
    """The named tools, instrumented, by name."""
    wrapped = instrument(
        [_tool(name, server) for name, server in servers],
        cache or RunCache(),
        settings or _settings(),
        sleep or _Clock(),
    )
    return {tool.name: tool for tool in wrapped}


def _call(tool: BaseTool, **arguments: object) -> str:
    """Invoke an instrumented tool and hand back the text it produced."""
    return asyncio.run(tool.ainvoke(arguments))


@pytest.fixture(autouse=True)
def _debug_records(caplog: pytest.LogCaptureFixture) -> None:
    """Capture everything this package logs, so a leak has nowhere to hide."""
    caplog.set_level(logging.DEBUG, logger="factchecker")


def test_the_two_tool_names_are_the_ones_bright_data_publishes() -> None:
    """Every other assertion about tool selection is built on these two."""
    assert SEARCH_TOOL_NAME == "search_engine"
    assert PAGE_TOOL_NAME == "scrape_as_markdown"


def test_load_tools_returns_only_the_two_named_tools_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server offers a catalogue; the agent is given the two tools it can spend."""
    offered = [
        _tool("session_stats", _Server()),
        _tool(PAGE_TOOL_NAME, _Server()),
        _tool("scraping_browser_navigate", _Server()),
        _tool(SEARCH_TOOL_NAME, _Server()),
    ]
    monkeypatch.setattr(tools, "MultiServerMCPClient", _client_offering(*offered))

    selected, _ = asyncio.run(load_tools(ENDPOINT))

    assert [tool.name for tool in selected] == [SEARCH_TOOL_NAME, PAGE_TOOL_NAME]


def test_load_tools_opens_the_endpoint_with_the_token_in_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bright Data authenticates by the URL, so the unredacted one has to reach it."""
    client = _client_offering(
        _tool(SEARCH_TOOL_NAME, _Server()), _tool(PAGE_TOOL_NAME, _Server())
    )
    monkeypatch.setattr(tools, "MultiServerMCPClient", client)

    asyncio.run(load_tools(ENDPOINT))

    (connections,) = client.opened
    (connection,) = connections.values()
    assert connection["url"] == ENDPOINT.unredacted_url()
    assert connection["transport"] == "streamable_http"


def test_the_release_callable_is_reachable_and_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 4 closes the run by calling this, and a caller cannot close a list."""
    monkeypatch.setattr(
        tools,
        "MultiServerMCPClient",
        _client_offering(
            _tool(SEARCH_TOOL_NAME, _Server()), _tool(PAGE_TOOL_NAME, _Server())
        ),
    )

    async def drive() -> object:
        _, release = await load_tools(ENDPOINT)
        return await release()

    assert asyncio.run(drive()) is None


def test_a_server_offering_neither_tool_is_rejected_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalogue without the two tools is a misconfiguration, not an empty run."""
    monkeypatch.setattr(tools, "MultiServerMCPClient", _offering_neither())

    with pytest.raises(ConfigurationError) as raised:
        asyncio.run(load_tools(ENDPOINT))

    assert SEARCH_TOOL_NAME in str(raised.value)
    assert PAGE_TOOL_NAME in str(raised.value)
    assert "session_stats" in str(raised.value)


def test_the_rejected_catalogue_releases_before_it_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure part-way through `load_tools` still gives back what it took.

    The release is a module-level function rather than a closure so that this test
    can replace it and watch it run. Replacing it substitutes both the callable
    `load_tools` invokes on the way out and the one it returns, because each is
    resolved as a module attribute when `load_tools` is called.
    """
    monkeypatch.setattr(tools, "MultiServerMCPClient", _offering_neither())
    released: list[str] = []

    async def _watched_release() -> None:
        released.append("released")

    monkeypatch.setattr(tools, "_release", _watched_release)

    with pytest.raises(ConfigurationError):
        asyncio.run(load_tools(ENDPOINT))

    assert released == ["released"]


def test_a_failure_to_connect_releases_before_it_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connect path gives back what it took as well as the catalogue path."""
    monkeypatch.setattr(
        tools,
        "MultiServerMCPClient",
        _client_offering(failing=_status_failure(503)),
    )
    released: list[str] = []

    async def _watched_release() -> None:
        released.append("released")

    monkeypatch.setattr(tools, "_release", _watched_release)

    with pytest.raises(ConfigurationError):
        asyncio.run(load_tools(ENDPOINT))

    assert released == ["released"]


def test_a_failure_to_connect_carries_no_token_out_of_load_tools(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`httpx` names the request URL in its message, and that URL is the credential."""
    original = _status_failure(503)
    assert CREDENTIAL in str(original.exceptions[0])
    monkeypatch.setattr(
        tools, "MultiServerMCPClient", _client_offering(failing=original)
    )

    with pytest.raises(ConfigurationError) as raised:
        asyncio.run(load_tools(ENDPOINT))

    rendered = "".join(traceback.format_exception(raised.value))
    assert CREDENTIAL not in str(raised.value)
    assert CREDENTIAL not in rendered
    assert CREDENTIAL not in caplog.text
    assert raised.value.__cause__ is None
    assert str(ENDPOINT) in str(raised.value)


@pytest.mark.parametrize("status", [401, 403])
def test_a_token_the_server_refuses_at_load_time_is_an_authentication_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, status: int
) -> None:
    """A refused token is not a setup mistake, and the command must not report one.

    An absent variable and a token the server will not take ask for different
    things of an operator. The command spends a different exit code on each, and
    this is the seam that decides which one a failed connect reaches.
    """
    original = _status_failure(status)
    assert CREDENTIAL in str(original.exceptions[0])
    monkeypatch.setattr(
        tools, "MultiServerMCPClient", _client_offering(failing=original)
    )

    with pytest.raises(AuthenticationFailed) as raised:
        asyncio.run(load_tools(ENDPOINT))

    assert str(raised.value) == f"the MCP server at {ENDPOINT} returned {status}"
    rendered = "".join(traceback.format_exception(raised.value))
    assert CREDENTIAL not in rendered
    assert CREDENTIAL not in caplog.text
    assert raised.value.__cause__ is None


def test_a_refused_token_releases_before_it_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run ends here, so the connection must not be left open behind it."""
    monkeypatch.setattr(
        tools, "MultiServerMCPClient", _client_offering(failing=_status_failure(401))
    )
    released: list[str] = []

    async def _watched_release() -> None:
        released.append("released")

    monkeypatch.setattr(tools, "_release", _watched_release)

    with pytest.raises(AuthenticationFailed):
        asyncio.run(load_tools(ENDPOINT))

    assert released == ["released"]


def test_the_rejected_catalogue_names_the_endpoint_redacted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The endpoint is worth naming and the token never is."""
    monkeypatch.setattr(tools, "MultiServerMCPClient", _offering_neither())

    with pytest.raises(ConfigurationError) as raised:
        asyncio.run(load_tools(ENDPOINT))

    assert str(ENDPOINT) in str(raised.value)
    assert CREDENTIAL not in str(raised.value)
    assert CREDENTIAL not in caplog.text


def test_load_tools_asks_the_adapter_to_raise_a_tool_error_rather_than_answer_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is the seam that keeps a server-side error out of the run's cache.

    `langchain-mcp-adapters` 0.3.2 returns the error text as though it were a result
    unless the client is built this way, and the wrappers cannot tell such a result
    from a real one.
    """
    client = _client_offering(
        _tool(SEARCH_TOOL_NAME, _Server()), _tool(PAGE_TOOL_NAME, _Server())
    )
    monkeypatch.setattr(tools, "MultiServerMCPClient", client)

    asyncio.run(load_tools(ENDPOINT))

    (options,) = client.options
    assert options == {"handle_tool_errors": False}


def test_a_server_offering_one_of_the_two_tools_is_rejected_by_the_missing_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half a catalogue is a misconfiguration too, and the message says which half."""
    monkeypatch.setattr(tools, "MultiServerMCPClient", _offering_only(SEARCH_TOOL_NAME))

    with pytest.raises(ConfigurationError) as raised:
        asyncio.run(load_tools(ENDPOINT))

    assert f"offers no {PAGE_TOOL_NAME};" in str(raised.value)
    assert " and no " not in str(raised.value)
    assert SEARCH_TOOL_NAME in str(raised.value)


def test_a_failure_to_connect_names_the_status_the_server_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected token, a refused connection and a 500 all group alike.

    Without the status the message reads the same for all three, and an operator
    reading it cannot tell a bad token from a server that is down.
    """
    monkeypatch.setattr(
        tools, "MultiServerMCPClient", _client_offering(failing=_status_failure(500))
    )

    with pytest.raises(ConfigurationError) as raised:
        asyncio.run(load_tools(ENDPOINT))

    assert str(raised.value) == f"the MCP server at {ENDPOINT} returned 500"


def test_an_instrumented_tool_keeps_its_name_and_its_description() -> None:
    """The agent chooses tools by what they say they are, so neither may move."""
    original = _tool(SEARCH_TOOL_NAME, _Server())

    (wrapped,) = instrument([original], RunCache(), _settings(), _Clock())

    assert wrapped.name == original.name
    assert wrapped.description == original.description


@pytest.mark.parametrize(
    ("name", "kept"), [(SEARCH_TOOL_NAME, "query"), (PAGE_TOOL_NAME, "url")]
)
def test_an_instrumented_tool_offers_only_the_argument_the_agent_may_send(
    name: str, kept: str
) -> None:
    """The cache constraint is written into the schema rather than asked for in prose.

    The run cache keys a search on its query and a page on its URL. Every other
    argument the server publishes is forwarded unkeyed, so two calls that differ only
    in one of those return the first one's cached answer. Offering the agent one
    argument is what stops it asking for the others.
    """
    original = _tool(name, _Server(), schema=WIDE_SCHEMAS[name])

    (wrapped,) = instrument([original], RunCache(), _settings(), _Clock())

    assert wrapped.args_schema == {
        "type": "object",
        "properties": {kept: WIDE_SCHEMAS[name]["properties"][kept]},
        "required": [kept],
    }


def test_narrowing_the_schema_leaves_the_servers_own_schema_untouched() -> None:
    """The narrowed copy is this package's; the entry itself belongs to the server."""
    original = _tool(SEARCH_TOOL_NAME, _Server(), schema=dict(WIDE_SEARCH_SCHEMA))

    instrument([original], RunCache(), _settings(), _Clock())

    assert original.args_schema == WIDE_SEARCH_SCHEMA


def test_an_instrumented_tool_keeps_the_annotations_the_server_published() -> None:
    """The adapter puts the server's tool annotations there, and an agent reads them."""
    annotations = {"readOnlyHint": True, "title": "Search the web"}
    original = _tool(SEARCH_TOOL_NAME, _Server(), metadata=annotations)

    (wrapped,) = instrument([original], RunCache(), _settings(), _Clock())

    assert wrapped.metadata == annotations


def test_a_search_miss_calls_the_server_and_a_hit_does_not() -> None:
    """A second statement asking the same question of the web costs nothing."""
    server = _Server(_blocks("boiling point: 100 C"))
    search = _instrumented((SEARCH_TOOL_NAME, server))[SEARCH_TOOL_NAME]

    first = _call(search, query=QUERY)
    second = _call(search, query=QUERY)

    assert first == second == "boiling point: 100 C"
    assert server.calls == [{"query": QUERY}]


def test_case_and_whitespace_differences_share_one_search() -> None:
    """Two agents phrasing one question alike spend one call between them."""
    server = _Server(_blocks("boiling point: 100 C"))
    search = _instrumented((SEARCH_TOOL_NAME, server))[SEARCH_TOOL_NAME]

    _call(search, query=QUERY)
    served = _call(search, query=f"  {QUERY.upper()}  ")

    assert served == "boiling point: 100 C"
    assert len(server.calls) == 1


def test_a_search_answer_of_several_blocks_reaches_the_agent_as_text() -> None:
    """The MCP adapter answers in content blocks, and a model reads text."""
    server = _Server(_blocks("first result", "second result"))
    search = _instrumented((SEARCH_TOOL_NAME, server))[SEARCH_TOOL_NAME]

    assert _call(search, query=QUERY) == "first result\n\nsecond result"


def test_a_tool_that_answers_with_plain_text_is_passed_through() -> None:
    """`BaseTool` lets a tool answer with a bare string, and a model reads that too."""
    server = _Server("boiling point: 100 C")
    search = _instrumented((SEARCH_TOOL_NAME, server))[SEARCH_TOOL_NAME]

    assert _call(search, query=QUERY) == "boiling point: 100 C"


@pytest.mark.parametrize(
    "answered",
    [
        ["a bare string where a block belongs"],
        [{"type": "text"}],
    ],
)
def test_a_block_of_an_unexpected_shape_reads_as_nothing(
    answered: list[object],
) -> None:
    """Neither shape is one the pinned adapter builds, and neither may end a check.

    Reading a block that is not a mapping, or a text block with no text in it, would
    otherwise raise where the reader can say only that an attribute or a key is
    missing.
    """
    search = _instrumented((SEARCH_TOOL_NAME, _Server(answered)))[SEARCH_TOOL_NAME]

    assert _call(search, query=QUERY) == ""


def test_a_page_within_the_ceiling_passes_through_whole() -> None:
    """The ceiling guards the context window; it is not a summarising step."""
    page = "a" * 100
    reader = _instrumented(
        (PAGE_TOOL_NAME, _Server(_blocks(page))), settings=_settings()
    )[PAGE_TOOL_NAME]

    assert _call(reader, url=URL) == page


def test_a_page_over_the_ceiling_is_cut_and_marked() -> None:
    """The agent has to know it read a fragment, so that it can say so."""
    reader = _instrumented(
        (PAGE_TOOL_NAME, _Server(_blocks("b" * 250))),
        settings=_settings(page_character_ceiling=100),
    )[PAGE_TOOL_NAME]

    served = _call(reader, url=URL)

    assert served == "b" * 100 + TRUNCATION_MARKER
    assert TRUNCATION_MARKER == "\n\n[truncated: page exceeded the character ceiling]"


def test_the_ceiling_is_the_configured_one() -> None:
    """The ceiling is configuration, so nothing here may hold a literal of its own."""
    reader = _instrumented(
        (PAGE_TOOL_NAME, _Server(_blocks("c" * 250))),
        settings=_settings(page_character_ceiling=7),
    )[PAGE_TOOL_NAME]

    assert _call(reader, url=URL) == "c" * 7 + TRUNCATION_MARKER


def test_a_page_hit_serves_the_cached_page_without_a_call() -> None:
    """Two statements reading one page fetch it once."""
    server = _Server(_blocks("# Boiling"))
    reader = _instrumented((PAGE_TOOL_NAME, server))[PAGE_TOOL_NAME]

    first = _call(reader, url=URL)
    second = _call(reader, url=URL)

    assert first == second == "# Boiling"
    assert server.calls == [{"url": URL}]


def test_one_cache_serves_every_tool_it_is_given() -> None:
    """One instrumented set of tools serves a whole run, so its cache is shared."""
    cache = RunCache()
    server = _Server(_blocks("boiling point: 100 C"))
    first = _instrumented((SEARCH_TOOL_NAME, server), cache=cache)[SEARCH_TOOL_NAME]
    second = _instrumented((SEARCH_TOOL_NAME, server), cache=cache)[SEARCH_TOOL_NAME]

    _call(first, query=QUERY)
    _call(second, query=QUERY)

    assert len(server.calls) == 1


def test_a_transient_failure_is_retried_as_often_as_the_settings_allow() -> None:
    """The count is configuration, so nothing here may hold a literal of its own."""
    server = _Server(_status_failure(503))
    clock = _Clock()
    search = _instrumented(
        (SEARCH_TOOL_NAME, server), settings=_settings(retry_attempts=2), sleep=clock
    )[SEARCH_TOOL_NAME]

    with pytest.raises(McpCallError):
        _call(search, query=QUERY)

    assert len(server.calls) == 2
    assert len(clock.delays) == 1


def test_a_failed_search_is_not_cached() -> None:
    """A failure is not fetched material, so the next asker gets a real attempt."""
    server = _Server(_status_failure(404), _blocks("boiling point: 100 C"))
    search = _instrumented((SEARCH_TOOL_NAME, server))[SEARCH_TOOL_NAME]

    with pytest.raises(McpCallError):
        _call(search, query=QUERY)

    assert _call(search, query=QUERY) == "boiling point: 100 C"


def test_an_error_the_server_reported_is_never_stored_as_evidence() -> None:
    """A blocked target or a spent quota is a failure, not a search result.

    The adapter answers a `CallToolResult(isError=True)` with the error text unless it
    is told otherwise, and a cache that took that text would hand it to every later
    statement of the run as the evidence it searched for.
    """
    server = _Server(_tool_error(), _blocks("boiling point: 100 C"))
    cache = RunCache()
    search = _instrumented((SEARCH_TOOL_NAME, server), cache=cache)[SEARCH_TOOL_NAME]

    with pytest.raises(McpCallError):
        _call(search, query=QUERY)

    assert cache.search(QUERY) is None
    assert _call(search, query=QUERY) == "boiling point: 100 C"


def test_an_error_the_server_reported_is_never_stored_as_a_page() -> None:
    """The page tool keeps its own store, so it needs its own assertion."""
    server = _Server(_tool_error(), _blocks("# Boiling"))
    cache = RunCache()
    reader = _instrumented((PAGE_TOOL_NAME, server), cache=cache)[PAGE_TOOL_NAME]

    with pytest.raises(McpCallError):
        _call(reader, url=URL)

    assert cache.page(URL) is None
    assert _call(reader, url=URL) == "# Boiling"


def test_a_rejected_credential_propagates_out_of_the_tool() -> None:
    """`run_check` ends the run on this, so nothing on the way may catch it."""
    search = _instrumented((SEARCH_TOOL_NAME, _Server(_status_failure(401))))[
        SEARCH_TOOL_NAME
    ]

    with pytest.raises(AuthenticationFailed):
        _call(search, query=QUERY)


def test_a_search_writes_one_debug_record_naming_what_it_did(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An operator reading DEBUG can see every call the run spent and what served it."""
    server = _Server(_blocks("first result", "second result"))
    search = _instrumented((SEARCH_TOOL_NAME, server))[SEARCH_TOOL_NAME]

    _call(search, query=QUERY)
    _call(search, query=QUERY)

    miss, hit = caplog.records
    assert [record.name for record in (miss, hit)] == ["factchecker.tools"] * 2
    assert [record.levelno for record in (miss, hit)] == [logging.DEBUG] * 2
    assert SEARCH_TOOL_NAME in miss.getMessage()
    assert QUERY in miss.getMessage()
    assert "2 results" in miss.getMessage()
    assert "cache miss" in miss.getMessage()
    assert "cache hit" in hit.getMessage()


def test_a_page_read_writes_one_debug_record_naming_what_it_did(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The page record names the URL and the size of the text the agent was handed."""
    reader = _instrumented(
        (PAGE_TOOL_NAME, _Server(_blocks("d" * 250))),
        settings=_settings(page_character_ceiling=100),
    )[PAGE_TOOL_NAME]

    _call(reader, url=URL)
    _call(reader, url=URL)

    miss, hit = caplog.records
    assert PAGE_TOOL_NAME in miss.getMessage()
    assert URL in miss.getMessage()
    assert f"{100 + len(TRUNCATION_MARKER)} characters" in miss.getMessage()
    assert "cache miss" in miss.getMessage()
    assert "cache hit" in hit.getMessage()


def test_no_record_a_tool_call_writes_carries_the_endpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The token travels inside the endpoint URL, so no record may hold either."""
    cache = RunCache()
    wrapped = _instrumented(
        (SEARCH_TOOL_NAME, _Server(_blocks("boiling point: 100 C"))),
        (PAGE_TOOL_NAME, _Server(_blocks("# Boiling"))),
        cache=cache,
    )

    _call(wrapped[SEARCH_TOOL_NAME], query=QUERY)
    _call(wrapped[PAGE_TOOL_NAME], url=URL)

    assert caplog.records
    assert CREDENTIAL not in caplog.text
    assert "mcp.brightdata.com" not in caplog.text


def _loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run `load_tools` against a server that offers both tools, and drop the result.

    What the tests below want from it is the logging guard it attaches on the way in.
    """
    monkeypatch.setattr(
        tools,
        "MultiServerMCPClient",
        _client_offering(
            _tool(SEARCH_TOOL_NAME, _Server()), _tool(PAGE_TOOL_NAME, _Server())
        ),
    )
    asyncio.run(load_tools(ENDPOINT))


def test_a_third_party_traceback_cannot_print_the_endpoint_whole(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`mcp` renders a whole traceback when a notification POST fails.

    `configure_logging` gives the `factchecker` logger a handler and nothing else, so
    such a record reaches `logging.lastResort` and prints to stderr. The `httpx`
    message inside it names the request URL, and the token travels in that URL.
    """
    _loaded(monkeypatch)
    caplog.set_level(logging.DEBUG)
    failed = _status_failure(500)
    assert CREDENTIAL in "".join(traceback.format_exception(failed))

    try:
        raise failed
    except BaseExceptionGroup:
        logging.getLogger("mcp.client.streamable_http").exception(
            "Error in post_writer"
        )

    printed = caplog.text + capsys.readouterr().err
    assert CREDENTIAL not in printed
    assert "token=REDACTED" in printed
    assert "Error in post_writer" in printed
    assert "HTTPStatusError" in printed


def test_a_third_party_message_naming_the_endpoint_is_redacted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`httpx` writes the request URL into the message itself, without a traceback."""
    _loaded(monkeypatch)
    caplog.set_level(logging.DEBUG)

    logging.getLogger("httpx").warning(
        "HTTP Request: %s %s", "POST", ENDPOINT.unredacted_url()
    )

    assert CREDENTIAL not in caplog.text
    assert "token=REDACTED" in caplog.text


def test_the_redaction_is_attached_once_however_often_the_tools_are_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logger holds its filters in a list, and one grown on every call is a leak."""
    _loaded(monkeypatch)
    _loaded(monkeypatch)

    guarded = logging.getLogger("mcp.client.streamable_http")
    attached = [
        one for one in guarded.filters if isinstance(one, tools._TokenRedaction)
    ]
    assert len(attached) == 1
