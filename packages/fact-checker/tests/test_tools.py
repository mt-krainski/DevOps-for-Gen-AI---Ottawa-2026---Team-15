"""The Bright Data connection, the two tools, and the layers wrapped round a call."""

import json
import logging
from contextlib import AbstractAsyncContextManager

import openai
import pytest
from langchain_core.tools import ToolException

from fact_checker.cache import RunCache
from fact_checker.errors import (
    AuthenticationFailure,
    CheckError,
    ErrorCode,
    StatementFailure,
)
from fact_checker.tools import (
    MAX_REPR_CHARACTERS,
    SCRAPE_AS_MARKDOWN,
    SEARCH_ENGINE,
    Toolkit,
    build_bright_data_client,
    open_toolkit,
)
from tests.conftest import (
    BRIGHT_DATA_CREDENTIAL,
    BRIGHT_DATA_ENDPOINT,
    FakeMCPClient,
    FakeTool,
    StatusCodeError,
    always,
    make_config,
    openai_status_error,
    quoting_the_tokened_url,
)

A_URL = "https://example.test/article"


def opening(
    client: FakeMCPClient,
    *,
    scrape_char_limit: int = 100000,
    api_token: str = BRIGHT_DATA_CREDENTIAL,
) -> AbstractAsyncContextManager[Toolkit]:
    """Open a toolkit over the fake client, with no connection made."""
    config = make_config(scrape_char_limit=scrape_char_limit, api_token=api_token)
    return open_toolkit(config, RunCache(), client_factory=lambda _url: client)


def a_blob_whose_repr_is(characters: int) -> dict[str, str]:
    """Return a result with no text to read, of exactly the `repr` size asked."""
    return {"data": "A" * (characters - len(repr({"data": ""})))}


def offering(*tools: FakeTool) -> FakeMCPClient:
    """Return a client offering the two tools plus whatever else is passed."""
    return FakeMCPClient(tools)


def both_tools() -> tuple[FakeTool, FakeTool]:
    """Return a search tool and a scrape tool, each with a fixed answer."""
    return (
        FakeTool(SEARCH_ENGINE, always("results")),
        FakeTool(SCRAPE_AS_MARKDOWN, always("# A page")),
    )


async def test_the_two_named_tools_are_selected_in_order() -> None:
    """The server offers more than two, and the toolkit binds exactly two."""
    search, scrape = both_tools()
    client = offering(
        FakeTool("session_stats", always("")),
        scrape,
        FakeTool("scrape_as_html", always("")),
        search,
    )

    async with opening(client) as toolkit:
        bound = [tool.name for tool in toolkit.bound_tools]

    assert bound == [SEARCH_ENGINE, SCRAPE_AS_MARKDOWN]


async def test_a_missing_tool_names_it_and_what_was_offered() -> None:
    """A server without `scrape_as_markdown` is a run-level failure."""
    search, _ = both_tools()
    client = offering(search, FakeTool("scrape_as_html", always("")))

    with pytest.raises(CheckError) as raised:
        async with opening(client):
            pass

    assert raised.value.code is ErrorCode.TOOL_ERROR
    assert SCRAPE_AS_MARKDOWN in raised.value.message
    assert "scrape_as_html" in raised.value.message
    assert SEARCH_ENGINE in raised.value.message


async def test_a_connection_failure_names_the_redacted_endpoint() -> None:
    """The message has to be readable in a log, so the token is not in it."""
    client = FakeMCPClient([], failure=RuntimeError("connection refused"))
    config = make_config()

    with pytest.raises(CheckError) as raised:
        async with open_toolkit(config, RunCache(), client_factory=lambda _url: client):
            pass

    assert raised.value.code is ErrorCode.TOOL_ERROR
    assert config.bright_data.redacted_endpoint_url() in raised.value.message
    assert BRIGHT_DATA_CREDENTIAL not in raised.value.message


async def test_a_rejected_token_at_connect_time_is_an_auth_error() -> None:
    """A rejected Bright Data token has to exit 3, not 1."""
    client = FakeMCPClient([], failure=StatusCodeError(401))

    with pytest.raises(CheckError) as raised:
        async with opening(client):
            pass

    assert raised.value.code is ErrorCode.AUTH_ERROR


async def test_a_wrapped_rejection_at_connect_time_is_still_an_auth_error() -> None:
    """The MCP stack runs on task groups, so a 401 can arrive inside a group."""
    wrapped = ExceptionGroup("unhandled errors in a TaskGroup", [StatusCodeError(401)])
    client = FakeMCPClient([], failure=wrapped)

    with pytest.raises(CheckError) as raised:
        async with opening(client):
            pass

    assert raised.value.code is ErrorCode.AUTH_ERROR


@pytest.mark.parametrize("status", [401, 500], ids=["rejected", "server-error"])
async def test_a_connection_failures_own_words_never_carry_the_token(
    status: int,
) -> None:
    """The transport quotes the whole request URL, and the token rides in it."""
    client = FakeMCPClient([], failure=quoting_the_tokened_url(status))

    with pytest.raises(CheckError) as raised:
        async with opening(client):
            pass

    message = raised.value.message
    assert BRIGHT_DATA_CREDENTIAL not in message
    assert f"Client error '{status}' for url" in message
    assert f"{BRIGHT_DATA_ENDPOINT}?token=***" in message


async def test_a_blank_token_leaves_a_message_as_it_was() -> None:
    """Replacing an empty string would put `***` between every character."""
    client = FakeMCPClient([], failure=RuntimeError("connection refused"))

    with pytest.raises(CheckError) as raised:
        async with opening(client, api_token=""):
            pass

    assert raised.value.message.endswith(": connection refused")


async def test_the_real_client_refuses_to_swallow_a_reported_tool_failure() -> None:
    """`handle_tool_errors=False` is what turns an `isError` result into a raise."""
    client = build_bright_data_client("https://mcp.invalid/mcp?token=x")

    assert client.handle_tool_errors is False


async def test_an_identical_search_is_served_from_the_cache() -> None:
    """One run pays for a query once, however many statements ask it."""
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        first = await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        second = await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        third = await toolkit.call(SEARCH_ENGINE, {"query": "other"})

    assert (first, second, third) == ("results", "results", "results")
    assert search.calls == [{"query": "who"}, {"query": "other"}]


async def test_the_cache_key_ignores_the_order_of_the_arguments() -> None:
    """Two dictionaries holding the same pairs name the same search."""
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        await toolkit.call(SEARCH_ENGINE, {"query": "who", "engine": "google"})
        await toolkit.call(SEARCH_ENGINE, {"engine": "google", "query": "who"})

    assert len(search.calls) == 1


async def test_the_search_counter_counts_only_calls_that_reached_the_server() -> None:
    """The counter reports what the run spent, so a cache hit is not one."""
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        await toolkit.call(SEARCH_ENGINE, {"query": "other"})
        await toolkit.call(SCRAPE_AS_MARKDOWN, {"url": A_URL})

        assert toolkit.searches == 2


async def test_a_page_over_the_ceiling_is_cut_and_marked() -> None:
    """The agent has to know it read a fragment."""
    search = FakeTool(SEARCH_ENGINE, always("results"))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("x" * 50))

    async with opening(offering(search, scrape), scrape_char_limit=20) as toolkit:
        page = await toolkit.call(SCRAPE_AS_MARKDOWN, {"url": A_URL})

    assert page == "x" * 20 + "\n\n[truncated at 20 characters]"


async def test_a_page_at_the_ceiling_is_returned_unmarked() -> None:
    """The limit is a ceiling, not a threshold."""
    search = FakeTool(SEARCH_ENGINE, always("results"))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("x" * 20))

    async with opening(offering(search, scrape), scrape_char_limit=20) as toolkit:
        page = await toolkit.call(SCRAPE_AS_MARKDOWN, {"url": A_URL})

    assert page == "x" * 20


async def test_a_search_result_is_never_cut() -> None:
    """The ceiling guards a fetched page, and nothing else."""
    search = FakeTool(SEARCH_ENGINE, always("y" * 50))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape), scrape_char_limit=20) as toolkit:
        results = await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert results == "y" * 50


async def test_a_scrape_without_a_url_names_the_keys_it_carried() -> None:
    """The model filled the schema wrongly, and the message says with what."""
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(StatementFailure) as raised:
            await toolkit.call(SCRAPE_AS_MARKDOWN, {"link": A_URL})

    assert raised.value.code is ErrorCode.TOOL_ERROR
    assert "link" in raised.value.message
    assert scrape.calls == []


async def test_an_unknown_tool_name_is_a_statement_failure() -> None:
    """Only the two selected tools can be called, whatever the model asks for."""
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(StatementFailure) as raised:
            await toolkit.call("scrape_as_html", {"url": A_URL})

    assert raised.value.code is ErrorCode.TOOL_ERROR
    assert "scrape_as_html" in raised.value.message


@pytest.mark.parametrize(
    "rejection",
    [
        pytest.param(
            openai_status_error(openai.AuthenticationError, 401), id="rejected-key"
        ),
        pytest.param(StatusCodeError(403), id="status-code-403"),
    ],
)
async def test_a_rejection_at_call_time_becomes_this_packages_own_failure(
    rejection: Exception,
) -> None:
    """Nothing above this layer should have to know the provider's types."""
    search = FakeTool(SEARCH_ENGINE, always(rejection))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(AuthenticationFailure) as raised:
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert raised.value.code is ErrorCode.AUTH_ERROR
    assert len(search.calls) == 1


async def test_a_rejection_at_call_times_own_words_never_carry_the_token() -> None:
    """This message reaches the operator's terminal, so the token cannot be in it."""
    search = FakeTool(SEARCH_ENGINE, always(quoting_the_tokened_url(401)))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(AuthenticationFailure) as raised:
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert BRIGHT_DATA_CREDENTIAL not in raised.value.message
    assert "Client error '401' for url" in raised.value.message
    assert f"{BRIGHT_DATA_ENDPOINT}?token=***" in raised.value.message


async def test_a_reported_tool_failures_own_words_never_carry_the_token() -> None:
    """This message lands in the statement's `error` field, which the run prints."""
    tokened = f"{BRIGHT_DATA_ENDPOINT}?token={BRIGHT_DATA_CREDENTIAL}"
    reported = ToolException(f"upstream said no to {tokened}")
    search = FakeTool(SEARCH_ENGINE, always(reported))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(StatementFailure) as raised:
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    redacted = f"{BRIGHT_DATA_ENDPOINT}?token=***"
    assert BRIGHT_DATA_CREDENTIAL not in raised.value.message
    assert raised.value.message == f"upstream said no to {redacted}"


async def test_a_reported_tool_failure_is_neither_retried_nor_cached() -> None:
    """The server said no to these arguments; a later statement may still ask."""
    outcomes: list[object] = [ToolException("the target site refused"), "results"]
    search = FakeTool(SEARCH_ENGINE, lambda _arguments: outcomes.pop(0))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(StatementFailure) as raised:
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})

        assert raised.value.code is ErrorCode.TOOL_ERROR
        assert "the target site refused" in raised.value.message
        assert len(search.calls) == 1

        assert await toolkit.call(SEARCH_ENGINE, {"query": "who"}) == "results"
        assert len(search.calls) == 2


async def test_an_unexpected_tool_failure_propagates_as_itself() -> None:
    """Only a rejection and a reported failure are translated; nothing else is."""
    search = FakeTool(SEARCH_ENGINE, always(ValueError("the adapter broke")))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(ValueError, match="the adapter broke"):
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})


async def test_a_single_text_block_arrives_as_a_string() -> None:
    """The adapter collapses one text block, and the string passes through."""
    search = FakeTool(SEARCH_ENGINE, always("one block of text"))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        results = await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert results == "one block of text"


async def test_content_blocks_are_joined_and_the_non_text_one_dropped() -> None:
    """An image block carries nothing the agent can read."""
    blocks = [
        {"type": "text", "text": "first"},
        {"type": "image", "data": "AAAA", "mime_type": "image/png"},
        {"type": "text", "text": "second"},
    ]
    search = FakeTool(SEARCH_ENGINE, always(blocks))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        assert await toolkit.call(SEARCH_ENGINE, {"query": "who"}) == "first\n\nsecond"


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param([{"type": "image", "data": "AAAA"}], id="no-text-block"),
        pytest.param([], id="no-blocks"),
        pytest.param({"structuredContent": {}}, id="not-a-list"),
        pytest.param(None, id="nothing"),
    ],
)
async def test_a_result_with_no_text_is_a_tool_failure(returned: object) -> None:
    """The agent reads text, so anything it cannot read is a failure."""
    search = FakeTool(SEARCH_ENGINE, always(returned))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(StatementFailure) as raised:
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert raised.value.code is ErrorCode.TOOL_ERROR


async def test_a_result_at_the_repr_ceiling_is_shown_whole() -> None:
    """The ceiling is a ceiling, not a threshold, so nothing is cut here."""
    returned = a_blob_whose_repr_is(MAX_REPR_CHARACTERS)
    search = FakeTool(SEARCH_ENGINE, always(returned))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(StatementFailure) as raised:
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert repr(returned) in raised.value.message
    assert "..." not in raised.value.message


async def test_a_result_over_the_repr_ceiling_is_cut_and_marked() -> None:
    """A base64 image block would otherwise land whole in a statement's error."""
    returned = a_blob_whose_repr_is(MAX_REPR_CHARACTERS + 1)
    search = FakeTool(SEARCH_ENGINE, always(returned))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        with pytest.raises(StatementFailure) as raised:
            await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    message = raised.value.message
    assert repr(returned) not in message
    assert message.endswith("...")
    assert repr(returned)[:MAX_REPR_CHARACTERS] in message


async def test_a_search_result_is_logged_by_its_entry_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JSON array's length says more than its character count does."""
    caplog.set_level(logging.DEBUG)
    search = FakeTool(SEARCH_ENGINE, always('[{"url": "a"}, {"url": "b"}]'))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert any("2 entries" in record.getMessage() for record in caplog.records)


async def test_a_scraped_page_is_logged_by_its_character_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Markdown is not an array, so its size is a character count."""
    caplog.set_level(logging.DEBUG)
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        await toolkit.call(SCRAPE_AS_MARKDOWN, {"url": A_URL})

    assert any("8 characters" in record.getMessage() for record in caplog.records)


async def test_a_cache_hit_says_so_in_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run that looks slow should show where the calls did not happen."""
    caplog.set_level(logging.DEBUG)
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert sum("cache hit" in record.getMessage() for record in caplog.records) == 1


async def test_a_result_is_not_parsed_when_debug_is_off(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sizing a result parses it, and a run above DEBUG never reads that size."""
    caplog.set_level(logging.INFO)
    parsed: list[str] = []
    real_loads = json.loads

    def counting_loads(text: str) -> object:
        parsed.append(text)
        return real_loads(text)

    monkeypatch.setattr(json, "loads", counting_loads)
    search = FakeTool(SEARCH_ENGINE, always('[{"url": "a"}]'))
    scrape = FakeTool(SCRAPE_AS_MARKDOWN, always("# A page"))

    async with opening(offering(search, scrape)) as toolkit:
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})

    assert parsed == []


async def test_no_log_record_carries_the_bright_data_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The endpoint carries the token, so nothing built from it may be logged."""
    caplog.set_level(logging.DEBUG)
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        await toolkit.call(SEARCH_ENGINE, {"query": "who"})
        await toolkit.call(SCRAPE_AS_MARKDOWN, {"url": A_URL})

    messages = [record.getMessage() for record in caplog.records]
    assert messages
    assert all(BRIGHT_DATA_CREDENTIAL not in message for message in messages)


async def test_the_toolkit_binds_the_tool_objects_the_server_gave() -> None:
    """The model fills each tool's own schema, so the objects pass through."""
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        assert isinstance(toolkit, Toolkit)
        assert toolkit.bound_tools == [search, scrape]


async def test_the_scrub_seam_hides_the_token_in_any_message_built_over_it() -> None:
    """The agent reports failures the toolkit re-raised, and they quote the URL."""
    search, scrape = both_tools()

    async with opening(offering(search, scrape)) as toolkit:
        reported = toolkit.without_the_token(str(quoting_the_tokened_url(500)))

    assert BRIGHT_DATA_CREDENTIAL not in reported
    assert f"{BRIGHT_DATA_ENDPOINT}?token=***" in reported
