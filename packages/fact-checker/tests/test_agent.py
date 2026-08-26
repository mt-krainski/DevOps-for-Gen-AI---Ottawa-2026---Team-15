"""One statement's checking loop: the budget it spends and the ruling it returns."""

import logging
from functools import partial

import openai
import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

from fact_checker import agent
from fact_checker.agent import (
    BUDGET_NOTICE_TEMPLATE,
    BUDGET_SPENT_NOTICE,
    AgentRun,
    build_models,
    check_one,
)
from fact_checker.cache import RunCache
from fact_checker.errors import AuthenticationFailure, ErrorCode, StatementFailure
from fact_checker.models import Classification, InputStatement, Reference, Ruling
from fact_checker.retry import with_retry
from fact_checker.tools import SCRAPE_AS_MARKDOWN, SEARCH_ENGINE, Toolkit
from tests.conftest import (
    BRIGHT_DATA_CREDENTIAL,
    BRIGHT_DATA_ENDPOINT,
    FakeCheckingModel,
    FakeRulingModel,
    FakeToolkit,
    make_config,
    openai_status_error,
    quoting_the_tokened_url,
)

A_QUERY = "who won the 2024 championship"
A_URL = "https://example.test/2024-final"

A_STATEMENT = InputStatement(
    id=None,
    surrounding_context="The report reviewed the 2024 season match by match.",
    statement="The team won the 2024 championship.",
    classification=Classification(**{"class": "fact", "confidence": 0.9}),
)


def a_ruling(verdict: str = "supported") -> Ruling:
    """Build the ruling a run comes back with."""
    return Ruling(
        verdict=verdict,
        confidence=0.8,
        justification="Both reports name the same winner [1].",
        references=[Reference(id="1", source=A_URL, excerpt="They won the final.")],
    )


def a_search_call(call_id: str = "c1", query: str = A_QUERY) -> ToolCall:
    """Build one `search_engine` call the way a model asks for it."""
    return {
        "name": SEARCH_ENGINE,
        "args": {"query": query},
        "id": call_id,
        "type": "tool_call",
    }


def a_scrape_call(call_id: str = "c2", url: str = A_URL) -> ToolCall:
    """Build one `scrape_as_markdown` call the way a model asks for it."""
    return {
        "name": SCRAPE_AS_MARKDOWN,
        "args": {"url": url},
        "id": call_id,
        "type": "tool_call",
    }


def a_turn(*calls: ToolCall, tokens: tuple[int, int] = (0, 0)) -> AIMessage:
    """Build one checking-model answer, with the tool calls it asks for."""
    prompt_tokens, completion_tokens = tokens
    return AIMessage(
        content="",
        tool_calls=list(calls),
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def a_ruling_result(
    ruling: Ruling, *, tokens: tuple[int, int] = (0, 0)
) -> dict[str, object]:
    """Build the `include_raw` result a structured-output model returns."""
    return {"raw": a_turn(tokens=tokens), "parsed": ruling, "parsing_error": None}


def text_of(message: BaseMessage) -> str:
    """Return one message's content as the text a model would read."""
    return str(message.content)


@pytest.fixture
def unwaited_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real retry policy, and take the real time out of its delays."""

    async def no_wait(_delay: float) -> None:
        return None

    monkeypatch.setattr(
        agent, "with_retry", partial(with_retry, sleep=no_wait, jitter=lambda: 0.0)
    )


async def run_check(
    *,
    toolkit: FakeToolkit,
    checking_model: FakeCheckingModel,
    ruling_model: FakeRulingModel,
    budget: int = 10,
    identifier: str = "s1",
) -> AgentRun:
    """Check the shared statement, naming only what the test varies."""
    return await check_one(
        A_STATEMENT,
        identifier,
        toolkit=toolkit,
        checking_model=checking_model,
        ruling_model=ruling_model,
        budget=budget,
    )


async def test_a_run_searches_scrapes_and_then_rules() -> None:
    """Two tools in one turn, a turn with nothing left to ask, then the ruling."""
    toolkit = FakeToolkit(["1. A report", "# The final"])
    checking_model = FakeCheckingModel(
        [
            a_turn(a_search_call(), a_scrape_call(), tokens=(100, 10)),
            a_turn(tokens=(200, 20)),
        ]
    )
    ruling = a_ruling()
    ruling_model = FakeRulingModel([a_ruling_result(ruling, tokens=(300, 30))])

    run = await run_check(
        toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
    )

    assert [name for name, _ in toolkit.calls] == [SEARCH_ENGINE, SCRAPE_AS_MARKDOWN]
    assert toolkit.calls[0][1] == {"query": A_QUERY}
    assert run.ruling is ruling
    assert run.tool_calls_used == 2
    assert run.prompt_tokens == 600
    assert run.completion_tokens == 60


async def test_a_first_turn_without_tool_calls_leaves_the_loop() -> None:
    """A statement the model can rule on unaided costs no tool call at all."""
    toolkit = FakeToolkit(["never asked for"])
    checking_model = FakeCheckingModel([a_turn(tokens=(50, 5))])
    ruling_model = FakeRulingModel([a_ruling_result(a_ruling("unverifiable"))])

    run = await run_check(
        toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
    )

    assert toolkit.calls == []
    assert run.tool_calls_used == 0
    assert run.ruling.verdict == "unverifiable"


async def test_the_opening_prompt_carries_the_context_and_the_statement() -> None:
    """The context is there to make the claim searchable, so the model sees it."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel([a_turn()])
    ruling_model = FakeRulingModel([a_ruling_result(a_ruling())])

    await run_check(
        toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
    )

    opening = text_of(checking_model.prompts[0][-1])
    assert A_STATEMENT.surrounding_context in opening
    assert A_STATEMENT.statement in opening
    for verdict in ("supported", "refuted", "mixed", "unverifiable"):
        assert verdict in opening


async def test_the_budget_notice_counts_the_remaining_calls_down() -> None:
    """The agent plans against the number left, rather than being cut off."""
    toolkit = FakeToolkit(["1. A report", "1. Another", "1. A third"])
    checking_model = FakeCheckingModel(
        [
            a_turn(a_search_call("c1")),
            a_turn(a_search_call("c2")),
            a_turn(a_search_call("c3")),
        ]
    )
    ruling_model = FakeRulingModel([a_ruling_result(a_ruling())])

    run = await run_check(
        toolkit=toolkit,
        checking_model=checking_model,
        ruling_model=ruling_model,
        budget=3,
    )

    notices = [text_of(prompt[-1]) for prompt in checking_model.prompts]
    assert len(notices) == 3
    for notice, remaining in zip(notices, (3, 2, 1), strict=True):
        assert BUDGET_NOTICE_TEMPLATE.format(remaining=remaining) in notice
    assert run.tool_calls_used == 3


async def test_a_turn_reporting_no_usage_contributes_nothing() -> None:
    """A model that reports no usage costs zero rather than failing the run."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel([AIMessage(content="")])
    ruling_model = FakeRulingModel(
        [{"raw": AIMessage(content=""), "parsed": a_ruling(), "parsing_error": None}]
    )

    run = await run_check(
        toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
    )

    assert run.prompt_tokens == 0
    assert run.completion_tokens == 0


async def test_a_tool_call_over_the_budget_is_never_made() -> None:
    """The turn asks for two calls with one left, so the second is refused."""
    toolkit = FakeToolkit(["1. A report"])
    checking_model = FakeCheckingModel(
        [a_turn(a_search_call("c1"), a_scrape_call("c2"))]
    )
    ruling_model = FakeRulingModel([a_ruling_result(a_ruling("unverifiable"))])

    run = await run_check(
        toolkit=toolkit,
        checking_model=checking_model,
        ruling_model=ruling_model,
        budget=1,
    )

    assert [name for name, _ in toolkit.calls] == [SEARCH_ENGINE]
    refused = [
        message
        for message in ruling_model.prompts[0]
        if isinstance(message, ToolMessage) and message.tool_call_id == "c2"
    ]
    assert [text_of(message) for message in refused] == [BUDGET_SPENT_NOTICE]
    assert run.tool_calls_used == 1
    assert run.ruling.verdict == "unverifiable"


async def test_a_transient_model_failure_is_retried_and_then_succeeds(
    unwaited_retries: None,
) -> None:
    """A rate limit costs a wait, not the statement."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel(
        [openai_status_error(openai.RateLimitError, 429), a_turn()]
    )
    ruling_model = FakeRulingModel([a_ruling_result(a_ruling())])

    run = await run_check(
        toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
    )

    assert len(checking_model.prompts) == 2
    assert run.ruling.verdict == "supported"


async def test_a_rejected_bright_data_token_ends_the_run() -> None:
    """The toolkit already classified it, so it leaves `check_one` unconverted."""
    toolkit = FakeToolkit([AuthenticationFailure("bright data rejected the token")])
    checking_model = FakeCheckingModel([a_turn(a_search_call())])
    ruling_model = FakeRulingModel([])

    with pytest.raises(AuthenticationFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.AUTH_ERROR
    assert ruling_model.prompts == []


async def test_a_rejected_key_on_the_checking_model_ends_the_run() -> None:
    """Fifty identical statement errors under exit code zero is the wrong report."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel(
        [openai_status_error(openai.AuthenticationError, 401)]
    )
    ruling_model = FakeRulingModel([])

    with pytest.raises(AuthenticationFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.AUTH_ERROR


async def test_a_rejected_key_on_the_ruling_model_ends_the_run() -> None:
    """The second model reaches the same gateway, and fails the same way."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel([a_turn()])
    ruling_model = FakeRulingModel(
        [openai_status_error(openai.AuthenticationError, 401)]
    )

    with pytest.raises(AuthenticationFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.AUTH_ERROR


async def test_a_ruling_that_would_not_parse_fails_the_statement() -> None:
    """The structured-output model reports what it could not read."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel([a_turn()])
    ruling_model = FakeRulingModel(
        [
            {
                "raw": a_turn(),
                "parsed": None,
                "parsing_error": ValueError("verdict 'probably' is not a verdict"),
            }
        ]
    )

    with pytest.raises(StatementFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.PARSE_ERROR
    assert "verdict 'probably' is not a verdict" in raised.value.message


async def test_a_ruling_that_is_not_a_ruling_fails_the_statement() -> None:
    """Nothing raised, and nothing came back that the output could carry."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel([a_turn()])
    ruling_model = FakeRulingModel(
        [{"raw": a_turn(), "parsed": {"verdict": "supported"}, "parsing_error": None}]
    )

    with pytest.raises(StatementFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.PARSE_ERROR
    assert "dict" in raised.value.message


async def test_a_failure_that_outlived_the_retries_fails_the_statement() -> None:
    """One statement fails and the run carries on with the other forty-nine."""
    toolkit = FakeToolkit([])
    checking_model = FakeCheckingModel(
        [openai_status_error(openai.BadRequestError, 400)]
    )
    ruling_model = FakeRulingModel([])

    with pytest.raises(StatementFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.AGENT_ERROR


async def test_a_tool_failure_keeps_the_code_the_toolkit_gave_it() -> None:
    """`TOOL_ERROR` says where it went wrong, and the catch-all would lose that."""
    toolkit = FakeToolkit(
        [StatementFailure(ErrorCode.TOOL_ERROR, "search_engine reported a failure")]
    )
    checking_model = FakeCheckingModel([a_turn(a_search_call())])
    ruling_model = FakeRulingModel([])

    with pytest.raises(StatementFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.TOOL_ERROR
    assert raised.value.message == "search_engine reported a failure"


async def test_a_reported_failure_never_carries_the_bright_data_token() -> None:
    """The transport quotes the request URL, and the token rides in it."""
    toolkit = FakeToolkit([quoting_the_tokened_url(500)])
    checking_model = FakeCheckingModel([a_turn(a_search_call())])
    ruling_model = FakeRulingModel([])

    with pytest.raises(StatementFailure) as raised:
        await run_check(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )

    assert raised.value.code is ErrorCode.AGENT_ERROR
    assert BRIGHT_DATA_CREDENTIAL not in raised.value.message
    assert f"{BRIGHT_DATA_ENDPOINT}?token=***" in raised.value.message


async def test_the_logged_failure_never_carries_the_bright_data_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The toolkit chains the tokened original, so the chain is scrubbed too."""
    caplog.set_level(logging.DEBUG)
    failure = StatementFailure(ErrorCode.TOOL_ERROR, "search_engine reported a failure")
    failure.__cause__ = quoting_the_tokened_url(500)
    toolkit = FakeToolkit([failure])
    checking_model = FakeCheckingModel([a_turn(a_search_call())])
    ruling_model = FakeRulingModel([])

    with pytest.raises(StatementFailure):
        await run_check(
            toolkit=toolkit,
            checking_model=checking_model,
            ruling_model=ruling_model,
            identifier="s7",
        )

    logged = caplog.text
    assert "s7" in logged
    assert BRIGHT_DATA_CREDENTIAL not in logged
    assert f"{BRIGHT_DATA_ENDPOINT}?token=***" in logged


def a_toolkit_over_real_tools() -> Toolkit:
    """Build a toolkit over two real tools, with no server behind them."""

    @tool
    def search_engine(query: str) -> str:
        """Search the web for a query."""
        return ""

    @tool
    def scrape_as_markdown(url: str) -> str:
        """Fetch one page as markdown."""
        return ""

    return Toolkit([search_engine, scrape_as_markdown], make_config(), RunCache())


def test_the_checking_model_is_the_configured_model_with_both_tools_bound() -> None:
    """The model can only ask for a tool it was told about."""
    config = make_config()

    checking_model, _ = build_models(config, a_toolkit_over_real_tools())

    bound = [tool["function"]["name"] for tool in checking_model.kwargs["tools"]]
    assert bound == [SEARCH_ENGINE, SCRAPE_AS_MARKDOWN]
    assert checking_model.bound.model_name == config.model
    assert checking_model.bound.openai_api_base == config.base_url


def test_the_ruling_model_asks_for_a_ruling_beside_the_raw_response() -> None:
    """The raw response is where the ruling call's own token usage is reported."""
    config = make_config()

    _, ruling_model = build_models(config, a_toolkit_over_real_tools())

    # `include_raw=True` builds a parallel step that keeps the model's own answer
    # under `raw`, and that branch is where the schema is asked for.
    raw_branch = ruling_model.steps[0].steps__["raw"]
    assert raw_branch.kwargs["response_format"] is Ruling
    assert raw_branch.bound.model_name == config.model
