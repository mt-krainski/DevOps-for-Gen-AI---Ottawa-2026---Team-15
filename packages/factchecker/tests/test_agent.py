"""Tests for the searching agent in `factchecker.agent`.

No test here reaches OpenRouter or Bright Data. Most drive a scripted stand-in model
that answers a turn at a time and records what it was asked, over real
`StructuredTool` objects built the way `instrument` builds them: a JSON-schema `dict`
for `args_schema`, and an answer returned as text.

The scripts are keyed on the claim rather than on call order, which is what lets one
`AgentChecker` serve two statements at once in the concurrency test while each keeps
its own place in its own script.

A stand-in model cannot see what the real client does with what the checker bound to
it, and the three things that path gets wrong — a tool the OpenAI SDK refuses to send,
a ruling the SDK parses before this package can, and a schema on a request that was
meant to carry a tool call — reach a stand-in as nothing at all. So the tests at the
end of this file drive a real `ChatOpenAI` over an `httpx.MockTransport`. That
exercises payload construction, tool validation, response parsing and the usage
object, and it opens no socket.
"""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Self

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from factchecker.agent import MALFORMED_RULING, AgentChecker
from factchecker.cache import RunCache
from factchecker.checker import StatementChecker
from factchecker.config import OPENROUTER_BASE_URL, McpEndpoint, Settings
from factchecker.errors import AuthenticationFailed, CheckFailed, McpCallError
from factchecker.models import IdentifiedStatement, InputPayload, Ruling
from factchecker.run import RunSettings, run_check
from factchecker.tools import PAGE_TOOL_NAME, SEARCH_TOOL_NAME, instrument
from tests.conftest import wire_statement

CLAIM = "Water boils at 100 C"
OTHER_CLAIM = "Mercury freezes at minus 39 C"

STARTED_AT = datetime(2026, 8, 22, 14, 3, 11, tzinfo=UTC)

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

# The two catalogue entries as Bright Data publishes them. The arguments past the
# first are the ones the run cache cannot key on.
SERVER_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "country": {"type": "string"},
        "num_results": {"type": "integer"},
    },
    "required": ["query"],
}
SERVER_PAGE_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string"}, "data_format": {"type": "string"}},
    "required": ["url"],
}
MALFORMED_ANSWER = "verdict: probably true"


def _ruling_text(verdict: str = "supported", justification: str = "Agreed [1].") -> str:
    """A ruling as the model writes it: one JSON object and nothing else."""
    return json.dumps(
        {
            "verdict": verdict,
            "confidence": 0.9,
            "justification": justification,
            "references": [
                {
                    "id": "1",
                    "source": "https://example.test/boiling",
                    "excerpt": "Water boils at 100 degrees Celsius at sea level.",
                }
            ],
        }
    )


PARTIAL_RULING_TEXT = json.dumps(
    {"verdict": "supported", "confidence": 0.9, "justification": "Agreed [1]."}
)


def _ai(
    content: str = "",
    tool_calls: Sequence[ToolCall] = (),
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> AIMessage:
    """One model answer, carrying the usage object OpenRouter returns beside it."""
    return AIMessage(
        content=content,
        tool_calls=list(tool_calls),
        response_metadata={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        },
    )


def _done(prompt_tokens: int = 0, completion_tokens: int = 0) -> AIMessage:
    """The answer that ends the searching turns: no tool call, and no ruling either.

    The searching turns end when the model stops asking for a tool. The ruling is
    asked for after that, in a request of its own.
    """
    return _ai(
        "I have the evidence I need.",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _search(query: str = "boiling point of water", call_id: str = "c1") -> ToolCall:
    """A request for one search, as the model writes it."""
    return tool_call(name=SEARCH_TOOL_NAME, args={"query": query}, id=call_id)


def _read(url: str = "https://example.test/boiling", call_id: str = "c2") -> ToolCall:
    """A request to read one page, as the model writes it."""
    return tool_call(name=PAGE_TOOL_NAME, args={"url": url}, id=call_id)


class _Model:
    """A chat model that answers from the script belonging to the claim it is given.

    Each `ainvoke` yields to the event loop before it answers, so two statements
    checked at once really do interleave rather than run one after the other.
    """

    def __init__(self, scripts: Mapping[str, Sequence[AIMessage]]) -> None:
        self.scripts = scripts
        self.turns: list[list[BaseMessage]] = []
        self.bindings: list[dict[str, object]] = []
        self._served: dict[str, int] = {}

    def bind_tools(self, tools: Sequence[BaseTool], **bound: object) -> Self:
        """Record one binding the checker made, and stand in for the bound model.

        The checker makes two, and a stand-in that answers from a script tells them
        apart by what each was bound with rather than by which object came back.
        """
        self.bindings.append({"tools": list(tools), **bound})
        return self

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Answer the next line of this claim's script, after yielding to the loop."""
        for _ in range(3):
            await asyncio.sleep(0)
        self.turns.append(list(messages))
        claim = self._claim(messages)
        served = self._served.get(claim, 0)
        self._served[claim] = served + 1
        script = self.scripts[claim]
        return script[min(served, len(script) - 1)]

    def turns_about(self, claim: str) -> list[list[BaseMessage]]:
        """Every turn this model was asked about the named claim, in order."""
        return [turn for turn in self.turns if self._claim(turn) == claim]

    def claim_order(self) -> list[str]:
        """The claim each turn was about, in the order the turns arrived."""
        return [self._claim(turn) for turn in self.turns]

    def _claim(self, messages: Sequence[BaseMessage]) -> str:
        """Read which claim a turn is about from the prompts it carries."""
        written = " ".join(str(message.content) for message in messages)
        return next(claim for claim in self.scripts if claim in written)


class _Tools:
    """The two instrumented tools as the agent sees them, recording every call."""

    def __init__(self, search: str | Exception = "one result") -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.tools = [
            self._built(SEARCH_TOOL_NAME, SEARCH_SCHEMA, search),
            self._built(PAGE_TOOL_NAME, PAGE_SCHEMA, "# Boiling"),
        ]

    def named(self, name: str) -> list[dict[str, object]]:
        """The arguments of every call made to the named tool, in order."""
        return [arguments for called, arguments in self.calls if called == name]

    def _built(
        self, name: str, schema: dict[str, object], answer: str | Exception
    ) -> BaseTool:
        """One tool of the given name, answering with text or raising."""

        async def run(**arguments: object) -> str:
            self.calls.append((name, arguments))
            if isinstance(answer, Exception):
                raise answer
            return answer

        return StructuredTool(
            name=name,
            description=f"the {name} tool",
            args_schema=schema,
            coroutine=run,
        )


def _settings(tool_call_budget: int = 10) -> Settings:
    """Settings whose only field this task reads is the budget."""
    return Settings(
        openrouter_api_key="not-a-real-key",
        model="google/gemma-4-31b-it",
        mcp_endpoint=McpEndpoint("not-a-real-token"),
        tool_call_budget=tool_call_budget,
        page_character_ceiling=100000,
        concurrency=8,
        statement_timeout_seconds=240.0,
        retry_attempts=3,
    )


async def _no_sleep(seconds: float) -> None:
    """Wait none of the backoff a retry would otherwise sleep through."""


def _real_tools(answer: str = "one result") -> list[BaseTool]:
    """The two tools as `instrument` builds them, over a server that always answers.

    These carry the real wrappers, so a call whose arguments do not fit meets the
    real signature rather than a fake that accepts anything.
    """

    async def call(**arguments: object) -> list[dict[str, str]]:
        return [{"type": "text", "text": answer}]

    published = [
        StructuredTool(
            name=SEARCH_TOOL_NAME,
            description=f"the {SEARCH_TOOL_NAME} tool",
            args_schema=SERVER_SEARCH_SCHEMA,
            coroutine=call,
        ),
        StructuredTool(
            name=PAGE_TOOL_NAME,
            description=f"the {PAGE_TOOL_NAME} tool",
            args_schema=SERVER_PAGE_SCHEMA,
            coroutine=call,
        ),
    ]
    return instrument(published, RunCache(), _settings(), _no_sleep)


def _wire_tool_call(name: str, **arguments: object) -> dict[str, object]:
    """A tool call as a gateway writes one: the arguments as a JSON string."""
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class _Gateway:
    """OpenRouter's chat-completions endpoint, answered from a script over `httpx`.

    An answer is the text the model wrote, or the tool calls it asked for. Every
    request body is kept, so a test reads what the pinned client really put on the
    wire. The last answer stands for every request after it, which is what lets a
    model that never rules be written as one malformed answer.
    """

    def __init__(
        self,
        *answers: str | list[dict[str, object]],
        prompt: int = 11,
        completion: int = 22,
    ) -> None:
        self.answers = answers
        self.prompt = prompt
        self.completion = completion
        self.sent: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record the body and answer the next line of the script."""
        self.sent.append(json.loads(request.content))
        answer = self.answers[min(len(self.sent) - 1, len(self.answers) - 1)]
        asked = isinstance(answer, list)
        message = (
            {"role": "assistant", "content": None, "tool_calls": answer}
            if asked
            else {"role": "assistant", "content": answer}
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "google/gemma-4-31b-it",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if asked else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": self.prompt,
                    "completion_tokens": self.completion,
                    "total_tokens": self.prompt + self.completion,
                },
            },
        )

    def model(self) -> ChatOpenAI:
        """The real client this package builds, pointed at this transport."""
        return _client(self)

    def tool_named(self, name: str, request: int = 0) -> dict[str, object]:
        """One request's entry for the named tool, as it went on the wire."""
        return next(
            one["function"]
            for one in self.sent[request]["tools"]
            if one["function"]["name"] == name
        )


class _Refusing:
    """OpenRouter's endpoint, refusing every request with the status it was given.

    Every request body is kept, so a test reads how many the refusal cost.
    """

    def __init__(self, status: int) -> None:
        self.status = status
        self.sent: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record the body, and answer with the refusal and nothing else."""
        self.sent.append(json.loads(request.content))
        return httpx.Response(
            self.status,
            json={"error": {"message": "no auth credentials found"}},
        )

    def model(self) -> ChatOpenAI:
        """The real client this package builds, pointed at this transport."""
        return _client(self)


def _client(answer: Callable[[httpx.Request], httpx.Response]) -> ChatOpenAI:
    """The real client this package builds, pointed at a transport that answers itself.

    The key is a literal that no service would accept, and no request leaves the
    process to find out.
    """
    return ChatOpenAI(
        model="google/gemma-4-31b-it",
        base_url=OPENROUTER_BASE_URL,
        api_key=SecretStr("not-a-real-key"),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )


def _statement(claim: str = CLAIM, identifier: str = "s1") -> IdentifiedStatement:
    """One identified statement, whose context names only its own claim.

    Each statement's context has to be its own, because the scripted model reads the
    claim out of the prompts to know whose turn it is answering.
    """
    return IdentifiedStatement(
        id=identifier,
        surrounding_context=f"The tables were checked. {claim}. They agree.",
        statement=claim,
        classification={"class": "fact", "confidence": 0.7},
    )


def _reminder(turn: Sequence[BaseMessage]) -> str:
    """The message a turn rides on, which is always the last one it carries."""
    return str(turn[-1].content)


def _answer(turn: Sequence[BaseMessage]) -> str:
    """The one tool answer a turn carries.

    Read by scanning rather than by position: the searching turns that follow a tool
    call leave their own answers in the conversation behind it.
    """
    answers = [one for one in turn if isinstance(one, ToolMessage)]
    assert len(answers) == 1
    return str(answers[0].content)


def _unanswered(turn: Sequence[BaseMessage]) -> list[str]:
    """The tool calls in a turn that no later message in it replies to.

    A gateway refuses a conversation holding one of these, so every path through the
    loop has to leave the list empty.
    """
    answered = {
        message.tool_call_id for message in turn if isinstance(message, ToolMessage)
    }
    return [
        call["id"]
        for message in turn
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["id"] not in answered
    ]


def test_the_agent_checker_satisfies_the_statement_checker_protocol() -> None:
    """The orchestrator finds a `check` on it, by that name and no other."""
    checker = AgentChecker(
        _Model({CLAIM: [_ai(_ruling_text())]}), _Tools().tools, _settings()
    )

    assert isinstance(checker, StatementChecker)


def test_the_searching_binding_carries_the_tools_and_no_ruling_schema() -> None:
    """The defect this file guards: a schema on a searching turn forbids a tool call.

    A `json_schema` `response_format` and a tool call are mutually exclusive on one
    request. The answer is constrained to the ruling schema, and a tool call is not a
    member of that schema, so a model bound to it on every turn never searches at all.
    """
    model = _Model({CLAIM: [_ai(_ruling_text())]})
    tools = _Tools()

    AgentChecker(model, tools.tools, _settings())

    searching = model.bindings[0]
    assert searching["tools"] == tools.tools
    assert "response_format" not in searching
    assert "tool_choice" not in searching


def test_the_ruling_binding_carries_the_schema_and_forbids_a_tool_call() -> None:
    """The other half of the split, and what keeps a tool call out of the ruling turn.

    The ruling schema is bound as a schema document rather than as the `Ruling` class.
    Given the class, the OpenAI client parses the answer itself and a malformed one
    raises out of `ainvoke`, where the retry this package owns cannot reach it.

    The tools are bound here as well, because the conversation this turn is sent holds
    their calls and their answers. `tool_choice` is what stops another being asked
    for.
    """
    model = _Model({CLAIM: [_ai(_ruling_text())]})
    tools = _Tools()

    AgentChecker(model, tools.tools, _settings())

    ruling = model.bindings[1]
    assert ruling["tools"] == tools.tools
    assert ruling["tool_choice"] == "none"
    response_format = ruling["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Ruling"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["required"] == list(
        Ruling.model_fields
    )
    assert "strict" not in ruling


def test_both_bindings_are_made_once_rather_than_per_statement() -> None:
    """The checker holds nothing that changes, so two checks bind nothing further."""
    model = _Model({CLAIM: [_done(), _ai(_ruling_text())]})
    checker = AgentChecker(model, _Tools().tools, _settings())

    asyncio.run(checker.check(_statement()))
    asyncio.run(checker.check(_statement(identifier="s2")))

    assert len(model.bindings) == 2


def test_a_check_that_searches_once_returns_the_ruling_and_what_it_cost() -> None:
    """The happy path: one search, one ruling, and the usage of all three turns."""
    model = _Model(
        {
            CLAIM: [
                _ai(tool_calls=[_search()], prompt_tokens=110, completion_tokens=12),
                _done(prompt_tokens=200, completion_tokens=5),
                _ai(_ruling_text(), prompt_tokens=340, completion_tokens=88),
            ]
        }
    )
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert outcome.ruling.references[0].source == "https://example.test/boiling"
    assert outcome.searches == 1
    assert outcome.prompt_tokens == 650
    assert outcome.completion_tokens == 105
    assert tools.named(SEARCH_TOOL_NAME) == [{"query": "boiling point of water"}]


def test_token_counts_add_up_across_every_turn_including_the_ruling_turn() -> None:
    """The run's cost is every turn's cost, and the ruling turn is a turn.

    The ruling is a request of its own, so its tokens are spend like any other. A
    count that dropped them would under-report every check by a whole turn, and by
    the turn that carries the longest conversation of the check.
    """
    model = _Model(
        {
            CLAIM: [
                _ai(tool_calls=[_search()], prompt_tokens=100, completion_tokens=10),
                _ai(tool_calls=[_read()], prompt_tokens=200, completion_tokens=20),
                _done(prompt_tokens=300, completion_tokens=30),
                _ai(_ruling_text(), prompt_tokens=400, completion_tokens=40),
            ]
        }
    )
    checker = AgentChecker(model, _Tools().tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.prompt_tokens == 1000
    assert outcome.completion_tokens == 100


def test_a_turn_that_reports_no_usage_costs_nothing_rather_than_failing() -> None:
    """A gateway that omits its usage object loses the count, not the check."""
    model = _Model({CLAIM: [AIMessage(content=_ruling_text())]})
    checker = AgentChecker(model, _Tools().tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.prompt_tokens == 0
    assert outcome.completion_tokens == 0


def test_only_the_search_tool_counts_towards_the_search_count() -> None:
    """Reading a page spends the budget, but it is not a search."""
    model = _Model(
        {
            CLAIM: [
                _ai(tool_calls=[_search()]),
                _ai(tool_calls=[_read(), _read(call_id="c3")]),
                _done(),
                _ai(_ruling_text()),
            ]
        }
    )
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.searches == 1
    assert len(tools.named(PAGE_TOOL_NAME)) == 2


def test_each_searching_turn_carries_a_budget_reminder_naming_what_is_left() -> None:
    """The reminder rides on the turn, and the count it names moves with the spend."""
    model = _Model(
        {
            CLAIM: [
                _ai(tool_calls=[_search()]),
                _ai(tool_calls=[_read()]),
                _done(),
                _ai(_ruling_text()),
            ]
        }
    )
    checker = AgentChecker(model, _Tools().tools, _settings(tool_call_budget=5))

    asyncio.run(checker.check(_statement()))

    first, second, third, _ = model.turns_about(CLAIM)
    assert "0 of 5" in _reminder(first)
    assert "1 of 5" in _reminder(second)
    assert "2 of 5" in _reminder(third)


def test_the_ruling_turn_asks_for_the_ruling_rather_than_reporting_the_budget() -> None:
    """The last turn is a different request, and it carries a different message."""
    model = _Model({CLAIM: [_ai(tool_calls=[_search()]), _done(), _ai(_ruling_text())]})
    checker = AgentChecker(model, _Tools().tools, _settings(tool_call_budget=5))

    asyncio.run(checker.check(_statement()))

    ruling_turn = model.turns_about(CLAIM)[-1]
    assert "Rule now" in _reminder(ruling_turn)
    assert "tool calls used" not in _reminder(ruling_turn)
    answered = [one for one in ruling_turn if isinstance(one, ToolMessage)]
    assert [str(one.content) for one in answered] == ["one result"]


def test_a_stale_reminder_is_not_kept_in_the_conversation() -> None:
    """One message per turn: a history of stale counts is a history that misleads."""
    model = _Model({CLAIM: [_ai(tool_calls=[_search()]), _done(), _ai(_ruling_text())]})
    checker = AgentChecker(model, _Tools().tools, _settings(tool_call_budget=5))

    asyncio.run(checker.check(_statement()))

    turns = model.turns_about(CLAIM)
    written = [str(message.content) for message in turns[1]]
    assert len([one for one in written if "of 5 tool calls used" in one]) == 1
    kept = [str(message.content) for message in turns[-1][:-1]]
    assert [one for one in kept if "of 5 tool calls used" in one] == []


def test_the_budget_stops_the_tool_calls_and_the_agent_still_rules() -> None:
    """Reaching the budget is not a failure: the agent is asked to rule on what it has.

    A model that would keep searching is not cut off with an error. The searching
    turns stop, and the ruling turn follows them as it follows any other ending.
    """
    model = _Model(
        {
            CLAIM: [
                _ai(tool_calls=[_search()]),
                _ai(tool_calls=[_search(call_id="c2")]),
                _ai(_ruling_text(verdict="unverifiable")),
            ]
        }
    )
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings(tool_call_budget=2))

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "unverifiable"
    assert len(tools.named(SEARCH_TOOL_NAME)) == 2
    assert len(model.turns_about(CLAIM)) == 3
    assert "Rule now" in _reminder(model.turns_about(CLAIM)[-1])


def test_a_tool_call_beyond_the_budget_is_refused_rather_than_made() -> None:
    """A turn asking for more calls than remain gets an answer to each, all the same.

    Every tool call an assistant message carries needs a reply of its own, so the
    calls the budget refuses are answered rather than dropped.
    """
    model = _Model(
        {
            CLAIM: [
                _ai(
                    tool_calls=[
                        _search(call_id="c1"),
                        _search(query="boiling", call_id="c2"),
                        _search(query="steam", call_id="c3"),
                    ]
                ),
                _ai(_ruling_text()),
            ]
        }
    )
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings(tool_call_budget=2))

    asyncio.run(checker.check(_statement()))

    assert len(tools.named(SEARCH_TOOL_NAME)) == 2
    answers = [
        message
        for message in model.turns_about(CLAIM)[-1]
        if isinstance(message, ToolMessage)
    ]
    assert len(answers) == 3
    assert "budget" in str(answers[-1].content)


def test_a_call_to_a_tool_that_does_not_exist_is_answered_rather_than_raised() -> None:
    """A hallucinated tool name costs one call and a correction, not the statement."""
    model = _Model(
        {
            CLAIM: [
                _ai(tool_calls=[tool_call(name="browse", args={}, id="c1")]),
                _done(),
                _ai(_ruling_text()),
            ]
        }
    )
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert tools.calls == []
    assert SEARCH_TOOL_NAME in _answer(model.turns_about(CLAIM)[-1])


@pytest.mark.parametrize("arguments", [{}, {"queries": ["boiling"]}])
def test_a_call_whose_arguments_do_not_fit_is_answered_rather_than_raised(
    arguments: dict[str, object],
) -> None:
    """A hallucinated argument costs a call and a correction, not the statement.

    A hallucinated tool *name* is already answered in writing. The same mistake one
    field over reaches the wrapper's own signature, because a `StructuredTool` on a
    JSON-schema `args_schema` validates no input, and the `TypeError` there would
    leave `check` and end the statement as `check_failed`.
    """
    model = _Model(
        {
            CLAIM: [
                _ai(
                    tool_calls=[
                        tool_call(name=SEARCH_TOOL_NAME, args=arguments, id="c1")
                    ]
                ),
                _done(),
                _ai(_ruling_text()),
            ]
        }
    )
    checker = AgentChecker(model, _real_tools(), _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert "query" in _answer(model.turns_about(CLAIM)[-1])


def test_a_malformed_ruling_is_retried_once_with_the_validation_error() -> None:
    """The model is shown what it got wrong, so its second try can be different.

    The retry is a ruling turn of its own, and the correction joins the conversation
    it is made over: the model reads its own rejected answer above the reason.
    """
    model = _Model({CLAIM: [_done(), _ai(MALFORMED_ANSWER), _ai(_ruling_text())]})
    checker = AgentChecker(model, _Tools().tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    correction = str(model.turns_about(CLAIM)[-1][-2].content)
    assert "not a valid ruling" in correction
    assert MALFORMED_ANSWER in correction


def test_a_second_malformed_ruling_fails_the_statement_by_its_own_kind() -> None:
    """Two malformed rulings end the check, and the failure names itself."""
    model = _Model({CLAIM: [_ai("verdict: probably true")]})
    checker = AgentChecker(model, _Tools().tools, _settings())

    with pytest.raises(CheckFailed) as raised:
        asyncio.run(checker.check(_statement()))

    assert raised.value.kind == MALFORMED_RULING


def test_a_malformed_ruling_message_repeats_nothing_an_upstream_library_wrote() -> None:
    """The message reaches the output payload, so this package writes all of it."""
    model = _Model({CLAIM: [_ai("verdict: probably true")]})
    checker = AgentChecker(model, _Tools().tools, _settings())

    with pytest.raises(CheckFailed) as raised:
        asyncio.run(checker.check(_statement()))

    assert "validation error" not in raised.value.message
    assert "verdict: probably true" not in raised.value.message


def test_a_ruling_missing_its_references_is_rejected_rather_than_repaired() -> None:
    """A ruling with no references is not a weaker ruling; it is a different claim."""
    model = _Model({CLAIM: [_ai(PARTIAL_RULING_TEXT)]})
    checker = AgentChecker(model, _Tools().tools, _settings())

    with pytest.raises(CheckFailed) as raised:
        asyncio.run(checker.check(_statement()))

    assert raised.value.kind == MALFORMED_RULING


def test_a_malformed_ruling_reaches_the_output_payload_under_its_own_kind() -> None:
    """The kind survives the orchestrator, which is the only place it is read."""
    model = _Model({CLAIM: [_ai("verdict: probably true")]})
    checker = AgentChecker(model, _Tools().tools, _settings())
    payload = InputPayload.model_validate(
        {"statements": [wire_statement(statement=CLAIM)]}
    )

    result = asyncio.run(
        run_check(
            payload,
            checker,
            RunSettings(model="google/gemma-4-31b-it", statement_timeout_seconds=5.0),
            lambda: STARTED_AT,
        )
    )

    entry = result.statements[0]
    assert entry.ruling is None
    assert entry.error is not None
    assert entry.error.kind == MALFORMED_RULING


@pytest.mark.parametrize(
    "failure",
    [
        McpCallError("the MCP server at https://example.test returned 422"),
        AuthenticationFailed("the Bright Data MCP server rejected the credential"),
    ],
)
def test_a_tool_failure_travels_out_of_the_check_unchanged(
    failure: Exception,
) -> None:
    """The tool layer decided what a failure means; the agent does not overrule it."""
    model = _Model({CLAIM: [_ai(tool_calls=[_search()])]})
    checker = AgentChecker(
        model, _Tools(search=failure).tools, _settings(tool_call_budget=3)
    )

    with pytest.raises(type(failure)):
        asyncio.run(checker.check(_statement()))


def test_every_tool_call_is_answered_before_the_next_turn_is_sent() -> None:
    """A model that keeps calling tools past its budget still leaves a sendable turn.

    This is the pathological path: the budget runs out, the ruling turn asks for
    another tool anyway and so carries no ruling, and the retry has to send that
    conversation back. It can only do that if every call was answered in writing.
    """
    model = _Model(
        {
            CLAIM: [
                _ai(tool_calls=[_search()]),
                _ai(tool_calls=[_search(query="steam", call_id="c2")]),
                _ai(_ruling_text()),
            ]
        }
    )
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings(tool_call_budget=1))

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert len(tools.named(SEARCH_TOOL_NAME)) == 1
    assert [_unanswered(turn) for turn in model.turns_about(CLAIM)] == [[], [], []]


def test_a_searching_request_carries_the_tools_and_no_ruling_schema() -> None:
    """The regression guard for the defect that made the agent rule from memory.

    A live run proved the two mutually exclusive on one request: the same messages and
    the same tools, sent with a strict `json_schema` `response_format`, came back with
    no tool call and a finished ruling, and sent without one came back asking to
    search. So a searching request must reach the gateway carrying tools and no
    schema, and the answer to it must drive a real tool call.
    """
    gateway = _Gateway(
        [_wire_tool_call(SEARCH_TOOL_NAME, query="boiling point of water")],
        "I have the evidence I need.",
        _ruling_text(),
    )
    checker = AgentChecker(gateway.model(), _real_tools(), _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.searches == 1
    searching, second, ruling = gateway.sent
    assert "response_format" not in searching
    assert "response_format" not in second
    assert ruling["response_format"]["json_schema"]["name"] == "Ruling"
    offered = [one["function"]["name"] for one in searching["tools"]]
    assert offered == [SEARCH_TOOL_NAME, PAGE_TOOL_NAME]


def test_a_searching_request_offers_each_tool_the_one_argument_it_keys_on() -> None:
    """The run cache keys a search on its query, so a further argument is unkeyed.

    `instrument` narrows each tool before the binding sees it, and what the wire
    carries is the only thing the gateway ever reads.
    """
    gateway = _Gateway(_ruling_text())
    checker = AgentChecker(gateway.model(), _real_tools(), _settings())

    asyncio.run(checker.check(_statement()))

    search = gateway.tool_named(SEARCH_TOOL_NAME)
    page = gateway.tool_named(PAGE_TOOL_NAME)
    assert search["parameters"]["required"] == ["query"]
    assert page["parameters"]["required"] == ["url"]
    assert list(search["parameters"]["properties"]) == ["query"]


def test_the_ruling_request_carries_strict_tools_beside_the_schema() -> None:
    """What the wire carries is the only thing the gateway ever sees.

    Two facts about the pinned client meet on this request. A `response_format` in the
    payload routes it through the OpenAI SDK's parsing path, which refuses to send any
    tool that is not strict. And strictness rewrites `required` to list every
    property, which would oblige the model to supply the arguments the run cache
    cannot key on. Both hold together only because `instrument` already narrowed each
    tool to one argument, so the rewrite has nothing left to add.

    The tools ride along because the conversation names their calls, and
    `tool_choice` is what forbids another.
    """
    gateway = _Gateway(_ruling_text())
    checker = AgentChecker(gateway.model(), _real_tools(), _settings())

    asyncio.run(checker.check(_statement()))

    search = gateway.tool_named(SEARCH_TOOL_NAME, request=-1)
    page = gateway.tool_named(PAGE_TOOL_NAME, request=-1)
    assert search["strict"] is True
    assert page["strict"] is True
    assert search["parameters"]["required"] == ["query"]
    assert list(search["parameters"]["properties"]) == ["query"]
    assert gateway.sent[-1]["tool_choice"] == "none"
    assert gateway.sent[-1]["response_format"]["json_schema"]["name"] == "Ruling"


def test_a_malformed_answer_from_the_real_client_reaches_the_retry() -> None:
    """The SDK hands the malformed text over rather than raising on it.

    Bound the `Ruling` class, the SDK validates the body itself and a `ValidationError`
    leaves `ainvoke` before this package sees the answer. Bound a schema document, the
    answer arrives as a message and the correction turn happens.
    """
    gateway = _Gateway("I have the evidence I need.", MALFORMED_ANSWER, _ruling_text())
    checker = AgentChecker(gateway.model(), _real_tools(), _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert len(gateway.sent) == 3
    correction = str(gateway.sent[2]["messages"][-2]["content"])
    assert "not a valid ruling" in correction
    assert MALFORMED_ANSWER in correction


def test_a_second_malformed_answer_from_the_real_client_fails_by_its_own_kind() -> None:
    """The kind the brief calls likeliest is the one that has to be reachable."""
    gateway = _Gateway(MALFORMED_ANSWER)
    checker = AgentChecker(gateway.model(), _real_tools(), _settings())

    with pytest.raises(CheckFailed) as raised:
        asyncio.run(checker.check(_statement()))

    assert raised.value.kind == MALFORMED_RULING
    assert MALFORMED_ANSWER not in raised.value.message


def test_a_whole_check_runs_over_the_real_client_from_tool_call_to_ruling() -> None:
    """The one path a stand-in model cannot reach: a strict tool call, on the wire.

    A strict tool is a tool whose call arguments the OpenAI SDK parses on the way
    back, so a tool this package narrowed is one the SDK is asked to read. The search
    runs, its text goes back as a tool message, and the ruling follows on the second
    turn with both turns' tokens counted.
    """
    gateway = _Gateway(
        [_wire_tool_call(SEARCH_TOOL_NAME, query="boiling point of water")],
        "I have the evidence I need.",
        _ruling_text(),
        prompt=100,
        completion=10,
    )
    checker = AgentChecker(gateway.model(), _real_tools(), _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert outcome.searches == 1
    assert outcome.prompt_tokens == 300
    assert outcome.completion_tokens == 30
    answered = gateway.sent[1]["messages"][-2]
    assert answered["role"] == "tool"
    assert answered["content"] == "one result"


def test_a_well_formed_answer_from_the_real_client_reports_what_it_billed() -> None:
    """The counts come off the gateway's own usage object, through the real client.

    Two requests are billed even where nothing is searched: the turn that decides no
    search is needed, and the ruling turn that follows it.
    """
    gateway = _Gateway(_ruling_text(), prompt=317, completion=64)
    checker = AgentChecker(gateway.model(), _real_tools(), _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert len(gateway.sent) == 2
    assert outcome.prompt_tokens == 634
    assert outcome.completion_tokens == 128
    assert outcome.searches == 0


@pytest.mark.parametrize("status", [401, 403])
def test_a_model_request_the_gateway_refuses_ends_the_run(status: int) -> None:
    """A rejected OpenRouter key fails every statement alike, so it ends the run.

    Left to the SDK, the refusal arrives as its own error class, becomes that one
    statement's `check_failed`, and every statement after it repeats the same rejected
    request while the command still exits with a payload and a zero.
    """
    refusing = _Refusing(status)
    checker = AgentChecker(refusing.model(), _real_tools(), _settings())

    with pytest.raises(AuthenticationFailed) as raised:
        asyncio.run(checker.check(_statement()))

    assert "OpenRouter" in str(raised.value)
    assert str(status) in str(raised.value)
    assert len(refusing.sent) == 1


def test_a_model_request_that_fails_some_other_way_is_left_as_it_is() -> None:
    """Only a refused credential ends the run. Everything else fails one statement.

    What the client raises for a 400 is a class of its own, from a package this one
    does not depend on, so the assertion is that the failure is not the run-ending one
    rather than that it is any particular class.
    """
    refusing = _Refusing(400)
    checker = AgentChecker(refusing.model(), _real_tools(), _settings())

    with pytest.raises(Exception) as raised:
        asyncio.run(checker.check(_statement()))

    assert not isinstance(raised.value, AuthenticationFailed)


def test_a_statements_search_count_is_its_own_while_another_runs_beside_it() -> None:
    """One `AgentChecker` serves many statements, and neither sees the other's calls."""
    model = _Model(
        {
            CLAIM: [_ai(tool_calls=[_search()]), _done(), _ai(_ruling_text())],
            OTHER_CLAIM: [
                _ai(tool_calls=[_search(query="mercury", call_id="m1")]),
                _ai(tool_calls=[_search(query="freezing", call_id="m2")]),
                _ai(tool_calls=[_read(call_id="m3")]),
                _done(),
                _ai(_ruling_text(verdict="refuted")),
            ],
        }
    )
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings())

    async def both() -> tuple[object, object]:
        return await asyncio.gather(
            checker.check(_statement()),
            checker.check(_statement(claim=OTHER_CLAIM, identifier="s2")),
        )

    water, mercury = asyncio.run(both())

    assert model.claim_order()[:2] == [CLAIM, OTHER_CLAIM]
    assert water.searches == 1
    assert mercury.searches == 2
    assert len(tools.named(SEARCH_TOOL_NAME)) == 3


def test_a_check_that_needs_no_tool_rules_without_spending_one() -> None:
    """A model that asks for no tool is not made to search for form's sake."""
    model = _Model({CLAIM: [_done(), _ai(_ruling_text(verdict="mixed"))]})
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "mixed"
    assert outcome.searches == 0
    assert tools.calls == []


def test_a_tool_call_on_the_ruling_turn_is_refused_rather_than_made() -> None:
    """The searching is over, and its budget is not the ruling turn's to spend.

    The binding leaves the model no way to ask for a tool on this turn, so a call
    here is a gateway forwarding one anyway. Answering it in writing is what keeps
    the conversation sendable: the retry could not go back with an assistant message
    whose call no message answers.
    """
    model = _Model({CLAIM: [_done(), _ai(tool_calls=[_search()]), _ai(_ruling_text())]})
    tools = _Tools()
    checker = AgentChecker(model, tools.tools, _settings())

    outcome = asyncio.run(checker.check(_statement()))

    assert outcome.ruling.verdict == "supported"
    assert tools.calls == []
    assert "searching is over" in _answer(model.turns_about(CLAIM)[-1])
    assert [_unanswered(turn) for turn in model.turns_about(CLAIM)] == [[], [], []]
