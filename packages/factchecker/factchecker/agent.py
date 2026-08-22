"""The searching agent: one claim, a budget of tool calls, and a ruling with sources.

The loop is written here rather than taken from `langgraph`'s prebuilt agent. Neither
of the two things this package needs from a loop — a budget reminder on every turn, and
a stop at the budget that still ends in a ruling — is something that agent does, and
the arithmetic that decides both is arithmetic a test has to be able to drive without a
live model.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from factchecker.checker import CheckOutcome
from factchecker.config import Settings
from factchecker.errors import CheckFailed
from factchecker.models import IdentifiedStatement, Ruling
from factchecker.prompts import (
    build_budget_reminder,
    build_statement_prompt,
    build_system_prompt,
)
from factchecker.tools import SEARCH_TOOL_NAME

MALFORMED_RULING = "malformed_ruling"

_BUDGET_SPENT = (
    "This call was not made: the tool-call budget for this claim is spent. "
    "Rule now on the evidence you already hold."
)


@dataclass
class _Run:
    """One statement's conversation, and what that statement has spent so far.

    Every field here is one statement's own. That is what lets a single
    `AgentChecker` serve many statements at once: the checker holds the model, the
    tools and the settings, and holds nothing that changes.
    """

    conversation: list[BaseMessage]
    used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AgentChecker:
    """A `StatementChecker` that searches the web and rules with cited sources.

    Every collaborator is passed in and none is built here, so a test drives the whole
    loop against a scripted model and tools that open no connection.
    """

    def __init__(
        self, model: BaseChatModel, tools: Sequence[BaseTool], settings: Settings
    ) -> None:
        """Bind the model to the tools and to the ruling schema, once for the run.

        The ruling is asked for as structured output, and is validated again when it
        arrives. That is not belt and braces: the free model this package offers as an
        override accepts `response_format` without enforcing it, so a package that
        trusted the gateway would break the moment somebody switched models to save
        money.

        Args:
            model: The chat model, with nothing bound to it yet.
            tools: The instrumented Bright Data tools. They already carry the run's
                cache, its retry policy and the page ceiling, and they hold no
                per-statement state, so one set serves every statement of a run.
            settings: The run's settings. This class reads the tool-call budget.
        """
        # `strict=False` is what keeps the tools' own schemas intact. Given a
        # `response_format` and no `strict`, `langchain-openai` 1.6.0 converts every
        # tool with `strict=True`, and that conversion rewrites `required` to list
        # every property. Bright Data's search tool would then oblige the model to
        # supply `country` and the rest on every call, and the run cache keys a search
        # on its query alone.
        self._model = model.bind_tools(tools, response_format=Ruling, strict=False)
        self._tools = {tool.name: tool for tool in tools}
        self._settings = settings

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Check one statement, spending at most the budget's worth of tool calls.

        Args:
            statement: The statement to check, with its identifier assigned.

        Returns:
            The ruling, what every turn of this check cost, and the searches this
            check asked for.

        Raises:
            AuthenticationFailed: A credential was rejected. It travels on from the
                tool layer untouched, because it ends the run and not this statement.
            CheckFailed: The ruling did not validate, and nor did the one the model
                wrote after being shown why. The kind is `malformed_ruling`.
        """
        run = _Run(
            conversation=[
                SystemMessage(build_system_prompt(self._settings.tool_call_budget)),
                HumanMessage(build_statement_prompt(statement)),
            ]
        )
        answer = await self._converse(run)
        ruling = await self._ruled(run, answer)
        return CheckOutcome(
            ruling=ruling,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            searches=_searches(run.conversation),
        )

    async def _converse(self, run: _Run) -> AIMessage:
        """Take turns with the model until it answers without asking for a tool.

        The budget is read between turns, so the turn that spends the last call is
        followed by one more: its reminder says the budget is gone, and the model
        rules on what it holds. Reaching the budget is not a failure.

        That last turn is passed to `_spend` as well. It has nothing left to spend, so
        every call it asks for is refused in writing. A refusal left unwritten would
        leave an assistant message whose tool calls no message answers, and the retry
        in `_ruled` could not send that conversation back.
        """
        while run.used < self._settings.tool_call_budget:
            answer = await self._turn(run)
            if not answer.tool_calls:
                return answer
            await self._spend(run, answer)
        last = await self._turn(run)
        await self._spend(run, last)
        return last

    async def _turn(self, run: _Run) -> AIMessage:
        """Ask the model once, with the budget reminder appended to what it sees.

        The reminder rides on the turn rather than joining the conversation, so the
        history holds one claim and its evidence instead of a stack of stale counts.
        """
        reminder = HumanMessage(
            build_budget_reminder(run.used, self._settings.tool_call_budget)
        )
        answer = await self._model.ainvoke([*run.conversation, reminder])
        prompt_tokens, completion_tokens = _tokens(answer)
        run.prompt_tokens += prompt_tokens
        run.completion_tokens += completion_tokens
        run.conversation.append(answer)
        return answer

    async def _spend(self, run: _Run, answer: AIMessage) -> None:
        """Make the calls the model asked for, as far as the budget still allows.

        A call past the budget is answered rather than dropped. Every tool call an
        assistant message carries needs a reply of its own, and one left unanswered
        is a conversation the next turn cannot be sent.
        """
        for call in answer.tool_calls:
            if run.used < self._settings.tool_call_budget:
                run.used += 1
                run.conversation.append(await self._answered(call))
            else:
                run.conversation.append(
                    ToolMessage(content=_BUDGET_SPENT, tool_call_id=call["id"])
                )

    async def _answered(self, call: ToolCall) -> ToolMessage:
        """Run one tool call, or say why it could not be run.

        A failure the tool layer raises travels on untouched. `AuthenticationFailed`
        ends the whole run and `McpCallError` ends this statement, which is what the
        tool layer already decided a permanent failure means.
        """
        tool = self._tools.get(call["name"])
        if tool is None:
            return ToolMessage(
                content=(
                    f"This call was not made: there is no tool named {call['name']}. "
                    f"Your tools are {' and '.join(self._tools)}."
                ),
                tool_call_id=call["id"],
            )
        return await tool.ainvoke(call)

    async def _ruled(self, run: _Run, answer: AIMessage) -> Ruling:
        """Read the ruling out of the model's answer, allowing one try at fixing it.

        A ruling that half validates is thrown away whole. A ruling missing its
        references is not a weaker ruling; it is a different claim.
        """
        try:
            return Ruling.model_validate_json(answer.text)
        except ValidationError as rejected:
            run.conversation.append(HumanMessage(_correction(rejected)))
            retried = await self._turn(run)

        try:
            return Ruling.model_validate_json(retried.text)
        except ValidationError:
            raise CheckFailed(
                MALFORMED_RULING,
                "the model's ruling did not validate, and nor did the one it wrote "
                "after being shown why",
            ) from None


def _correction(rejected: ValidationError) -> str:
    """Tell the model what its ruling got wrong, in the validator's own words.

    Quoting the validator is safe here and useful. Its text repeats the model's own
    output back at it, which is what the model needs to see, and it names no endpoint
    and no credential.

    Args:
        rejected: What validating the answer as a `Ruling` raised.

    Returns:
        The turn that carries the rejection back to the model.
    """
    return (
        "That is not a valid ruling. Validation rejected it like this:\n\n"
        f"{rejected}\n\n"
        "Answer again with the ruling alone, as one JSON object holding `verdict`, "
        "`confidence`, `justification` and `references`. Every field is required. "
        "Do not call a tool, and write nothing outside the object."
    )


def _tokens(answer: BaseMessage) -> tuple[int, int]:
    """Read the prompt and completion tokens one turn cost.

    OpenRouter returns its `usage` object on every response, and `langchain-openai`
    1.6.0 puts it on the message as `response_metadata["token_usage"]`. A response
    that carries none costs nothing here rather than failing the check: a lost count
    is worth less than a lost ruling.

    Args:
        answer: One turn's answer from the model.

    Returns:
        The prompt tokens and the completion tokens, in that order.
    """
    usage = answer.response_metadata.get("token_usage") or {}
    return usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0


def _searches(conversation: Sequence[BaseMessage]) -> int:
    """Count the searches this statement's own run asked for.

    Counted from this run's messages rather than beside the tools. The tools are
    shared by every statement running at that moment, so a counter living there would
    report the whole run's traffic to each statement, and would report a different
    figure depending on who else was searching.

    A cache hit counts, and so does a call the budget refused, because the number
    reports what the agent asked for.

    Args:
        conversation: This statement's messages, as the run left them.

    Returns:
        How many search calls the run's assistant messages hold.
    """
    return sum(
        1
        for message in conversation
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == SEARCH_TOOL_NAME
    )
