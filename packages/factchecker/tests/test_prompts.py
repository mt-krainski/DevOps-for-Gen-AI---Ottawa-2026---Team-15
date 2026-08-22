"""Tests for the three prompts in `factchecker.prompts`.

The wording is tuned against the evaluation suite in `eval/`, so these tests pin the
elements a prompt must carry rather than the sentences it carries them in. Each test
names one element and asserts the least that shows it is there.
"""

import pytest

from factchecker.models import IdentifiedStatement
from factchecker.prompts import (
    build_budget_reminder,
    build_ruling_request,
    build_statement_prompt,
    build_system_prompt,
)
from factchecker.tools import PAGE_TOOL_NAME, SEARCH_TOOL_NAME

BUDGET = 10

CONTEXT = (
    "Ada Lovelace worked with Charles Babbage on the Analytical Engine. "
    "She wrote the first algorithm intended for a machine. "
    "Her notes were published in 1843."
)
CLAIM = "She wrote the first algorithm intended for a machine"


def _statement() -> IdentifiedStatement:
    """One identified statement whose claim leans on its context to be searchable."""
    return IdentifiedStatement(
        id="s1",
        surrounding_context=CONTEXT,
        statement=CLAIM,
        classification={"class": "fact", "confidence": 0.8},
    )


def test_the_system_prompt_sets_the_task_as_one_claim_against_web_evidence() -> None:
    """The reader is told what it is doing before it is told how to do it."""
    prompt = build_system_prompt(BUDGET).lower()

    assert "one factual claim" in prompt
    assert "web" in prompt
    assert "evidence" in prompt


@pytest.mark.parametrize("verdict", ["supported", "refuted", "mixed", "unverifiable"])
def test_the_system_prompt_names_each_verdict(verdict: str) -> None:
    """A verdict the prompt never names is a verdict the model never returns."""
    prompt = build_system_prompt(BUDGET)

    assert verdict in prompt


def test_the_system_prompt_presents_unverifiable_as_a_legitimate_outcome() -> None:
    """A model that reads `unverifiable` as failure guesses instead of reporting."""
    prompt = build_system_prompt(BUDGET).lower()

    assert "not a failure" in prompt


def test_the_system_prompt_says_confidence_measures_the_verdict() -> None:
    """Confidence in the verdict and belief in the claim are different numbers."""
    prompt = build_system_prompt(BUDGET).lower()

    assert "confidence" in prompt
    assert "not how likely the claim is to be true" in prompt


def test_the_system_prompt_gives_the_citation_format() -> None:
    """Each reference carries three fields, and the justification points at them."""
    prompt = build_system_prompt(BUDGET)

    assert "`id`" in prompt
    assert "`source`" in prompt
    assert "`excerpt`" in prompt
    assert "[1]" in prompt


@pytest.mark.parametrize("budget", [1, 4, 25])
def test_the_system_prompt_states_the_budget_it_was_given(budget: int) -> None:
    """The budget is configuration, so the prompt reports it rather than assuming it."""
    prompt = build_system_prompt(budget)

    assert str(budget) in prompt


def test_the_system_prompt_asks_for_the_budget_to_be_planned_across_both_tools() -> (
    None
):
    """Searching and reading spend one budget, so neither may exhaust it alone."""
    prompt = build_system_prompt(BUDGET)

    assert SEARCH_TOOL_NAME in prompt
    assert PAGE_TOOL_NAME in prompt
    assert "plan" in prompt.lower()


def test_the_system_prompt_says_the_ruling_is_asked_for_in_a_turn_of_its_own() -> None:
    """The mechanism the loop uses is the mechanism the prompt describes.

    The ruling is a request of its own, made after the searching turns. A prompt that
    asked for a ruling in the same breath as a search would describe a turn the agent
    never takes.
    """
    prompt = build_system_prompt(BUDGET).lower()

    assert "turn of its own" in prompt


def test_the_statement_prompt_carries_the_claim_and_its_context() -> None:
    """Both reach the model, because the claim alone is often unsearchable."""
    prompt = build_statement_prompt(_statement())

    assert CLAIM in prompt
    assert CONTEXT in prompt


def test_the_statement_prompt_asks_for_references_to_be_resolved_first() -> None:
    """`She` is searchable only once the context has said who she is."""
    prompt = build_statement_prompt(_statement()).lower()

    assert "resolve" in prompt
    assert "pronoun" in prompt


def test_the_budget_reminder_reports_the_whole_budget_at_the_start() -> None:
    """Nothing is spent yet, so everything is left."""
    reminder = build_budget_reminder(0, 10)

    assert "0 of 10" in reminder
    assert "10 left" in reminder


def test_the_budget_reminder_reports_what_is_left_in_the_middle() -> None:
    """An agent that knows what it has left paces itself."""
    reminder = build_budget_reminder(4, 10)

    assert "4 of 10" in reminder
    assert "6 left" in reminder


def test_the_budget_reminder_reports_the_last_call_as_one_still_to_spend() -> None:
    """A searching turn happens only while a call is left, so one is always left."""
    reminder = build_budget_reminder(9, 10)

    assert "9 of 10" in reminder
    assert "1 left" in reminder


def test_the_ruling_request_asks_for_the_ruling_alone_as_one_object() -> None:
    """The ruling turn carries the shape of the answer, not the state of the budget."""
    request = build_ruling_request()

    assert "rule now" in request.lower()
    for field in ("`verdict`", "`confidence`", "`justification`", "`references`"):
        assert field in request


def test_the_ruling_request_asks_for_the_evidence_the_conversation_holds() -> None:
    """The ruling turn is where invented sources get in, so it says where to cite from.

    This is the turn the agent rules on, and the model that reaches it holds both what
    it read and what it already believed. The request names which of the two counts.
    """
    request = build_ruling_request().lower()

    assert "this conversation holds" in request
    assert "cite only pages you read here" in request


def test_the_ruling_request_says_an_unsettled_claim_is_unverifiable() -> None:
    """The stop is not a failure: a ruling the model guessed at is worse."""
    request = build_ruling_request()

    assert "`unverifiable`" in request
