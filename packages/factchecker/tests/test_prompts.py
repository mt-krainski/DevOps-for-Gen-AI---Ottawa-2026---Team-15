"""Tests for the three prompts in `factchecker.prompts`.

The wording is the implementer's to choose, and Task 4 tunes it against the evaluation
suite, so these tests pin the elements a prompt must carry rather than the sentences it
carries them in. Each test names one element and asserts the least that shows it is
there.
"""

import pytest

from factchecker.models import IdentifiedStatement
from factchecker.prompts import (
    build_budget_reminder,
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


def test_the_system_prompt_asks_the_search_tool_for_a_query_and_nothing_else() -> None:
    """The run cache keys a search on its query, so a varied argument is a wasted call.

    This is a property of `factchecker.tools`, not a preference. A prompt that invited
    the model to vary `num_results` or `country` would spend calls on results the cache
    had already served.
    """
    prompt = build_system_prompt(BUDGET).lower()

    assert "query and nothing else" in prompt


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


def test_the_budget_reminder_says_to_rule_now_when_the_budget_is_spent() -> None:
    """The stop is not a failure: the agent rules on what it holds."""
    reminder = build_budget_reminder(10, 10)

    assert "10 of 10" in reminder
    assert "none left" in reminder
    assert "rule now" in reminder.lower()
