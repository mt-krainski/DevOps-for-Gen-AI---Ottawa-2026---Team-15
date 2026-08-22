"""Tests for the single-statement call in `statement_classifier.classifier`."""

import asyncio

import pytest

from statement_classifier import classifier as classifier_module
from statement_classifier.classifier import classify_one
from statement_classifier.errors import ClassificationFailure, ErrorCode
from statement_classifier.models import Classification, Statement
from tests.conftest import FakeModel


async def test_classifies_a_fact_example(fact_statement: Statement) -> None:
    """A checkable claim comes back labelled `fact`, with a confidence in range."""
    model = FakeModel([Classification(**{"class": "fact", "confidence": 0.9})])

    result = await classify_one(fact_statement, model)

    assert result.class_ == "fact"
    assert 0.0 <= result.confidence <= 1.0


async def test_classifies_an_opinion_example(opinion_statement: Statement) -> None:
    """A subjective statement comes back labelled `opinion`."""
    model = FakeModel([Classification(**{"class": "opinion", "confidence": 0.85})])

    result = await classify_one(opinion_statement, model)

    assert result.class_ == "opinion"
    assert 0.0 <= result.confidence <= 1.0


async def test_prompt_includes_surrounding_context_and_statement(
    fact_statement: Statement,
) -> None:
    """The context reaches the model, so an ambiguous reference can be resolved."""
    model = FakeModel([Classification(**{"class": "fact", "confidence": 0.9})])

    await classify_one(fact_statement, model)

    assert len(model.calls) == 1
    assert fact_statement.surrounding_context in model.calls[0]
    assert fact_statement.statement in model.calls[0]


async def test_llm_error_after_retries_exhausted_raises_llm_error(
    fact_statement: Statement,
) -> None:
    """Every attempt failing surfaces as one `LLM_ERROR`, not three."""
    model = FakeModel([RuntimeError("boom")] * 3)

    with pytest.raises(ClassificationFailure) as exc_info:
        await classify_one(fact_statement, model)

    assert exc_info.value.code == ErrorCode.LLM_ERROR
    assert len(model.calls) == 3


async def test_timeout_after_retries_exhausted_raises_llm_timeout(
    fact_statement: Statement,
) -> None:
    """A model that never answers is cut off by the call timeout."""

    async def hang(_prompt: str) -> Classification:
        await asyncio.sleep(10)
        raise AssertionError("should not reach here")

    model = FakeModel([])
    model.ainvoke = hang

    original_timeout = classifier_module.CALL_TIMEOUT_SECONDS
    classifier_module.CALL_TIMEOUT_SECONDS = 0.01
    try:
        with pytest.raises(ClassificationFailure) as exc_info:
            await classify_one(fact_statement, model)
    finally:
        classifier_module.CALL_TIMEOUT_SECONDS = original_timeout

    assert exc_info.value.code == ErrorCode.LLM_TIMEOUT


async def test_recovers_after_transient_failure(fact_statement: Statement) -> None:
    """One failed attempt is retried, and the retry's answer is returned."""
    model = FakeModel(
        [
            RuntimeError("transient"),
            Classification(**{"class": "fact", "confidence": 0.7}),
        ]
    )

    result = await classify_one(fact_statement, model)

    assert result.class_ == "fact"
    assert len(model.calls) == 2


async def test_unparseable_output_after_retries_raises_parse_error(
    fact_statement: Statement,
) -> None:
    """Output that is not a `Classification` is a parse failure, not an LLM one."""
    model = FakeModel([{"not": "a classification"}] * 3)

    with pytest.raises(ClassificationFailure) as exc_info:
        await classify_one(fact_statement, model)

    assert exc_info.value.code == ErrorCode.PARSE_ERROR
