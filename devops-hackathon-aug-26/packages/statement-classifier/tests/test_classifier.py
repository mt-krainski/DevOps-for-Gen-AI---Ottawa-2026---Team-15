import asyncio

import pytest
from conftest import FakeModel

from statement_classifier.classifier import ClassificationFailure, classify_one
from statement_classifier.errors import ErrorCode
from statement_classifier.models import Classification


async def test_classifies_a_fact_example(fact_statement):
    model = FakeModel([Classification(**{"class": "fact", "confidence": 0.9})])

    result = await classify_one(fact_statement, model)

    assert result.class_ == "fact"
    assert 0.0 <= result.confidence <= 1.0


async def test_classifies_an_opinion_example(opinion_statement):
    model = FakeModel([Classification(**{"class": "opinion", "confidence": 0.85})])

    result = await classify_one(opinion_statement, model)

    assert result.class_ == "opinion"
    assert 0.0 <= result.confidence <= 1.0


async def test_prompt_includes_surrounding_context_and_statement(fact_statement):
    model = FakeModel([Classification(**{"class": "fact", "confidence": 0.9})])

    await classify_one(fact_statement, model)

    assert len(model.calls) == 1
    assert fact_statement.surroundingContext in model.calls[0]
    assert fact_statement.statement in model.calls[0]


async def test_llm_error_after_retries_exhausted_raises_llm_error(fact_statement):
    model = FakeModel([RuntimeError("boom")] * 3)

    with pytest.raises(ClassificationFailure) as exc_info:
        await classify_one(fact_statement, model)

    assert exc_info.value.code == ErrorCode.LLM_ERROR
    assert len(model.calls) == 3


async def test_timeout_after_retries_exhausted_raises_llm_timeout(fact_statement):
    async def hang(_prompt: str) -> Classification:
        await asyncio.sleep(10)
        raise AssertionError("should not reach here")

    model = FakeModel([])
    model.ainvoke = hang

    from statement_classifier import classifier as classifier_module

    original_timeout = classifier_module.CALL_TIMEOUT_SECONDS
    classifier_module.CALL_TIMEOUT_SECONDS = 0.01
    try:
        with pytest.raises(ClassificationFailure) as exc_info:
            await classify_one(fact_statement, model)
    finally:
        classifier_module.CALL_TIMEOUT_SECONDS = original_timeout

    assert exc_info.value.code == ErrorCode.LLM_TIMEOUT


async def test_recovers_after_transient_failure(fact_statement):
    model = FakeModel(
        [RuntimeError("transient"), Classification(**{"class": "fact", "confidence": 0.7})]
    )

    result = await classify_one(fact_statement, model)

    assert result.class_ == "fact"
    assert len(model.calls) == 2


async def test_unparseable_output_after_retries_raises_parse_error(fact_statement):
    model = FakeModel([{"not": "a classification"}] * 3)

    with pytest.raises(ClassificationFailure) as exc_info:
        await classify_one(fact_statement, model)

    assert exc_info.value.code == ErrorCode.PARSE_ERROR
