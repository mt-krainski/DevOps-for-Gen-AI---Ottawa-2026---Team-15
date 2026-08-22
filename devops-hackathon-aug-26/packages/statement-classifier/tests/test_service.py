import asyncio

import httpx
import pytest
from conftest import FakeModel

from statement_classifier.errors import ClassifierError, ErrorCode
from statement_classifier.models import Classification, ClassifierInput, Statement
from statement_classifier.service import classify_statements, classify_statements_sync


def make_input(*statements: Statement) -> ClassifierInput:
    return ClassifierInput(statements=list(statements))


async def test_mixed_batch_classifies_facts_and_opinions(fact_statement, opinion_statement):
    model = FakeModel(
        [
            Classification(**{"class": "fact", "confidence": 0.9}),
            Classification(**{"class": "opinion", "confidence": 0.8}),
        ]
    )

    output = await classify_statements(
        make_input(fact_statement, opinion_statement), model=model
    )

    assert len(output.statements) == 2
    assert output.statements[0].classification.class_ == "fact"
    assert output.statements[0].error is None
    assert output.statements[1].classification.class_ == "opinion"
    assert output.statements[1].error is None


async def test_empty_batch_returns_empty_result():
    model = FakeModel([])

    output = await classify_statements(make_input(), model=model)

    assert output.statements == []


async def test_one_failing_statement_does_not_affect_siblings(
    fact_statement, opinion_statement
):
    # First statement fails on all 3 attempts; second succeeds first try.
    model = FakeModel(
        [
            RuntimeError("boom"),
            RuntimeError("boom"),
            RuntimeError("boom"),
            Classification(**{"class": "opinion", "confidence": 0.8}),
        ]
    )

    output = await classify_statements(
        make_input(fact_statement, opinion_statement), model=model, concurrency=1
    )

    failed, succeeded = output.statements
    assert failed.classification is None
    assert failed.error.code == ErrorCode.LLM_ERROR
    assert succeeded.classification.class_ == "opinion"
    assert succeeded.error is None


async def test_malformed_input_raises_invalid_input_before_llm_calls():
    model = FakeModel([])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_statements(
            {"statements": [{"surroundingContext": "ctx"}]}, model=model
        )

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert model.calls == []


async def test_empty_batch_returns_empty_result_without_requiring_a_model(
    monkeypatch,
):
    # No `model=` override and no API key: proves the empty-batch short-circuit
    # happens before config/model construction, not just when a model is given.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    output = await classify_statements(make_input())

    assert output.statements == []


async def test_invalid_auth_error_aborts_whole_batch(fact_statement, opinion_statement):
    import openai

    auth_error = openai.AuthenticationError(
        message="invalid api key",
        response=httpx.Response(
            status_code=401, request=httpx.Request("POST", "https://example.com")
        ),
        body=None,
    )
    model = FakeModel([auth_error, Classification(**{"class": "fact", "confidence": 0.9})])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_statements(
            make_input(fact_statement, opinion_statement), model=model, concurrency=1
        )

    assert exc_info.value.code == ErrorCode.AUTH_ERROR


async def test_concurrency_below_one_raises_invalid_input(fact_statement):
    model = FakeModel([])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_statements(make_input(fact_statement), model=model, concurrency=0)

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert model.calls == []


async def test_missing_api_key_raises_before_llm_calls(monkeypatch, fact_statement):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ClassifierError) as exc_info:
        # No `model=` override: forces config loading, which should fail fast.
        await classify_statements(make_input(fact_statement))

    assert exc_info.value.code == ErrorCode.MISSING_API_KEY


def test_sync_wrapper_returns_same_result_as_async(fact_statement):
    model = FakeModel([Classification(**{"class": "fact", "confidence": 0.9})])

    output = classify_statements_sync(make_input(fact_statement), model=model)

    assert output.statements[0].classification.class_ == "fact"


async def test_concurrency_is_bounded_by_semaphore():
    concurrency_limit = 2
    in_flight = 0
    max_observed = 0

    class TrackingModel:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, prompt: str) -> Classification:
            nonlocal in_flight, max_observed
            self.calls.append(prompt)
            in_flight += 1
            max_observed = max(max_observed, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return Classification(**{"class": "fact", "confidence": 0.9})

    statements = [
        Statement(surroundingContext="ctx", statement=f"statement {i}") for i in range(6)
    ]

    output = await classify_statements(
        make_input(*statements), model=TrackingModel(), concurrency=concurrency_limit
    )

    assert len(output.statements) == 6
    assert max_observed <= concurrency_limit
