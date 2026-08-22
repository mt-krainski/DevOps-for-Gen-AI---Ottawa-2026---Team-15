"""Tests for the batch call in `statement_classifier.service`."""

import asyncio
import json

import httpx
import openai
import pytest

from statement_classifier.errors import ClassifierError, ErrorCode
from statement_classifier.models import Classification, ClassifierInput, Statement
from statement_classifier.segmenter import _Segments
from statement_classifier.service import (
    classify_paragraph,
    classify_paragraph_sync,
    classify_statements,
    classify_statements_sync,
)
from tests.conftest import FakeModel

PARAGRAPH = (
    "Carney confirmed he was adding tariffs that would add costs for Canadians, "
    "but insisted they were necessary to retaliate against Trump's levies"
)
FIRST_CLAUSE = (
    "Carney confirmed he was adding tariffs that would add costs for Canadians"
)
SECOND_CLAUSE = "but insisted they were necessary to retaliate against Trump's levies"


def make_input(*statements: Statement) -> ClassifierInput:
    """Wrap the statements as the batch the service reads."""
    return ClassifierInput(statements=list(statements))


async def test_mixed_batch_classifies_facts_and_opinions(
    fact_statement: Statement, opinion_statement: Statement
) -> None:
    """Each statement gets its own label, in the order it was submitted."""
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


async def test_camel_case_payload_round_trips_on_the_wire() -> None:
    """The batch reads the pipeline's camelCase keys and writes them back."""
    model = FakeModel([Classification(**{"class": "fact", "confidence": 0.9})])

    output = await classify_statements(
        {"statements": [{"surroundingContext": "ctx", "statement": "s"}]}, model=model
    )

    assert output.statements[0].surrounding_context == "ctx"
    assert json.loads(output.model_dump_json())["statements"][0] == {
        "surroundingContext": "ctx",
        "statement": "s",
        "classification": {"class": "fact", "confidence": 0.9},
        "error": None,
    }


async def test_empty_batch_returns_empty_result() -> None:
    """Nothing in, nothing out, and no call made."""
    model = FakeModel([])

    output = await classify_statements(make_input(), model=model)

    assert output.statements == []


async def test_one_failing_statement_does_not_affect_siblings(
    fact_statement: Statement, opinion_statement: Statement
) -> None:
    """A failure is isolated onto its own statement's `error` field."""
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


async def test_malformed_input_raises_invalid_input_before_llm_calls() -> None:
    """A statement with no `statement` field is rejected before anything is called."""
    model = FakeModel([])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_statements(
            {"statements": [{"surroundingContext": "ctx"}]}, model=model
        )

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert model.calls == []


async def test_empty_batch_returns_empty_result_without_requiring_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty-batch short-circuit happens before config and model construction."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    output = await classify_statements(make_input())

    assert output.statements == []


async def test_invalid_auth_error_aborts_whole_batch(
    fact_statement: Statement, opinion_statement: Statement
) -> None:
    """A rejected credential ends the batch rather than marking one statement."""
    auth_error = openai.AuthenticationError(
        message="invalid api key",
        response=httpx.Response(
            status_code=401, request=httpx.Request("POST", "https://example.com")
        ),
        body=None,
    )
    model = FakeModel(
        [auth_error, Classification(**{"class": "fact", "confidence": 0.9})]
    )

    with pytest.raises(ClassifierError) as exc_info:
        await classify_statements(
            make_input(fact_statement, opinion_statement), model=model, concurrency=1
        )

    assert exc_info.value.code == ErrorCode.AUTH_ERROR


async def test_concurrency_below_one_raises_invalid_input(
    fact_statement: Statement,
) -> None:
    """A concurrency that admits no calls is rejected as bad input."""
    model = FakeModel([])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_statements(
            make_input(fact_statement), model=model, concurrency=0
        )

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert model.calls == []


async def test_missing_api_key_raises_before_llm_calls(
    monkeypatch: pytest.MonkeyPatch, fact_statement: Statement
) -> None:
    """With no `model=` override, config loading fails fast on the missing key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ClassifierError) as exc_info:
        await classify_statements(make_input(fact_statement))

    assert exc_info.value.code == ErrorCode.MISSING_API_KEY


def test_sync_wrapper_returns_same_result_as_async(fact_statement: Statement) -> None:
    """The sync wrapper answers with what the async call would have answered."""
    model = FakeModel([Classification(**{"class": "fact", "confidence": 0.9})])

    output = classify_statements_sync(make_input(fact_statement), model=model)

    assert output.statements[0].classification.class_ == "fact"


async def test_concurrency_is_bounded_by_semaphore() -> None:
    """No more calls are ever in flight at once than the concurrency allows."""
    concurrency_limit = 2
    in_flight = 0
    max_observed = 0

    class TrackingModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def ainvoke(self, prompt: str) -> Classification:
            nonlocal in_flight, max_observed
            self.calls.append(prompt)
            in_flight += 1
            max_observed = max(max_observed, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return Classification(**{"class": "fact", "confidence": 0.9})

    statements = [
        Statement(surrounding_context="ctx", statement=f"statement {i}")
        for i in range(6)
    ]

    output = await classify_statements(
        make_input(*statements), model=TrackingModel(), concurrency=concurrency_limit
    )

    assert len(output.statements) == 6
    assert max_observed <= concurrency_limit


async def test_paragraph_splits_and_classifies_fact_then_opinion() -> None:
    """The example from the spec: a fact clause, then an opinion clause."""
    segmenter_model = FakeModel([_Segments(statements=[FIRST_CLAUSE, SECOND_CLAUSE])])
    classifier_model = FakeModel(
        [
            Classification(**{"class": "fact", "confidence": 0.95}),
            Classification(**{"class": "opinion", "confidence": 0.95}),
        ]
    )

    output = await classify_paragraph(
        {"paragraph": PARAGRAPH},
        classifier_model=classifier_model,
        segmenter_model=segmenter_model,
    )

    assert [s.statement for s in output.statements] == [FIRST_CLAUSE, SECOND_CLAUSE]
    assert output.statements[0].classification.class_ == "fact"
    assert output.statements[1].classification.class_ == "opinion"


async def test_paragraph_output_omits_surrounding_context() -> None:
    """The wire shape carries `statement`, `classification`, `error` only."""
    segmenter_model = FakeModel([_Segments(statements=[FIRST_CLAUSE])])
    classifier_model = FakeModel(
        [Classification(**{"class": "fact", "confidence": 0.95})]
    )

    output = await classify_paragraph(
        {"paragraph": PARAGRAPH},
        classifier_model=classifier_model,
        segmenter_model=segmenter_model,
    )

    assert json.loads(output.model_dump_json())["statements"][0] == {
        "statement": FIRST_CLAUSE,
        "classification": {"class": "fact", "confidence": 0.95},
        "error": None,
    }


async def test_paragraph_with_no_statements_returns_empty_result() -> None:
    """A paragraph the segmenter finds nothing splittable in makes no calls."""
    segmenter_model = FakeModel([_Segments(statements=[])])
    classifier_model = FakeModel([])

    output = await classify_paragraph(
        {"paragraph": PARAGRAPH},
        classifier_model=classifier_model,
        segmenter_model=segmenter_model,
    )

    assert output.statements == []
    assert classifier_model.calls == []


async def test_segmentation_failure_aborts_before_any_classification_call() -> None:
    """A paragraph that can't be split never reaches the classifier."""
    segmenter_model = FakeModel([RuntimeError("boom")] * 3)
    classifier_model = FakeModel([])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_paragraph(
            {"paragraph": PARAGRAPH},
            classifier_model=classifier_model,
            segmenter_model=segmenter_model,
        )

    assert exc_info.value.code == ErrorCode.SEGMENTATION_ERROR
    assert classifier_model.calls == []


async def test_empty_paragraph_raises_invalid_input_before_any_llm_call() -> None:
    """A whitespace-only paragraph is rejected before segmentation is attempted."""
    segmenter_model = FakeModel([])
    classifier_model = FakeModel([])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_paragraph(
            {"paragraph": "   "},
            classifier_model=classifier_model,
            segmenter_model=segmenter_model,
        )

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert segmenter_model.calls == []
    assert classifier_model.calls == []


async def test_paragraph_one_failing_statement_does_not_affect_siblings() -> None:
    """A classification failure is isolated onto that statement, as in batch mode."""
    segmenter_model = FakeModel([_Segments(statements=[FIRST_CLAUSE, SECOND_CLAUSE])])
    classifier_model = FakeModel(
        [
            RuntimeError("boom"),
            RuntimeError("boom"),
            RuntimeError("boom"),
            Classification(**{"class": "opinion", "confidence": 0.8}),
        ]
    )

    output = await classify_paragraph(
        {"paragraph": PARAGRAPH},
        classifier_model=classifier_model,
        segmenter_model=segmenter_model,
        concurrency=1,
    )

    failed, succeeded = output.statements
    assert failed.classification is None
    assert failed.error.code == ErrorCode.LLM_ERROR
    assert succeeded.classification.class_ == "opinion"
    assert succeeded.error is None


async def test_paragraph_concurrency_below_one_raises_invalid_input() -> None:
    """A concurrency that admits no calls is rejected as bad input."""
    segmenter_model = FakeModel([])
    classifier_model = FakeModel([])

    with pytest.raises(ClassifierError) as exc_info:
        await classify_paragraph(
            {"paragraph": PARAGRAPH},
            classifier_model=classifier_model,
            segmenter_model=segmenter_model,
            concurrency=0,
        )

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert segmenter_model.calls == []


async def test_paragraph_missing_api_key_raises_before_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no model overrides, config loading fails fast on the missing key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ClassifierError) as exc_info:
        await classify_paragraph({"paragraph": PARAGRAPH})

    assert exc_info.value.code == ErrorCode.MISSING_API_KEY


def test_paragraph_sync_wrapper_returns_same_result_as_async() -> None:
    """The sync wrapper answers with what the async call would have answered."""
    segmenter_model = FakeModel([_Segments(statements=[FIRST_CLAUSE])])
    classifier_model = FakeModel(
        [Classification(**{"class": "fact", "confidence": 0.9})]
    )

    output = classify_paragraph_sync(
        {"paragraph": PARAGRAPH},
        classifier_model=classifier_model,
        segmenter_model=segmenter_model,
    )

    assert output.statements[0].classification.class_ == "fact"
