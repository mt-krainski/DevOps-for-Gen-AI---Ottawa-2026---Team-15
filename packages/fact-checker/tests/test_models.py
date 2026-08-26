"""Tests for the wire contract in `fact_checker.models`."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import fact_checker
from fact_checker.errors import CheckError, ErrorCode
from fact_checker.models import (
    CheckedStatement,
    CheckerInput,
    CheckerOutput,
    Classification,
    Counts,
    InputStatement,
    Reference,
    Ruling,
    RunMeta,
    StatementError,
    Usage,
    assign_identifiers,
    format_timestamp,
)

CONTEXT = "We are testing. This is a test. Test is now over."

INPUT_PAYLOAD = {
    "statements": [
        {
            "id": "s1",
            "surroundingContext": CONTEXT,
            "statement": "This is a test",
            "classification": {"class": "fact", "confidence": 0.7},
        }
    ]
}


def make_statement(identifier: str | None) -> InputStatement:
    """Build one factual input statement under the given identifier."""
    return InputStatement(
        id=identifier,
        surrounding_context=CONTEXT,
        statement="This is a test",
        classification=Classification(**{"class": "fact", "confidence": 0.7}),
    )


def test_input_payload_round_trips_through_camel_case() -> None:
    """The camelCase keys read into snake_case fields and dump back unchanged."""
    checker_input = CheckerInput.model_validate(INPUT_PAYLOAD)

    statement = checker_input.statements[0]
    assert statement.surrounding_context == CONTEXT
    assert statement.classification.class_ == "fact"
    assert checker_input.model_dump(mode="json") == INPUT_PAYLOAD


def test_output_payload_writes_camel_case() -> None:
    """Every multi-word key leaves as camelCase, `class` included."""
    output = CheckerOutput(
        meta=RunMeta(
            model="google/gemma-4-31b-it",
            started_at="2026-08-22T14:03:11Z",
            finished_at="2026-08-22T14:05:47Z",
            counts=Counts(total=1, checked=1, skipped=0, failed=0),
            usage=Usage(prompt_tokens=184203, completion_tokens=9877, searches=74),
        ),
        statements=[
            CheckedStatement(
                id="s1",
                surrounding_context=CONTEXT,
                statement="This is a test",
                classification=Classification(**{"class": "fact", "confidence": 0.7}),
                ruling=Ruling(
                    verdict="supported",
                    confidence=0.92,
                    justification="The evidence backs it [1].",
                    references=[
                        Reference(
                            id="1",
                            source="https://example.test/page",
                            excerpt="It is a test",
                        )
                    ],
                ),
            )
        ],
    )

    payload = output.model_dump(mode="json")

    assert payload["meta"]["startedAt"] == "2026-08-22T14:03:11Z"
    assert payload["meta"]["finishedAt"] == "2026-08-22T14:05:47Z"
    assert payload["meta"]["usage"]["promptTokens"] == 184203
    assert payload["meta"]["usage"]["completionTokens"] == 9877
    assert payload["statements"][0]["surroundingContext"] == CONTEXT
    assert payload["statements"][0]["classification"]["class"] == "fact"
    assert payload["statements"][0]["ruling"]["references"][0]["source"] == (
        "https://example.test/page"
    )


def test_checked_statement_serializes_both_outcome_keys_as_null() -> None:
    """An opinion carries `ruling` and `error` as present, null keys."""
    checked = CheckedStatement(
        id="s1",
        surrounding_context=CONTEXT,
        statement="This is a test",
        classification=Classification(**{"class": "opinion", "confidence": 0.7}),
    )

    payload = checked.model_dump(mode="json")

    assert payload["ruling"] is None
    assert payload["error"] is None


def test_checked_statement_serializes_an_error_beside_a_null_ruling() -> None:
    """A failed statement names what went wrong and rules on nothing."""
    checked = CheckedStatement(
        id="s1",
        surrounding_context=CONTEXT,
        statement="This is a test",
        classification=Classification(**{"class": "fact", "confidence": 0.7}),
        error=StatementError(code=ErrorCode.TOOL_ERROR, message="search failed"),
    )

    payload = checked.model_dump(mode="json")

    assert payload["ruling"] is None
    assert payload["error"] == {"code": "TOOL_ERROR", "message": "search failed"}


def test_class_outside_the_vocabulary_names_the_offending_value() -> None:
    """A third label fails validation rather than passing through unchecked."""
    with pytest.raises(ValidationError) as raised:
        Classification.model_validate({"class": "speculation", "confidence": 0.5})

    assert "speculation" in str(raised.value)


def test_classification_confidence_above_one_is_rejected() -> None:
    """`confidence` is a probability, so 1.5 is not a value it can hold."""
    with pytest.raises(ValidationError):
        Classification.model_validate({"class": "fact", "confidence": 1.5})


def test_ruling_confidence_below_zero_is_rejected() -> None:
    """The ruling's confidence is bounded the same way the classifier's is."""
    with pytest.raises(ValidationError):
        Ruling(
            verdict="supported",
            confidence=-0.1,
            justification="The evidence backs it [1].",
            references=[],
        )


def test_missing_surrounding_context_is_rejected() -> None:
    """The agent needs the surroundings, so the key is required by name."""
    with pytest.raises(ValidationError) as raised:
        InputStatement.model_validate(
            {
                "statement": "This is a test",
                "classification": {"class": "fact", "confidence": 0.7},
            }
        )

    assert "surroundingContext" in str(raised.value)


def test_assign_identifiers_fills_the_gaps_by_input_position() -> None:
    """A supplied id passes through; an absent one takes its 1-based index."""
    statements = [make_statement(None), make_statement("custom"), make_statement(None)]

    assert assign_identifiers(statements) == ["s1", "custom", "s3"]


def test_assign_identifiers_rejects_a_repeat_by_name() -> None:
    """A supplied id that collides with an assigned one fails the whole run."""
    statements = [make_statement("s2"), make_statement(None)]

    with pytest.raises(CheckError) as raised:
        assign_identifiers(statements)

    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert "s2" in raised.value.message


def test_format_timestamp_renders_the_spec_form() -> None:
    """The wire form is second-resolution UTC with a trailing `Z`."""
    moment = datetime(2026, 8, 22, 14, 3, 11, 123456, tzinfo=UTC)

    assert format_timestamp(moment) == "2026-08-22T14:03:11Z"


def test_format_timestamp_converts_an_offset_to_utc() -> None:
    """A moment recorded at another offset is written in UTC."""
    moment = datetime(2026, 8, 22, 16, 3, 11, tzinfo=timezone(timedelta(hours=2)))

    assert format_timestamp(moment) == "2026-08-22T14:03:11Z"


def test_format_timestamp_reads_a_naive_moment_as_utc() -> None:
    """A moment with no offset is taken as UTC rather than as local time."""
    moment = datetime(2026, 8, 22, 14, 3, 11)

    assert format_timestamp(moment) == "2026-08-22T14:03:11Z"


def test_the_package_root_re_exports_the_public_surface() -> None:
    """Everything `__all__` promises resolves on the package root."""
    missing = [name for name in fact_checker.__all__ if not hasattr(fact_checker, name)]

    assert missing == []
    assert {"CheckerInput", "CheckerOutput", "CheckError", "load_config"} <= set(
        fact_checker.__all__
    )
