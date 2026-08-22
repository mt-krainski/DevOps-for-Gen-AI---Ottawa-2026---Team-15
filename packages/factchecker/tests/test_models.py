"""Tests for the wire contract in `factchecker.models`."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from factchecker.models import (
    CheckError,
    Classification,
    Counts,
    IdentifiedStatement,
    InputPayload,
    InputStatement,
    Meta,
    OutputPayload,
    OutputStatement,
    Reference,
    Ruling,
    Usage,
)

WIRE_STATEMENT = {
    "id": "s1",
    "surroundingContext": "Water is odd. Water boils at 100 C. The tables agree.",
    "statement": "Water boils at 100 C",
    "classification": {"class": "fact", "confidence": 0.7},
}

WIRE_META = {
    "model": "anthropic/claude-sonnet-4",
    "startedAt": "2026-08-22T14:03:11Z",
    "finishedAt": "2026-08-22T14:05:47Z",
    "counts": {"total": 1, "checked": 1, "skipped": 0, "failed": 0},
    "usage": {"promptTokens": 184203, "completionTokens": 9877, "searches": 74},
}

WIRE_RULING = {
    "verdict": "supported",
    "confidence": 0.92,
    "justification": "At standard pressure water boils at 100 C [1].",
    "references": [
        {
            "id": "1",
            "source": "https://example.invalid/boiling-point",
            "excerpt": "At 1 atm, water boils at 100 C",
        }
    ],
}


def test_input_statement_reads_camel_case_wire_keys() -> None:
    """Wire keys are camelCase and land on the snake_case fields."""
    statement = InputStatement.model_validate(WIRE_STATEMENT)

    assert statement.id == "s1"
    assert statement.surrounding_context == WIRE_STATEMENT["surroundingContext"]
    assert statement.statement == "Water boils at 100 C"
    assert statement.classification.class_ == "fact"
    assert statement.classification.confidence == 0.7


def test_input_statement_writes_camel_case_wire_keys() -> None:
    """Serialization returns the wire shape it was read from."""
    statement = InputStatement.model_validate(WIRE_STATEMENT)

    assert statement.model_dump(mode="json") == WIRE_STATEMENT


def test_input_statement_reads_python_field_names() -> None:
    """The Python field names are accepted alongside the wire aliases."""
    statement = InputStatement.model_validate(
        {
            "id": "s1",
            "surrounding_context": "Water is odd.",
            "statement": "Water boils at 100 C",
            "classification": {"class_": "fact", "confidence": 0.7},
        }
    )

    assert statement.surrounding_context == "Water is odd."
    assert statement.classification.class_ == "fact"


def test_classification_writes_class_and_not_class_underscore() -> None:
    """The reserved-word field reaches the wire as `class`."""
    wire = Classification(class_="fact", confidence=0.7).model_dump(mode="json")

    assert wire == {"class": "fact", "confidence": 0.7}


def test_classification_reads_the_class_alias() -> None:
    """The wire key `class` lands on `class_`."""
    wire = {"class": "opinion", "confidence": 0.1}

    assert Classification.model_validate(wire).class_ == "opinion"


def test_input_statement_drops_a_field_the_contract_does_not_name() -> None:
    """A classifier that adds a field does not break this package."""
    statement = InputStatement.model_validate(
        {**WIRE_STATEMENT, "classifierBuild": "2026-08-01"}
    )

    assert statement.model_dump(mode="json") == WIRE_STATEMENT


def test_input_statement_id_defaults_to_none() -> None:
    """An input statement may arrive without an identifier."""
    without_id = {key: value for key, value in WIRE_STATEMENT.items() if key != "id"}

    assert InputStatement.model_validate(without_id).id is None


def test_identified_statement_requires_an_id() -> None:
    """Once identifiers are assigned, every statement carries one."""
    without_id = {key: value for key, value in WIRE_STATEMENT.items() if key != "id"}

    with pytest.raises(ValidationError) as caught:
        IdentifiedStatement.model_validate(without_id)

    assert "id" in str(caught.value)


def test_input_payload_reads_a_list_of_statements() -> None:
    """The payload wraps the statements it was given, in order."""
    payload = InputPayload.model_validate(
        {"statements": [WIRE_STATEMENT, {**WIRE_STATEMENT, "id": "s2"}]}
    )

    assert [statement.id for statement in payload.statements] == ["s1", "s2"]


def test_unrecognised_class_value_is_rejected_naming_that_value() -> None:
    """A third label is a deliberate change here, not a silent pass-through."""
    with pytest.raises(ValidationError) as caught:
        Classification.model_validate({"class": "rumour", "confidence": 0.7})

    assert "rumour" in str(caught.value)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_classification_confidence_outside_zero_to_one_is_rejected(
    confidence: float,
) -> None:
    """Confidence is a closed interval from 0 to 1."""
    with pytest.raises(ValidationError):
        Classification.model_validate({"class": "fact", "confidence": confidence})


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_ruling_confidence_outside_zero_to_one_is_rejected(confidence: float) -> None:
    """A ruling's confidence carries the same closed interval."""
    with pytest.raises(ValidationError):
        Ruling.model_validate({**WIRE_RULING, "confidence": confidence})


def test_meta_writes_a_zero_utc_offset_as_z() -> None:
    """A zero UTC offset serializes with the `Z` suffix, not `+00:00`."""
    meta = Meta(
        model="anthropic/claude-sonnet-4",
        started_at=datetime(2026, 8, 22, 14, 3, 11, tzinfo=UTC),
        finished_at=datetime(2026, 8, 22, 14, 5, 47, tzinfo=UTC),
        counts=Counts(total=1, checked=1, skipped=0, failed=0),
        usage=Usage(prompt_tokens=184203, completion_tokens=9877, searches=74),
    )

    assert meta.model_dump(mode="json") == WIRE_META


def test_output_statement_writes_ruling_and_error_when_both_are_null() -> None:
    """An opinion writes both keys so every consumer reads one shape."""
    statement = OutputStatement(
        id="s1",
        surrounding_context="Water is odd.",
        statement="Water is the best liquid",
        classification=Classification(class_="opinion", confidence=0.4),
        ruling=None,
        error=None,
    )

    wire = statement.model_dump(mode="json")

    assert {"ruling", "error"} <= wire.keys()
    assert wire["ruling"] is None
    assert wire["error"] is None


def test_output_statement_writes_an_error_beside_a_null_ruling() -> None:
    """A failed check names what went wrong and rules on nothing."""
    statement = OutputStatement(
        id="s1",
        surrounding_context="Water is odd.",
        statement="Water boils at 100 C",
        classification=Classification(class_="fact", confidence=0.7),
        ruling=None,
        error=CheckError(kind="timeout", message="statement exceeded 240 seconds"),
    )

    wire = statement.model_dump(mode="json")

    assert wire["ruling"] is None
    assert wire["error"] == {
        "kind": "timeout",
        "message": "statement exceeded 240 seconds",
    }


def test_ruling_writes_its_references() -> None:
    """A ruling carries the sources behind it."""
    ruling = Ruling(
        verdict="supported",
        confidence=0.92,
        justification="At standard pressure water boils at 100 C [1].",
        references=[
            Reference(
                id="1",
                source="https://example.invalid/boiling-point",
                excerpt="At 1 atm, water boils at 100 C",
            )
        ],
    )

    assert ruling.model_dump(mode="json") == WIRE_RULING


def test_output_payload_round_trips_the_wire_shape() -> None:
    """The full output envelope reads and writes the same JSON."""
    wire = {
        "meta": WIRE_META,
        "statements": [{**WIRE_STATEMENT, "ruling": WIRE_RULING, "error": None}],
    }

    assert OutputPayload.model_validate(wire).model_dump(mode="json") == wire
