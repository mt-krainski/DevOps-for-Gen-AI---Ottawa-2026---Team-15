"""Tests for payload validation and identifier assignment in `factchecker.ingest`."""

import pytest

from factchecker.errors import InputValidationError
from factchecker.ingest import assign_ids, parse_input
from factchecker.models import InputPayload

WIRE_PAYLOAD = {
    "statements": [
        {
            "id": "s1",
            "surroundingContext": "Water is odd. Water boils at 100 C.",
            "statement": "Water boils at 100 C",
            "classification": {"class": "fact", "confidence": 0.7},
        }
    ]
}


def _payload(*statements: dict[str, object]) -> InputPayload:
    """Validate wire statements into a payload, so the tests share one shape."""
    return InputPayload.model_validate({"statements": list(statements)})


def _wire_statement(**overrides: object) -> dict[str, object]:
    """A valid wire statement, with the named keys replaced."""
    statement: dict[str, object] = {
        "surroundingContext": "A paragraph that holds the claim.",
        "statement": "Water boils at 100 C",
        "classification": {"class": "fact", "confidence": 0.7},
    }
    statement.update(overrides)
    return statement


def test_parse_input_validates_a_wire_payload() -> None:
    """A payload on the wire shape becomes the validated input model."""
    payload = parse_input(WIRE_PAYLOAD)

    assert isinstance(payload, InputPayload)
    assert payload.statements[0].id == "s1"
    assert payload.statements[0].classification.class_ == "fact"


def test_parse_input_names_a_missing_required_key() -> None:
    """A required key that is absent is named, because there is no value to quote."""
    raw = {"statements": [_wire_statement()]}
    del raw["statements"][0]["surroundingContext"]

    with pytest.raises(InputValidationError) as rejection:
        parse_input(raw)

    message = str(rejection.value)
    assert "statements.0.surroundingContext" in message
    assert "absent" in message


def test_parse_input_names_a_rejected_field_and_its_value() -> None:
    """A value outside the contract is named alongside the field that carried it."""
    raw = {
        "statements": [
            _wire_statement(classification={"class": "maybe", "confidence": 0.7})
        ]
    }

    with pytest.raises(InputValidationError) as rejection:
        parse_input(raw)

    message = str(rejection.value)
    assert "statements.0.classification.class" in message
    assert "'maybe'" in message


def test_parse_input_rejects_a_payload_that_is_not_the_input_shape() -> None:
    """A payload with no statements key is rejected by name."""
    with pytest.raises(InputValidationError) as rejection:
        parse_input({})

    assert "statements" in str(rejection.value)


def test_assign_ids_numbers_statements_that_carry_no_identifier() -> None:
    """An absent identifier becomes s1, s2, and so on by position in the input."""
    payload = _payload(_wire_statement(), _wire_statement(), _wire_statement())

    identified = assign_ids(payload)

    assert [statement.id for statement in identified] == ["s1", "s2", "s3"]


def test_assign_ids_preserves_a_supplied_identifier() -> None:
    """A supplied identifier passes through, and numbering still follows position."""
    payload = _payload(_wire_statement(id="claim-a"), _wire_statement())

    identified = assign_ids(payload)

    assert [statement.id for statement in identified] == ["claim-a", "s2"]


def test_assign_ids_carries_the_statement_fields_through() -> None:
    """The identified statement repeats the input fields it was built from."""
    payload = _payload(_wire_statement(statement="Water boils at 100 C"))

    identified = assign_ids(payload)

    assert identified[0].statement == "Water boils at 100 C"
    assert identified[0].surrounding_context == "A paragraph that holds the claim."
    assert identified[0].classification.class_ == "fact"


def test_assign_ids_rejects_a_repeated_identifier() -> None:
    """An integrator keying by identifier cannot be handed two entries sharing one."""
    payload = _payload(_wire_statement(id="s1"), _wire_statement(id="s1"))

    with pytest.raises(InputValidationError) as rejection:
        assign_ids(payload)

    assert "s1" in str(rejection.value)


def test_assign_ids_rejects_a_supplied_id_that_collides_with_an_assigned_one() -> None:
    """The uniqueness rule spans assigned and supplied identifiers alike."""
    payload = _payload(_wire_statement(), _wire_statement(id="s1"))

    with pytest.raises(InputValidationError) as rejection:
        assign_ids(payload)

    assert "s1" in str(rejection.value)
