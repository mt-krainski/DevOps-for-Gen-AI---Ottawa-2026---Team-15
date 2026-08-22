"""Ingest: validate a raw payload and give every statement an identifier."""

from collections.abc import Mapping

from pydantic import ValidationError

from factchecker.errors import InputValidationError
from factchecker.models import IdentifiedStatement, InputPayload


def parse_input(raw: Mapping[str, object]) -> InputPayload:
    """Validate a raw payload against the input contract.

    Args:
        raw: The payload as read from JSON, before any validation.

    Returns:
        The validated payload.

    Raises:
        InputValidationError: The payload does not satisfy the contract.
    """
    try:
        return InputPayload.model_validate(raw)
    except ValidationError as rejection:
        raise InputValidationError(_describe(rejection)) from rejection


def assign_ids(payload: InputPayload) -> list[IdentifiedStatement]:
    """Give every statement an identifier, keeping the ones the caller supplied.

    Args:
        payload: The validated input payload.

    Returns:
        One identified statement per input statement, in input order.

    Raises:
        InputValidationError: Two statements share an identifier.
    """
    identified: list[IdentifiedStatement] = []
    taken: set[str] = set()
    for position, statement in enumerate(payload.statements, start=1):
        identifier = statement.id if statement.id is not None else f"s{position}"
        if identifier in taken:
            raise InputValidationError(
                f"statement identifier {identifier!r} appears more than once; "
                "identifiers must be unique across the payload"
            )
        taken.add(identifier)
        identified.append(
            IdentifiedStatement(
                id=identifier,
                surrounding_context=statement.surrounding_context,
                statement=statement.statement,
                classification=statement.classification,
            )
        )
    return identified


def _describe(rejection: ValidationError) -> str:
    """Render a validation failure as a message naming each field and its value."""
    failures: list[str] = []
    for detail in rejection.errors():
        field = ".".join(str(part) for part in detail["loc"])
        if detail["type"] == "missing":
            failures.append(f"{field} is absent")
        else:
            failures.append(f"{field} is {detail['input']!r}: {detail['msg']}")
    return "; ".join(failures)
