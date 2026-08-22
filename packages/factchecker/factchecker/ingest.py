"""Ingest: validate a raw payload and give every statement an identifier."""

from collections.abc import Mapping

from pydantic import ValidationError

from factchecker.errors import InputValidationError
from factchecker.models import IdentifiedStatement, InputPayload

_MAX_REPORTED_FAILURES = 5
_MAX_QUOTED_WIDTH = 120


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
    """Render a validation failure as a message naming each field and its value.

    The whole message reaches the user as one log record, so the number of
    failures it names and the width of each value it quotes are both bounded. A
    payload that fails in every statement would otherwise echo itself into the
    log.
    """
    details = rejection.errors()
    failures: list[str] = []
    for detail in details[:_MAX_REPORTED_FAILURES]:
        field = ".".join(str(part) for part in detail["loc"])
        if detail["type"] == "missing":
            failures.append(f"{field} is absent")
        else:
            failures.append(f"{field} is {_quoted(detail['input'])}: {detail['msg']}")
    unreported = len(details) - len(failures)
    if unreported:
        failures.append(f"and {unreported} more not shown")
    return "; ".join(failures)


def _quoted(value: object) -> str:
    """The value's repr, cut to a width one log record can carry."""
    rendered = repr(value)
    if len(rendered) <= _MAX_QUOTED_WIDTH:
        return rendered
    return f"{rendered[:_MAX_QUOTED_WIDTH]}... (truncated)"
