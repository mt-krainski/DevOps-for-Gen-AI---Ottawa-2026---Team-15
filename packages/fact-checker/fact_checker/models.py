"""The wire contract: the JSON shapes this package reads and writes."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from fact_checker.errors import CheckError, ErrorCode

StatementClass = Literal["fact", "opinion"]
Verdict = Literal["supported", "refuted", "mixed", "unverifiable"]

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class _WireModel(BaseModel):
    """Base for every model on the wire: camelCase JSON, snake_case Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="ignore",
    )


class Classification(_WireModel):
    # `class` is a keyword, so the field is `class_`. The alias generator fills
    # any alias slot left empty, and it fills it with the generated name
    # `class_` — not with `alias`. Writing all three slots leaves nothing to
    # fill, so the wire key never depends on which of them `Field` populates.
    class_: StatementClass = Field(
        alias="class",
        validation_alias="class",
        serialization_alias="class",
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("class_", mode="before")
    @classmethod
    def _name_the_offending_class(cls, value: object) -> object:
        # A `Literal` rejects a third label, but its message lists the labels it
        # accepts rather than the one that arrived. The value that failed is
        # what the operator needs to see.
        if value not in ("fact", "opinion"):
            raise ValueError(
                f"classification.class must be 'fact' or 'opinion', got {value!r}"
            )
        return value


class InputStatement(_WireModel):
    id: str | None = None
    surrounding_context: str
    statement: str
    classification: Classification


class CheckerInput(_WireModel):
    statements: list[InputStatement]


class Reference(_WireModel):
    id: str
    source: str
    excerpt: str


class Ruling(_WireModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    references: list[Reference]


class StatementError(_WireModel):
    code: ErrorCode
    message: str


class CheckedStatement(_WireModel):
    id: str
    surrounding_context: str
    statement: str
    classification: Classification
    ruling: Ruling | None = None
    error: StatementError | None = None


class Counts(_WireModel):
    total: int
    checked: int
    skipped: int
    failed: int


class Usage(_WireModel):
    prompt_tokens: int
    completion_tokens: int
    searches: int


class RunMeta(_WireModel):
    model: str
    started_at: str
    finished_at: str
    counts: Counts
    usage: Usage


class CheckerOutput(_WireModel):
    meta: RunMeta
    statements: list[CheckedStatement]


def assign_identifiers(statements: Sequence[InputStatement]) -> list[str]:
    """Name every statement, in input order.

    Args:
        statements: The run's statements, as they were submitted.

    Returns:
        One identifier per statement: the supplied `id` where there is one, and
        `s` followed by the 1-based input position where there is not.

    Raises:
        CheckError: Two statements ended up under one identifier, which would
            make the output ambiguous to every consumer.
    """
    identifiers = [
        statement.id if statement.id is not None else f"s{position}"
        for position, statement in enumerate(statements, start=1)
    ]

    seen: set[str] = set()
    for identifier in identifiers:
        if identifier in seen:
            raise CheckError(
                ErrorCode.INVALID_INPUT,
                f"statement id {identifier!r} is used twice; ids must be unique",
            )
        seen.add(identifier)

    return identifiers


def format_timestamp(moment: datetime) -> str:
    """Render a moment in the wire form, `2026-08-22T14:03:11Z`.

    Args:
        moment: The moment to render. One carrying no offset is read as UTC.

    Returns:
        The moment in UTC, to the second, with a trailing `Z`.
    """
    utc_moment = (
        moment.astimezone(UTC)
        if moment.tzinfo is not None
        else moment.replace(tzinfo=UTC)
    )
    return utc_moment.strftime(_TIMESTAMP_FORMAT)
