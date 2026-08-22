"""The wire contract: the JSON shapes this package reads and writes."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Verdict = Literal["supported", "refuted", "mixed", "unverifiable"]
StatementClass = Literal["fact", "opinion"]


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


class InputStatement(_WireModel):
    id: str | None = None
    surrounding_context: str
    statement: str
    classification: Classification


class InputPayload(_WireModel):
    statements: list[InputStatement]


class IdentifiedStatement(_WireModel):
    id: str
    surrounding_context: str
    statement: str
    classification: Classification


class Reference(_WireModel):
    id: str
    source: str
    excerpt: str


class Ruling(_WireModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    references: list[Reference]


class CheckError(_WireModel):
    kind: str
    message: str


class OutputStatement(_WireModel):
    id: str
    surrounding_context: str
    statement: str
    classification: Classification
    ruling: Ruling | None
    error: CheckError | None


class Counts(_WireModel):
    total: int
    checked: int
    skipped: int
    failed: int


class Usage(_WireModel):
    prompt_tokens: int
    completion_tokens: int
    searches: int


class Meta(_WireModel):
    model: str
    started_at: AwareDatetime
    finished_at: AwareDatetime
    counts: Counts
    usage: Usage


class OutputPayload(_WireModel):
    meta: Meta
    statements: list[OutputStatement]
