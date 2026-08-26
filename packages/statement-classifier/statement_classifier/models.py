"""The wire contract: the JSON shapes this package reads and writes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from statement_classifier.errors import ErrorCode

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


class Statement(_WireModel):
    surrounding_context: str
    statement: str


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


class StatementError(_WireModel):
    code: ErrorCode
    message: str


class ClassifiedStatement(_WireModel):
    surrounding_context: str
    statement: str
    classification: Classification | None = None
    error: StatementError | None = None


class ClassifierInput(_WireModel):
    statements: list[Statement]


class ClassifierOutput(_WireModel):
    statements: list[ClassifiedStatement]


class TextInput(_WireModel):
    """Text mode's input: raw text this package splits itself."""

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value
