from typing import Literal

from pydantic import BaseModel, Field

from statement_classifier.errors import ErrorCode


class Statement(BaseModel):
    surroundingContext: str
    statement: str


class Classification(BaseModel):
    class_: Literal["fact", "opinion"] = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class StatementError(BaseModel):
    code: ErrorCode
    message: str


class ClassifiedStatement(BaseModel):
    surroundingContext: str
    statement: str
    classification: Classification | None = None
    error: StatementError | None = None


class ClassifierInput(BaseModel):
    statements: list[Statement]


class ClassifierOutput(BaseModel):
    statements: list[ClassifiedStatement]
