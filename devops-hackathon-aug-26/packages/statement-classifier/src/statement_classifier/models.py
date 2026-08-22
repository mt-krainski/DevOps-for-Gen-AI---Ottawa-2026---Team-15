from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Statement(BaseModel):
    statement: str
    surroundingContext: str


class Classification(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: Literal["fact", "opinion"] = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
