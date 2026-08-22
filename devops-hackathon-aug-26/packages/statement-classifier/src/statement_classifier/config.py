import os
from dataclasses import dataclass

from statement_classifier.errors import ClassifierError, ErrorCode

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_CONCURRENCY = 5


@dataclass(frozen=True)
class ClassifierConfig:
    api_key: str
    model: str
    base_url: str


def load_config() -> ClassifierConfig:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ClassifierError(
            ErrorCode.MISSING_API_KEY,
            "OPENROUTER_API_KEY environment variable is not set",
        )

    return ClassifierConfig(
        api_key=api_key,
        model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    )
