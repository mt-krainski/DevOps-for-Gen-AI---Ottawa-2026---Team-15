import os
from functools import lru_cache
from typing import Any, Protocol

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from statement_classifier.models import Classification, Statement

DEFAULT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class StructuredOutputModel(Protocol):
    async def ainvoke(self, input: Any) -> Any: ...


class ChatModel(Protocol):
    """The slice of LangChain's chat-model interface classify_statement needs.

    Kept narrow (rather than typing against ``BaseChatModel`` directly) so
    tests can substitute a minimal fake at this exact seam.
    """

    def with_structured_output(
        self, schema: type[Classification]
    ) -> StructuredOutputModel: ...


SYSTEM_PROMPT = (
    "You classify a single statement as either a checkable factual claim "
    '("fact") or a subjective, unverifiable statement ("opinion"). '
    "You are given surrounding context only to help disambiguate the "
    "statement — classify the statement itself, not the context around it. "
    "Report your confidence in the classification as a number between 0 and 1."
)


@lru_cache(maxsize=1)
def _build_chat_model() -> ChatOpenAI:
    # Explicit env var, never falling back to ChatOpenAI's own OPENAI_API_KEY
    # lookup: an unrelated OpenAI key in the environment must not silently
    # get sent to OpenRouter under a different provider's credentials.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required to classify statements"
        )
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        api_key=SecretStr(api_key),
    )


async def classify_statement(
    statement: Statement, *, model: ChatModel | None = None
) -> Classification:
    chat_model = model if model is not None else _build_chat_model()
    structured_model = chat_model.with_structured_output(Classification)
    human_message = (
        f"Surrounding context:\n{statement.surroundingContext}\n\n"
        f"Statement to classify:\n{statement.statement}"
    )
    result = await structured_model.ainvoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", human_message),
        ]
    )
    if not isinstance(result, Classification):
        raise TypeError(
            f"expected structured output to be a Classification, got {type(result)!r}"
        )
    return result
