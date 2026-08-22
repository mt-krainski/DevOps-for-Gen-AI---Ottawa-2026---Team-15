import asyncio
from typing import Protocol

from openai import AuthenticationError as OpenAIAuthenticationError
from pydantic import ValidationError

from statement_classifier.config import ClassifierConfig
from statement_classifier.errors import ErrorCode
from statement_classifier.models import Classification, Statement

PROMPT_TEMPLATE = """You are classifying a statement as either a checkable factual claim \
("fact") or a subjective, non-checkable statement ("opinion").

Use the surrounding context only to understand ambiguous references in the statement \
itself (e.g. "this", "it") — do not classify the surrounding context, only the statement.

Surrounding context:
{surrounding_context}

Statement to classify:
{statement}
"""

CALL_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.1


class StructuredClassifierModel(Protocol):
    """The seam tests mock: a runnable returning a `Classification` per call."""

    async def ainvoke(self, prompt: str) -> Classification: ...


class ClassificationFailure(Exception):
    """A per-statement failure: isolated onto that statement's `error` field."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AuthenticationFailure(Exception):
    """Invalid credentials: aborts the whole batch, not just this statement."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = ErrorCode.AUTH_ERROR
        self.message = message


def build_classifier_model(config: ClassifierConfig) -> StructuredClassifierModel:
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    return chat.with_structured_output(Classification)


async def classify_one(
    statement: Statement, model: StructuredClassifierModel
) -> Classification:
    prompt = PROMPT_TEMPLATE.format(
        surrounding_context=statement.surroundingContext,
        statement=statement.statement,
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        is_last_attempt = attempt == MAX_ATTEMPTS
        try:
            result = await asyncio.wait_for(
                model.ainvoke(prompt), timeout=CALL_TIMEOUT_SECONDS
            )
        except TimeoutError as exc:
            if is_last_attempt:
                raise ClassificationFailure(
                    ErrorCode.LLM_TIMEOUT, "LLM call timed out"
                ) from exc
        except OpenAIAuthenticationError as exc:
            # Invalid credentials won't be fixed by retrying; abort immediately.
            raise AuthenticationFailure(str(exc)) from exc
        except ValidationError as exc:
            if is_last_attempt:
                raise ClassificationFailure(
                    ErrorCode.PARSE_ERROR,
                    f"Model output failed schema validation: {exc}",
                ) from exc
        except Exception as exc:  # noqa: BLE001 - any other LLM/transport failure
            if is_last_attempt:
                raise ClassificationFailure(ErrorCode.LLM_ERROR, str(exc)) from exc
        else:
            if not isinstance(result, Classification):
                if is_last_attempt:
                    raise ClassificationFailure(
                        ErrorCode.PARSE_ERROR,
                        f"Model returned unexpected output type: {type(result)!r}",
                    )
            else:
                return result

        await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    # Unreachable: the loop above always returns or raises on the last attempt.
    raise AssertionError("unreachable")
