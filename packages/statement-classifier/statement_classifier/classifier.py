"""One statement, one classification: the model call and the retries around it."""

import asyncio
from typing import Protocol

from openai import AuthenticationError as OpenAIAuthenticationError
from pydantic import ValidationError

from statement_classifier.config import ClassifierConfig
from statement_classifier.errors import (
    AuthenticationFailure,
    ClassificationFailure,
    ErrorCode,
)
from statement_classifier.models import Classification, Statement

PROMPT_TEMPLATE = """You are classifying a statement as either a checkable factual \
claim ("fact") or a subjective, non-checkable statement ("opinion").

Use the surrounding context only to understand ambiguous references in the \
statement itself (e.g. "this", "it") — do not classify the surrounding context, \
only the statement.

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

    async def ainvoke(self, prompt: str) -> Classification:
        """Classify the one statement the prompt carries."""
        ...


def build_classifier_model(config: ClassifierConfig) -> StructuredClassifierModel:
    """Bind the configured gateway to the classification schema.

    Args:
        config: The gateway to call.

    Returns:
        A runnable that answers with a `Classification`.
    """
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
    """Classify one statement, retrying a transient failure up to `MAX_ATTEMPTS`.

    Args:
        statement: The statement to classify, with the context that disambiguates it.
        model: The runnable to call.

    Returns:
        The classification the model answered with.

    Raises:
        ClassificationFailure: Every attempt failed. Isolated onto this statement.
        AuthenticationFailure: The credential was rejected. Aborts the batch.
    """
    prompt = PROMPT_TEMPLATE.format(
        surrounding_context=statement.surrounding_context,
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
        # Any other LLM or transport failure: retried, then reported as one code.
        except Exception as exc:
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
