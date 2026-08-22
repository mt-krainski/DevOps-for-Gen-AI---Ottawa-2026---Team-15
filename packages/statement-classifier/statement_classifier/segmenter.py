"""One paragraph, many statements: splitting text before it's classified."""

import asyncio
from typing import Protocol

from openai import AuthenticationError as OpenAIAuthenticationError
from pydantic import BaseModel, ValidationError

from statement_classifier.config import ClassifierConfig
from statement_classifier.errors import ClassifierError, ErrorCode

SEGMENTATION_PROMPT_TEMPLATE = """Split the following paragraph into separate \
statements, each one a self-contained clause or sentence that could be judged \
independently as fact or opinion.

Preserve the original wording verbatim — do not paraphrase, summarize, or \
correct it — and preserve reading order. Split on a clause boundary where a \
conjunction changes what is being claimed (e.g. "but", "although", "however"), \
not only on sentence-ending punctuation.

Paragraph:
{paragraph}
"""

CALL_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.1


class _Segments(BaseModel):
    """The schema the segmentation model answers with."""

    statements: list[str]


class StructuredSegmenterModel(Protocol):
    """The seam tests mock: a runnable returning the paragraph's statements."""

    async def ainvoke(self, prompt: str) -> _Segments:
        """Split the one paragraph the prompt carries."""
        ...


def build_segmenter_model(config: ClassifierConfig) -> StructuredSegmenterModel:
    """Bind the configured gateway to the segmentation schema.

    Args:
        config: The gateway to call.

    Returns:
        A runnable that answers with a list of statements.
    """
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    return chat.with_structured_output(_Segments)


async def segment_paragraph(
    paragraph: str, model: StructuredSegmenterModel
) -> list[str]:
    """Split a paragraph into statements, retrying a transient failure.

    Unlike a per-statement classification failure, a segmentation failure has
    nothing to isolate it onto — a paragraph that can't be split yields no
    statements to classify — so it aborts the whole call rather than being
    reported per item.

    Args:
        paragraph: The text to split.
        model: The runnable to call.

    Returns:
        The statements the paragraph was split into, in reading order.

    Raises:
        ClassifierError: Every attempt failed (`SEGMENTATION_ERROR`), or the
            credential was rejected (`AUTH_ERROR`).
    """
    prompt = SEGMENTATION_PROMPT_TEMPLATE.format(paragraph=paragraph)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        is_last_attempt = attempt == MAX_ATTEMPTS
        try:
            result = await asyncio.wait_for(
                model.ainvoke(prompt), timeout=CALL_TIMEOUT_SECONDS
            )
        except TimeoutError as exc:
            if is_last_attempt:
                raise ClassifierError(
                    ErrorCode.SEGMENTATION_ERROR, "Segmentation call timed out"
                ) from exc
        except OpenAIAuthenticationError as exc:
            # Invalid credentials won't be fixed by retrying; abort immediately.
            raise ClassifierError(ErrorCode.AUTH_ERROR, str(exc)) from exc
        except ValidationError as exc:
            if is_last_attempt:
                raise ClassifierError(
                    ErrorCode.SEGMENTATION_ERROR,
                    f"Segmentation output failed schema validation: {exc}",
                ) from exc
        # Any other LLM or transport failure: retried, then reported as one code.
        except Exception as exc:
            if is_last_attempt:
                raise ClassifierError(ErrorCode.SEGMENTATION_ERROR, str(exc)) from exc
        else:
            if not isinstance(result, _Segments):
                if is_last_attempt:
                    raise ClassifierError(
                        ErrorCode.SEGMENTATION_ERROR,
                        f"Segmentation returned unexpected output type: "
                        f"{type(result)!r}",
                    )
            else:
                return result.statements

        await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    # Unreachable: the loop above always returns or raises on the last attempt.
    raise AssertionError("unreachable")
