"""Tests for the paragraph-splitting call in `statement_classifier.segmenter`."""

import asyncio

import httpx
import openai
import pytest

from statement_classifier import segmenter as segmenter_module
from statement_classifier.errors import ClassifierError, ErrorCode
from statement_classifier.segmenter import _Segments, segment_paragraph
from tests.conftest import FakeModel

PARAGRAPH = (
    "Carney confirmed he was adding tariffs that would add costs for Canadians, "
    "but insisted they were necessary to retaliate against Trump's levies"
)


async def test_splits_a_paragraph_into_its_statements() -> None:
    """The paragraph comes back split, verbatim, in reading order."""
    model = FakeModel(
        [
            _Segments(
                statements=[
                    "Carney confirmed he was adding tariffs that would add "
                    "costs for Canadians",
                    "but insisted they were necessary to retaliate against "
                    "Trump's levies",
                ]
            )
        ]
    )

    result = await segment_paragraph(PARAGRAPH, model)

    assert result == [
        "Carney confirmed he was adding tariffs that would add costs for Canadians",
        "but insisted they were necessary to retaliate against Trump's levies",
    ]


async def test_prompt_includes_the_paragraph() -> None:
    """The paragraph reaches the model."""
    model = FakeModel([_Segments(statements=[PARAGRAPH])])

    await segment_paragraph(PARAGRAPH, model)

    assert len(model.calls) == 1
    assert PARAGRAPH in model.calls[0]


async def test_paragraph_with_no_statements_returns_empty_list() -> None:
    """A paragraph the model finds nothing splittable in yields no statements."""
    model = FakeModel([_Segments(statements=[])])

    result = await segment_paragraph(PARAGRAPH, model)

    assert result == []


async def test_llm_error_after_retries_exhausted_raises_segmentation_error() -> None:
    """Every attempt failing surfaces as one `SEGMENTATION_ERROR`, not three."""
    model = FakeModel([RuntimeError("boom")] * 3)

    with pytest.raises(ClassifierError) as exc_info:
        await segment_paragraph(PARAGRAPH, model)

    assert exc_info.value.code == ErrorCode.SEGMENTATION_ERROR
    assert len(model.calls) == 3


async def test_timeout_after_retries_exhausted_raises_segmentation_error() -> None:
    """A model that never answers is cut off by the call timeout."""

    async def hang(_prompt: str) -> _Segments:
        await asyncio.sleep(10)
        raise AssertionError("should not reach here")

    model = FakeModel([])
    model.ainvoke = hang

    original_timeout = segmenter_module.CALL_TIMEOUT_SECONDS
    segmenter_module.CALL_TIMEOUT_SECONDS = 0.01
    try:
        with pytest.raises(ClassifierError) as exc_info:
            await segment_paragraph(PARAGRAPH, model)
    finally:
        segmenter_module.CALL_TIMEOUT_SECONDS = original_timeout

    assert exc_info.value.code == ErrorCode.SEGMENTATION_ERROR


async def test_recovers_after_transient_failure() -> None:
    """One failed attempt is retried, and the retry's answer is returned."""
    model = FakeModel([RuntimeError("transient"), _Segments(statements=[PARAGRAPH])])

    result = await segment_paragraph(PARAGRAPH, model)

    assert result == [PARAGRAPH]
    assert len(model.calls) == 2


async def test_unparseable_output_after_retries_raises_segmentation_error() -> None:
    """Output that is not a `_Segments` is a segmentation failure."""
    model = FakeModel([{"not": "segments"}] * 3)

    with pytest.raises(ClassifierError) as exc_info:
        await segment_paragraph(PARAGRAPH, model)

    assert exc_info.value.code == ErrorCode.SEGMENTATION_ERROR


async def test_rejected_credential_raises_auth_error_immediately() -> None:
    """An invalid credential aborts on the first attempt, not after retries."""
    auth_error = openai.AuthenticationError(
        message="invalid api key",
        response=httpx.Response(
            status_code=401, request=httpx.Request("POST", "https://example.com")
        ),
        body=None,
    )
    model = FakeModel([auth_error])

    with pytest.raises(ClassifierError) as exc_info:
        await segment_paragraph(PARAGRAPH, model)

    assert exc_info.value.code == ErrorCode.AUTH_ERROR
    assert len(model.calls) == 1
