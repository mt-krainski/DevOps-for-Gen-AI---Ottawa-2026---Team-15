"""The batch: every statement classified concurrently, one failure isolated."""

import asyncio
from typing import Any

from pydantic import ValidationError

from statement_classifier.classifier import (
    StructuredClassifierModel,
    build_classifier_model,
    classify_one,
)
from statement_classifier.config import DEFAULT_CONCURRENCY, load_config
from statement_classifier.context import windows_for
from statement_classifier.errors import (
    AuthenticationFailure,
    ClassificationFailure,
    ClassifierError,
    ErrorCode,
)
from statement_classifier.models import (
    ClassifiedStatement,
    ClassifierInput,
    ClassifierOutput,
    Statement,
    StatementError,
    TextInput,
)
from statement_classifier.segmenter import (
    StructuredSegmenterModel,
    build_segmenter_model,
    segment_text,
)


def _coerce_input(payload: ClassifierInput | dict[str, Any]) -> ClassifierInput:
    if isinstance(payload, ClassifierInput):
        return payload
    try:
        return ClassifierInput.model_validate(payload)
    except ValidationError as exc:
        raise ClassifierError(ErrorCode.INVALID_INPUT, str(exc)) from exc


def _coerce_text_input(
    payload: TextInput | dict[str, Any],
) -> TextInput:
    if isinstance(payload, TextInput):
        return payload
    try:
        return TextInput.model_validate(payload)
    except ValidationError as exc:
        raise ClassifierError(ErrorCode.INVALID_INPUT, str(exc)) from exc


async def _classify_with_isolation(
    statement: Statement, model: StructuredClassifierModel, semaphore: asyncio.Semaphore
) -> ClassifiedStatement:
    async with semaphore:
        try:
            classification = await classify_one(statement, model)
        except ClassificationFailure as exc:
            return ClassifiedStatement(
                surrounding_context=statement.surrounding_context,
                statement=statement.statement,
                classification=None,
                error=StatementError(code=exc.code, message=exc.message),
            )
        return ClassifiedStatement(
            surrounding_context=statement.surrounding_context,
            statement=statement.statement,
            classification=classification,
            error=None,
        )


async def _classify_batch(
    statements: list[Statement],
    model: StructuredClassifierModel,
    concurrency: int,
) -> list[ClassifiedStatement]:
    """Classify every statement concurrently, isolating per-statement failures.

    The shared core of `classify_statements` and `classify_text`: once
    there's a flat list of `Statement`s and a model to call, both do the same
    concurrency-bounded, isolation-and-auth-short-circuit dance.

    Args:
        statements: The statements to classify. Each carries its own context.
        model: The runnable to call.
        concurrency: The ceiling on LLM calls in flight at once.

    Returns:
        One `ClassifiedStatement` per input statement, in the same order.

    Raises:
        ClassifierError: The credential was rejected. Aborts before returning
            anything partial.
    """
    if not statements:
        return []

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.ensure_future(_classify_with_isolation(statement, model, semaphore))
        for statement in statements
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    auth_failure = next(
        (
            exc
            for task in done
            if (exc := task.exception()) is not None
            and isinstance(exc, AuthenticationFailure)
        ),
        None,
    )
    if auth_failure is not None:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise ClassifierError(ErrorCode.AUTH_ERROR, auth_failure.message)

    if pending:
        await asyncio.wait(pending)

    return [task.result() for task in tasks]


async def classify_statements(
    payload: ClassifierInput | dict[str, Any],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: StructuredClassifierModel | None = None,
) -> ClassifierOutput:
    """Classify every statement in the batch concurrently.

    A per-statement LLM failure is isolated onto that statement's `error` field;
    only malformed input or missing/invalid credentials abort the whole call.

    Args:
        payload: The batch, as a `ClassifierInput` or the dict it validates from.
        concurrency: The ceiling on LLM calls in flight at once.
        model: The runnable to call. `None` builds one from the environment.

    Returns:
        The same batch, each statement carrying a classification or an error.

    Raises:
        ClassifierError: The input is malformed, the concurrency is below one, or
            the credential is missing or rejected. Nothing partial is returned.
    """
    classifier_input = _coerce_input(payload)

    if concurrency < 1:
        raise ClassifierError(
            ErrorCode.INVALID_INPUT, f"concurrency must be >= 1, got {concurrency}"
        )

    if not classifier_input.statements:
        return ClassifierOutput(statements=[])

    if model is None:
        config = load_config()
        model = build_classifier_model(config)

    results = await _classify_batch(classifier_input.statements, model, concurrency)
    return ClassifierOutput(statements=results)


async def classify_text(
    payload: TextInput | dict[str, Any],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    classifier_model: StructuredClassifierModel | None = None,
    segmenter_model: StructuredSegmenterModel | None = None,
) -> ClassifierOutput:
    """Split the text into statements, then classify each concurrently.

    Each statement is classified against the sentences around it rather than
    against the whole input, so an article-length input does not put itself
    into every classification call. A per-statement classification failure is
    isolated the same way as in `classify_statements`; a segmentation failure
    has no per-item granularity to isolate it onto, so it aborts the whole
    call.

    Args:
        payload: The text, as a `TextInput` or the dict it validates from.
        concurrency: The ceiling on classification LLM calls in flight at once.
        classifier_model: The runnable that classifies one statement. `None`
            builds one from the environment.
        segmenter_model: The runnable that splits the text. `None` builds one
            from the environment.

    Returns:
        The statements the text was split into, in `classify`'s output shape:
        each carries the sentences around it as its surrounding context, and a
        classification or an error.

    Raises:
        ClassifierError: The input is malformed, the concurrency is below one,
            segmentation failed, or the credential is missing or rejected.
            Nothing partial is returned.
    """
    text_input = _coerce_text_input(payload)

    if concurrency < 1:
        raise ClassifierError(
            ErrorCode.INVALID_INPUT, f"concurrency must be >= 1, got {concurrency}"
        )

    if classifier_model is None or segmenter_model is None:
        config = load_config()
        if segmenter_model is None:
            segmenter_model = build_segmenter_model(config)
        if classifier_model is None:
            classifier_model = build_classifier_model(config)

    statement_texts = await segment_text(text_input.text, segmenter_model)
    windows = windows_for(text_input.text, statement_texts)

    statements = [
        Statement(surrounding_context=window, statement=statement)
        for statement, window in zip(statement_texts, windows, strict=True)
    ]
    results = await _classify_batch(statements, classifier_model, concurrency)

    return ClassifierOutput(statements=results)


def classify_statements_sync(
    payload: ClassifierInput | dict[str, Any],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: StructuredClassifierModel | None = None,
) -> ClassifierOutput:
    """Classify a batch from a caller not already in an async context.

    Args:
        payload: The batch, as a `ClassifierInput` or the dict it validates from.
        concurrency: The ceiling on LLM calls in flight at once.
        model: The runnable to call. `None` builds one from the environment.

    Returns:
        Whatever `classify_statements` returns for the same arguments.
    """
    return asyncio.run(
        classify_statements(payload, concurrency=concurrency, model=model)
    )


def classify_text_sync(
    payload: TextInput | dict[str, Any],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    classifier_model: StructuredClassifierModel | None = None,
    segmenter_model: StructuredSegmenterModel | None = None,
) -> ClassifierOutput:
    """Classify a text from a caller not already in an async context.

    Args:
        payload: The text, as a `TextInput` or the dict it validates from.
        concurrency: The ceiling on classification LLM calls in flight at once.
        classifier_model: The runnable that classifies one statement. `None`
            builds one from the environment.
        segmenter_model: The runnable that splits the text. `None` builds one
            from the environment.

    Returns:
        Whatever `classify_text` returns for the same arguments.
    """
    return asyncio.run(
        classify_text(
            payload,
            concurrency=concurrency,
            classifier_model=classifier_model,
            segmenter_model=segmenter_model,
        )
    )
