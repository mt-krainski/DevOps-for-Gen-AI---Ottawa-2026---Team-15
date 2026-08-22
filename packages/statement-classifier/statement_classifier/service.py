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
)


def _coerce_input(payload: ClassifierInput | dict[str, Any]) -> ClassifierInput:
    if isinstance(payload, ClassifierInput):
        return payload
    try:
        return ClassifierInput.model_validate(payload)
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

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.ensure_future(_classify_with_isolation(statement, model, semaphore))
        for statement in classifier_input.statements
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

    return ClassifierOutput(statements=[task.result() for task in tasks])


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
