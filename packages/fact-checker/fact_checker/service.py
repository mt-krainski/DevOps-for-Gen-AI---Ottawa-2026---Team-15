"""The batch: identifiers, concurrency, ordering, and the run's own record."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from fact_checker.agent import CheckingModel, RulingModel, build_models, check_one
from fact_checker.cache import RunCache
from fact_checker.config import CheckerConfig, load_config
from fact_checker.errors import (
    AuthenticationFailure,
    CheckError,
    ErrorCode,
    StatementFailure,
)
from fact_checker.models import (
    CheckedStatement,
    CheckerInput,
    CheckerOutput,
    Counts,
    InputStatement,
    Ruling,
    RunMeta,
    StatementError,
    Usage,
    assign_identifiers,
    format_timestamp,
)
from fact_checker.tools import Toolkit, open_toolkit

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class CheckingRuntime:
    """What one run checks with: the connected tools, and the two models."""

    toolkit: Toolkit
    checking_model: CheckingModel
    ruling_model: RulingModel


@dataclass(frozen=True)
class _Outcome:
    """One statement's published entry, and what its model calls cost."""

    entry: CheckedStatement
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def check_statements(
    payload: CheckerInput | Mapping[str, object],
    *,
    config: CheckerConfig | None = None,
    runtime: CheckingRuntime | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> CheckerOutput:
    """Check every factual statement in a batch, and report the whole batch.

    Args:
        payload: The batch, as a `CheckerInput` or the mapping it validates from.
        config: The run's settings. `None` reads them from the environment.
        runtime: The tools and models to check with. `None` connects to Bright
            Data and builds the models from `config`.
        now: The clock the run's start and finish times are read from.

    Returns:
        One entry for every input statement, in input order, beside the run's
        model, times, counts and usage.

    Raises:
        CheckError: The payload failed the contract, two statements share an
            identifier, or a credential was rejected part-way through.
    """
    checker_input = _validated(payload)
    identifiers = assign_identifiers(checker_input.statements)
    settings = config if config is not None else load_config()

    async with _opened(settings, runtime) as opened:
        started_at = now()
        outcomes = await _check_all(
            checker_input.statements, identifiers, settings, opened
        )
        finished_at = now()
        return _assembled(
            outcomes,
            config=settings,
            toolkit=opened.toolkit,
            started_at=started_at,
            finished_at=finished_at,
        )


@asynccontextmanager
async def _opened(
    config: CheckerConfig, runtime: CheckingRuntime | None
) -> AsyncIterator[CheckingRuntime]:
    """Yield the injected runtime, or connect and build the one this run needs."""
    if runtime is not None:
        yield runtime
        return
    cache = RunCache()
    async with open_toolkit(config, cache) as toolkit:
        checking_model, ruling_model = build_models(config, toolkit)
        yield CheckingRuntime(
            toolkit=toolkit, checking_model=checking_model, ruling_model=ruling_model
        )


def _validated(payload: CheckerInput | Mapping[str, object]) -> CheckerInput:
    if isinstance(payload, CheckerInput):
        return payload
    try:
        return CheckerInput.model_validate(payload)
    except ValidationError as exc:
        raise CheckError(ErrorCode.INVALID_INPUT, str(exc)) from exc


async def _check_all(
    statements: Sequence[InputStatement],
    identifiers: Sequence[str],
    config: CheckerConfig,
    runtime: CheckingRuntime,
) -> list[_Outcome]:
    semaphore = asyncio.Semaphore(config.concurrency)
    tasks = [
        asyncio.create_task(
            _outcome_for(
                statement,
                identifier,
                config=config,
                runtime=runtime,
                semaphore=semaphore,
            )
        )
        for statement, identifier in zip(statements, identifiers, strict=True)
    ]
    try:
        return await asyncio.gather(*tasks)
    except AuthenticationFailure as exc:
        raise CheckError(ErrorCode.AUTH_ERROR, exc.message) from exc
    finally:
        await _stop_the_rest(tasks)


async def _stop_the_rest(tasks: Sequence[asyncio.Task[_Outcome]]) -> None:
    """Cancel every check still running, and wait for each to finish stopping."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _outcome_for(
    statement: InputStatement,
    identifier: str,
    *,
    config: CheckerConfig,
    runtime: CheckingRuntime,
    semaphore: asyncio.Semaphore,
) -> _Outcome:
    if statement.classification.class_ == "opinion":
        _log_outcome(identifier, "skipped", elapsed=0.0, tool_calls=0)
        return _Outcome(entry=_entry(statement, identifier))
    async with semaphore:
        return await _checked(statement, identifier, config=config, runtime=runtime)


async def _checked(
    statement: InputStatement,
    identifier: str,
    *,
    config: CheckerConfig,
    runtime: CheckingRuntime,
) -> _Outcome:
    started = time.perf_counter()
    try:
        run = await asyncio.wait_for(
            check_one(
                statement,
                identifier,
                toolkit=runtime.toolkit,
                checking_model=runtime.checking_model,
                ruling_model=runtime.ruling_model,
                budget=config.tool_call_budget,
            ),
            config.statement_timeout_seconds,
        )
    except (TimeoutError, StatementFailure) as failure:
        error = _error_for(failure, config.statement_timeout_seconds)
        _log_outcome(
            identifier,
            f"failed ({error.code})",
            elapsed=time.perf_counter() - started,
            tool_calls=_calls_spent(failure),
        )
        return _Outcome(entry=_entry(statement, identifier, error=error))

    _log_outcome(
        identifier,
        run.ruling.verdict,
        elapsed=time.perf_counter() - started,
        tool_calls=run.tool_calls_used,
    )
    return _Outcome(
        entry=_entry(statement, identifier, ruling=run.ruling),
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
    )


def _error_for(
    failure: TimeoutError | StatementFailure, timeout_seconds: int
) -> StatementError:
    if isinstance(failure, StatementFailure):
        return StatementError(code=failure.code, message=failure.message)
    return StatementError(
        code=ErrorCode.TIMEOUT,
        message=f"the check did not finish inside {timeout_seconds} seconds",
    )


def _calls_spent(failure: TimeoutError | StatementFailure) -> int | None:
    """Return the tool calls the failed check spent, or `None` where none is known.

    A timeout cancels the check part-way, which takes the running count with it.
    """
    if isinstance(failure, StatementFailure):
        return failure.tool_calls_used
    return None


def _entry(
    statement: InputStatement,
    identifier: str,
    *,
    ruling: Ruling | None = None,
    error: StatementError | None = None,
) -> CheckedStatement:
    return CheckedStatement(
        id=identifier,
        surrounding_context=statement.surrounding_context,
        statement=statement.statement,
        classification=statement.classification,
        ruling=ruling,
        error=error,
    )


def _log_outcome(
    identifier: str, outcome: str, *, elapsed: float, tool_calls: int | None
) -> None:
    spent = "an unknown number of" if tool_calls is None else tool_calls
    logger.info("%s: %s in %.2fs, %s tool calls", identifier, outcome, elapsed, spent)


def _assembled(
    outcomes: Sequence[_Outcome],
    *,
    config: CheckerConfig,
    toolkit: Toolkit,
    started_at: datetime,
    finished_at: datetime,
) -> CheckerOutput:
    entries = [outcome.entry for outcome in outcomes]
    return CheckerOutput(
        meta=RunMeta(
            model=config.model,
            started_at=format_timestamp(started_at),
            finished_at=format_timestamp(finished_at),
            counts=_counts(entries),
            usage=Usage(
                prompt_tokens=sum(outcome.prompt_tokens for outcome in outcomes),
                completion_tokens=sum(
                    outcome.completion_tokens for outcome in outcomes
                ),
                searches=toolkit.searches,
            ),
        ),
        statements=entries,
    )


def _counts(entries: Sequence[CheckedStatement]) -> Counts:
    return Counts(
        total=len(entries),
        checked=sum(1 for entry in entries if entry.ruling is not None),
        skipped=sum(1 for entry in entries if entry.classification.class_ == "opinion"),
        failed=sum(1 for entry in entries if entry.error is not None),
    )
