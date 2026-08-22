"""The orchestrator: one checker call per factual statement, bounded and timed."""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from factchecker.checker import CheckOutcome, StatementChecker
from factchecker.errors import AuthenticationFailed, CheckFailed
from factchecker.ingest import assign_ids
from factchecker.models import (
    CheckError,
    Counts,
    IdentifiedStatement,
    InputPayload,
    Meta,
    OutputPayload,
    OutputStatement,
    Ruling,
    Usage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSettings:
    """How one run is bounded, and what model it reports."""

    model: str
    concurrency: int = 8
    statement_timeout_seconds: float = 240.0


@dataclass(frozen=True)
class _StatementResult:
    """One statement's output entry, what it consumed, and how its log record reads."""

    entry: OutputStatement
    usage: Usage | None
    summary: str


async def run_check(
    payload: InputPayload,
    checker: StatementChecker,
    settings: RunSettings,
    now: Callable[[], datetime],
) -> OutputPayload:
    """Check every factual statement in the payload and assemble the output.

    Args:
        payload: The validated input payload.
        checker: The checker each factual statement is passed to.
        settings: The model name, the concurrency bound, and the per-statement limit.
        now: Returns the current time as a timezone-aware datetime.

    Returns:
        One entry per input statement in input order, under a meta record of the run.

    Raises:
        InputValidationError: Two statements share an identifier.
        AuthenticationFailed: A credential was rejected, which fails every statement
            alike, so the run ends rather than reporting it once per statement.
    """
    statements = assign_ids(payload)
    started_at = now()
    results = await _check_all(statements, checker, settings)
    finished_at = now()
    entries = [result.entry for result in results]
    return OutputPayload(
        meta=Meta(
            model=settings.model,
            started_at=started_at,
            finished_at=finished_at,
            counts=_count(entries),
            usage=_total_usage(results),
        ),
        statements=entries,
    )


async def _check_all(
    statements: Sequence[IdentifiedStatement],
    checker: StatementChecker,
    settings: RunSettings,
) -> list[_StatementResult]:
    """Run every statement concurrently, under a bound, and keep the input order."""
    semaphore = asyncio.Semaphore(settings.concurrency)
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(_check_one(statement, checker, settings, semaphore))
                for statement in statements
            ]
    except* AuthenticationFailed as rejections:
        raise rejections.exceptions[0] from None
    return [task.result() for task in tasks]


async def _check_one(
    statement: IdentifiedStatement,
    checker: StatementChecker,
    settings: RunSettings,
    semaphore: asyncio.Semaphore,
) -> _StatementResult:
    """Settle one statement, and write its INFO record."""
    started = time.monotonic()
    result = await _settle(statement, checker, settings, semaphore)
    logger.info(
        "%s in %.3fs: %s", statement.id, time.monotonic() - started, result.summary
    )
    return result


async def _settle(
    statement: IdentifiedStatement,
    checker: StatementChecker,
    settings: RunSettings,
    semaphore: asyncio.Semaphore,
) -> _StatementResult:
    """Pass a fact to the checker under the bound and the limit; skip an opinion."""
    if statement.classification.class_ == "opinion":
        return _StatementResult(entry=_entry(statement), usage=None, summary="skipped")

    async with semaphore:
        try:
            async with asyncio.timeout(settings.statement_timeout_seconds) as limit:
                outcome = await checker.check(statement)
            # `_ruled` reads the outcome apart here, inside the `try` rather than in
            # an `else`: nothing validates a `CheckOutcome` at runtime, so a malformed
            # one has to fail its own check instead of escaping the task group.
            return _ruled(statement, outcome)
        except TimeoutError as exc:
            if limit.expired():
                return _failure(
                    statement,
                    kind="timeout",
                    message=(
                        "the check exceeded the per-statement limit of "
                        f"{settings.statement_timeout_seconds} seconds"
                    ),
                )
            # `socket.timeout` and `asyncio.TimeoutError` are both `TimeoutError`, so
            # a checker's own read timeout arrives here without the limit expiring.
            return _failure(statement, kind="check_failed", message=str(exc))
        except AuthenticationFailed:
            raise
        except CheckFailed as named:
            return _failure(statement, kind=named.kind, message=named.message)
        except Exception as exc:  # noqa: BLE001 — a failed check must not end the run
            return _failure(statement, kind="check_failed", message=str(exc))


def _entry(
    statement: IdentifiedStatement,
    ruling: Ruling | None = None,
    error: CheckError | None = None,
) -> OutputStatement:
    """Repeat the input fields, and add the ruling and the error the run produced."""
    return OutputStatement(
        id=statement.id,
        surrounding_context=statement.surrounding_context,
        statement=statement.statement,
        classification=statement.classification,
        ruling=ruling,
        error=error,
    )


def _ruled(statement: IdentifiedStatement, outcome: CheckOutcome) -> _StatementResult:
    """Read a checker's outcome apart into the entry, the usage, and the log line."""
    return _StatementResult(
        entry=_entry(statement, ruling=outcome.ruling),
        usage=Usage(
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            searches=outcome.searches,
        ),
        summary=f"{outcome.ruling.verdict}, {outcome.searches} searches",
    )


def _failure(
    statement: IdentifiedStatement, kind: str, message: str
) -> _StatementResult:
    """A statement the run could not rule on: an error, and never a ruling."""
    return _StatementResult(
        entry=_entry(statement, error=CheckError(kind=kind, message=message)),
        usage=None,
        summary=f"failed: {kind}",
    )


def _count(entries: Sequence[OutputStatement]) -> Counts:
    """Sort the entries into the three disjoint buckets."""
    return Counts(
        total=len(entries),
        checked=sum(1 for entry in entries if entry.ruling is not None),
        skipped=sum(1 for entry in entries if entry.classification.class_ == "opinion"),
        failed=sum(1 for entry in entries if entry.error is not None),
    )


def _total_usage(results: Sequence[_StatementResult]) -> Usage:
    """Add up what the checks consumed."""
    consumed = [result.usage for result in results if result.usage is not None]
    return Usage(
        prompt_tokens=sum(usage.prompt_tokens for usage in consumed),
        completion_tokens=sum(usage.completion_tokens for usage in consumed),
        searches=sum(usage.searches for usage in consumed),
    )
