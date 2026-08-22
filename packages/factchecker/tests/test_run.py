"""Tests for the checker protocol and the orchestrator in `factchecker.run`."""

import asyncio
import logging
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

import pytest

from factchecker.checker import CheckOutcome, OfflineChecker
from factchecker.errors import AuthenticationFailed, InputValidationError
from factchecker.models import (
    IdentifiedStatement,
    InputPayload,
    Ruling,
    StatementClass,
    Verdict,
)
from factchecker.run import RunSettings, run_check

STARTED_AT = datetime(2026, 8, 22, 14, 3, 11, tzinfo=UTC)
FINISHED_AT = datetime(2026, 8, 22, 14, 5, 47, tzinfo=UTC)


def _payload(*classes: StatementClass) -> InputPayload:
    """A payload with one statement per class named, identified s1, s2, and so on."""
    return InputPayload.model_validate(
        {
            "statements": [
                {
                    "surroundingContext": f"The paragraph around claim {position}.",
                    "statement": f"Claim {position}",
                    "classification": {"class": class_, "confidence": 0.7},
                }
                for position, class_ in enumerate(classes, start=1)
            ]
        }
    )


def _statement() -> IdentifiedStatement:
    """One identified statement, for a checker called directly."""
    return IdentifiedStatement(
        id="s1",
        surrounding_context="The paragraph around the claim.",
        statement="Water boils at 100 C",
        classification={"class": "fact", "confidence": 0.7},
    )


def _outcome(verdict: Verdict = "supported", searches: int = 3) -> CheckOutcome:
    """A checker result with usage the test can add up."""
    return CheckOutcome(
        ruling=Ruling(
            verdict=verdict,
            confidence=0.9,
            justification="The sources agree [1].",
            references=[],
        ),
        prompt_tokens=100,
        completion_tokens=20,
        searches=searches,
    )


def _clock(*times: datetime) -> Callable[[], datetime]:
    """A now callable returning each time in turn, then repeating the last."""
    remaining = list(times)

    def now() -> datetime:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return now


def _settings(
    model: str = "offline",
    concurrency: int = 8,
    statement_timeout_seconds: float = 5.0,
) -> RunSettings:
    """Run settings for a test, bounded so that no test waits on the real limit."""
    return RunSettings(
        model=model,
        concurrency=concurrency,
        statement_timeout_seconds=statement_timeout_seconds,
    )


class _ScriptedChecker:
    """Returns or raises whatever the script names for each statement identifier."""

    def __init__(self, script: Mapping[str, CheckOutcome | Exception]) -> None:
        self.script = script
        self.seen: list[str] = []

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Record the call, then act out this statement's script."""
        self.seen.append(statement.id)
        scripted = self.script[statement.id]
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


class _HangingChecker:
    """Waits for an event no test ever sets, and records its own cancellation."""

    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Wait forever, so only a cancellation ends this call."""
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(statement.id)
            raise
        raise AssertionError("the event is never set")


class _CredentialRejectingChecker:
    """Rejects one statement's credential and waits forever on every other."""

    def __init__(self, rejected_id: str) -> None:
        self.rejected_id = rejected_id
        self.cancelled: list[str] = []

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Reject the named statement, and hang on the rest until cancelled."""
        if statement.id == self.rejected_id:
            raise AuthenticationFailed("openrouter rejected the key")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(statement.id)
            raise
        raise AssertionError("the event is never set")


class _DelayedChecker:
    """Finishes each statement after its own delay, and records the finishing order."""

    def __init__(self, delays: Mapping[str, float]) -> None:
        self.delays = delays
        self.finished: list[str] = []

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Sleep this statement's delay, so the run completes out of input order."""
        await asyncio.sleep(self.delays[statement.id])
        self.finished.append(statement.id)
        return _outcome()


class _ConcurrencyProbe:
    """Records the highest number of checks that were ever running at once."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Hold a slot across several loop turns, so overlap has a chance to show."""
        self.running += 1
        self.peak = max(self.peak, self.running)
        # Yielding without sleeping lets every other ready check enter, so an
        # unbounded run would drive `running` to the number of statements.
        for _ in range(5):
            await asyncio.sleep(0)
        self.running -= 1
        return _outcome()


def test_run_settings_carry_the_documented_defaults() -> None:
    """The bound is eight checks and the per-statement limit is 240 seconds."""
    settings = RunSettings(model="offline")

    assert settings.concurrency == 8
    assert settings.statement_timeout_seconds == 240.0


def test_the_offline_checker_rules_unverifiable_without_searching() -> None:
    """The stand-in returns a populated ruling that says no search ran."""
    outcome = asyncio.run(OfflineChecker().check(_statement()))

    assert outcome.ruling.verdict == "unverifiable"
    assert outcome.ruling.confidence == 0.0
    assert outcome.ruling.references == []
    assert "no search" in outcome.ruling.justification.lower()
    assert "no checking agent is configured" in outcome.ruling.justification.lower()
    assert (outcome.prompt_tokens, outcome.completion_tokens, outcome.searches) == (
        0,
        0,
        0,
    )


def test_an_opinion_statement_never_reaches_the_checker() -> None:
    """An opinion is passed through with a null ruling and a null error."""
    checker = _ScriptedChecker({})

    result = asyncio.run(
        run_check(
            _payload("opinion"),
            checker,
            _settings(),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    assert checker.seen == []
    assert result.statements[0].ruling is None
    assert result.statements[0].error is None


def test_a_returned_outcome_becomes_the_entrys_ruling() -> None:
    """A fact reaches the checker once, and the ruling it returns is reported."""
    checker = _ScriptedChecker({"s1": _outcome(verdict="refuted")})

    result = asyncio.run(
        run_check(
            _payload("fact"),
            checker,
            _settings(),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    assert checker.seen == ["s1"]
    entry = result.statements[0]
    assert entry.ruling is not None
    assert entry.ruling.verdict == "refuted"
    assert entry.error is None


def test_a_check_that_exceeds_the_timeout_reports_an_error_and_no_ruling() -> None:
    """The run was cut off, so it reports no finding for that statement."""
    checker = _HangingChecker()

    result = asyncio.run(
        run_check(
            _payload("fact"),
            checker,
            _settings(statement_timeout_seconds=0.05),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    entry = result.statements[0]
    assert entry.ruling is None
    assert entry.error is not None
    assert entry.error.kind == "timeout"
    assert "0.05" in entry.error.message


def test_a_check_that_exceeds_the_timeout_is_cancelled() -> None:
    """A checker left running after its statement was recorded is a leak."""
    checker = _HangingChecker()

    async def drive() -> list[str]:
        await run_check(
            _payload("fact"),
            checker,
            _settings(statement_timeout_seconds=0.05),
            _clock(STARTED_AT, FINISHED_AT),
        )
        # Read the record inside the loop: closing the loop cancels whatever is
        # still pending, which would hide a checker the timeout merely abandoned.
        return list(checker.cancelled)

    assert asyncio.run(drive()) == ["s1"]


def test_a_checker_exception_becomes_an_error_and_the_run_continues() -> None:
    """One statement's failure is reported, and the rest of the run still rules."""
    checker = _ScriptedChecker(
        {"s1": RuntimeError("the upstream service said no"), "s2": _outcome()}
    )

    result = asyncio.run(
        run_check(
            _payload("fact", "fact"),
            checker,
            _settings(),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    failed, checked = result.statements
    assert failed.ruling is None
    assert failed.error is not None
    assert failed.error.kind == "check_failed"
    assert "the upstream service said no" in failed.error.message
    assert checked.ruling is not None
    assert checked.error is None


def test_authentication_failure_ends_the_run_without_waiting_for_the_rest() -> None:
    """A rejected credential fails every statement, so it is reported once at once."""
    checker = _CredentialRejectingChecker(rejected_id="s2")

    async def drive() -> list[str]:
        with pytest.raises(AuthenticationFailed, match="openrouter rejected the key"):
            await asyncio.wait_for(
                run_check(
                    _payload("fact", "fact"),
                    checker,
                    _settings(),
                    _clock(STARTED_AT, FINISHED_AT),
                ),
                timeout=5.0,
            )
        # Read the record inside the loop: closing the loop cancels whatever is
        # still pending, which would hide a run that left the other check going.
        return list(checker.cancelled)

    assert asyncio.run(drive()) == ["s1"]


def test_output_order_follows_input_order_when_checks_finish_out_of_order() -> None:
    """Whatever order the checks settle in, the entries stay in input order."""
    checker = _DelayedChecker({"s1": 0.03, "s2": 0.02, "s3": 0.01})

    result = asyncio.run(
        run_check(
            _payload("fact", "fact", "fact"),
            checker,
            _settings(),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    assert checker.finished == ["s3", "s2", "s1"]
    assert [entry.id for entry in result.statements] == ["s1", "s2", "s3"]


def test_the_concurrency_bound_is_never_exceeded() -> None:
    """The semaphore holds: the observed peak is the bound, not the statement count."""
    probe = _ConcurrencyProbe()

    result = asyncio.run(
        run_check(
            _payload("fact", "fact", "fact", "fact", "fact", "fact"),
            probe,
            _settings(concurrency=2),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    assert len(result.statements) == 6
    assert probe.peak == 2


def test_each_statement_writes_one_info_record_naming_its_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run log names each statement, its elapsed time, and what became of it."""
    checker = _ScriptedChecker(
        {"s1": _outcome(searches=4), "s3": RuntimeError("the upstream service said no")}
    )

    with caplog.at_level(logging.INFO, logger="factchecker.run"):
        asyncio.run(
            run_check(
                _payload("fact", "opinion", "fact"),
                checker,
                _settings(),
                _clock(STARTED_AT, FINISHED_AT),
            )
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "factchecker.run" and record.levelno == logging.INFO
    ]
    assert len(messages) == 3
    ruled = next(message for message in messages if message.startswith("s1 "))
    skipped = next(message for message in messages if message.startswith("s2 "))
    failed = next(message for message in messages if message.startswith("s3 "))
    assert "supported" in ruled
    assert "4 searches" in ruled
    assert "skipped" in skipped
    assert "check_failed" in failed


def test_elapsed_time_comes_from_a_monotonic_clock_not_the_injected_now(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A constant now still leaves a real duration in the record."""
    checker = _DelayedChecker({"s1": 0.02})
    constant = _clock(STARTED_AT)

    with caplog.at_level(logging.INFO, logger="factchecker.run"):
        result = asyncio.run(
            run_check(_payload("fact"), checker, _settings(), constant)
        )

    assert result.meta.started_at == result.meta.finished_at == STARTED_AT
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == "factchecker.run"
    )
    elapsed = re.search(r"in (\d+\.\d+)s", message)
    assert elapsed is not None
    assert float(elapsed.group(1)) > 0.0


def test_counts_and_usage_report_the_whole_run() -> None:
    """The three buckets are disjoint, they sum to the total, and usage adds up."""
    checker = _ScriptedChecker(
        {
            "s1": _outcome(searches=3),
            "s3": RuntimeError("the upstream service said no"),
            "s4": _outcome(searches=5),
        }
    )

    result = asyncio.run(
        run_check(
            _payload("fact", "opinion", "fact", "fact"),
            checker,
            _settings(),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    counts = result.meta.counts
    assert (counts.total, counts.checked, counts.skipped, counts.failed) == (4, 2, 1, 1)
    assert counts.total == counts.checked + counts.skipped + counts.failed
    assert result.meta.usage.prompt_tokens == 200
    assert result.meta.usage.completion_tokens == 40
    assert result.meta.usage.searches == 8


def test_meta_records_the_model_and_the_injected_times() -> None:
    """The model is the setting, and both times come from the injected clock."""
    checker = _ScriptedChecker({"s1": _outcome()})

    result = asyncio.run(
        run_check(
            _payload("fact"),
            checker,
            _settings(model="anthropic/claude-sonnet-4"),
            _clock(STARTED_AT, FINISHED_AT),
        )
    )

    assert result.meta.model == "anthropic/claude-sonnet-4"
    assert result.meta.started_at == STARTED_AT
    assert result.meta.finished_at == FINISHED_AT


def test_a_repeated_identifier_is_rejected_before_any_check_runs() -> None:
    """`run_check` assigns identifiers, so the uniqueness rule reaches it too."""
    checker = _ScriptedChecker({})
    payload = InputPayload.model_validate(
        {
            "statements": [
                {
                    "id": "dup",
                    "surroundingContext": "context",
                    "statement": "one",
                    "classification": {"class": "fact", "confidence": 0.7},
                },
                {
                    "id": "dup",
                    "surroundingContext": "context",
                    "statement": "two",
                    "classification": {"class": "fact", "confidence": 0.7},
                },
            ]
        }
    )

    with pytest.raises(InputValidationError, match="dup"):
        asyncio.run(
            run_check(payload, checker, _settings(), _clock(STARTED_AT, FINISHED_AT))
        )

    assert checker.seen == []
