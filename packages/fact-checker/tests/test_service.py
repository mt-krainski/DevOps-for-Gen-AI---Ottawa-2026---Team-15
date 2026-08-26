"""The batch: what each statement comes back as, in what order, and at what cost."""

import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from fact_checker import service
from fact_checker.cache import RunCache
from fact_checker.config import CheckerConfig
from fact_checker.errors import (
    AuthenticationFailure,
    CheckError,
    ErrorCode,
    StatementFailure,
)
from fact_checker.models import CheckerInput, CheckerOutput
from fact_checker.service import CheckingRuntime, check_statements
from fact_checker.tools import Toolkit
from tests.conftest import (
    A_MODEL,
    FakeToolkit,
    Plan,
    Script,
    a_payload,
    a_script,
    a_statement,
    make_config,
)


def a_runtime(
    script: Script,
    *,
    searches: int = 0,
    tool_answers: Sequence[str | BaseException] = (),
) -> CheckingRuntime:
    """Build the injected seam: a toolkit answering from a queue, and the fakes."""
    toolkit = FakeToolkit(list(tool_answers))
    toolkit.searches = searches
    return CheckingRuntime(
        toolkit=toolkit,
        checking_model=script.checking_model,
        ruling_model=script.ruling_model,
    )


async def run_batch(
    payload: Mapping[str, object] | CheckerInput,
    script: Script,
    *,
    searches: int = 0,
    concurrency: int = 8,
    statement_timeout_seconds: int = 240,
    tool_answers: Sequence[str | BaseException] = (),
) -> CheckerOutput:
    """Check a batch through the fakes, naming only what the test varies."""
    return await check_statements(
        payload,
        config=make_config(
            concurrency=concurrency,
            statement_timeout_seconds=statement_timeout_seconds,
        ),
        runtime=a_runtime(script, searches=searches, tool_answers=tool_answers),
    )


async def test_an_opinion_passes_through_and_a_fact_is_ruled_on() -> None:
    """The two published shapes, side by side in one batch."""
    payload = a_payload(
        a_statement("Water boils at 100C at one atmosphere."),
        a_statement("Boiling water is the best way to cook pasta.", kind="opinion"),
    )

    output = await run_batch(
        payload, a_script("Water boils at 100C at one atmosphere.")
    )

    fact, opinion = output.statements
    assert fact.ruling is not None
    assert fact.ruling.verdict == "supported"
    assert fact.error is None
    assert opinion.ruling is None
    assert opinion.error is None


async def test_a_fact_at_low_confidence_is_still_checked() -> None:
    """A doubtful label is where a check is most informative, so it runs."""
    payload = a_payload(a_statement("The bridge opened in 1937.", confidence=0.05))

    output = await run_batch(payload, a_script("The bridge opened in 1937."))

    assert output.statements[0].ruling is not None


async def test_an_opinion_at_high_confidence_is_still_skipped() -> None:
    """Confidence never decides whether the agent runs; the class does."""
    payload = a_payload(
        a_statement("The bridge is beautiful.", kind="opinion", confidence=1.0)
    )

    output = await run_batch(payload, a_script())

    assert output.statements[0].ruling is None
    assert output.meta.counts.skipped == 1


async def test_the_output_keeps_input_order_whatever_order_the_runs_settle_in() -> None:
    """The second statement finishes first, and the output still reads first-first."""
    slow, quick = "The first claim.", "The second claim."
    payload = a_payload(a_statement(slow), a_statement(quick))
    script = a_script(slow, quick, **{slow: Plan(yields=4)})

    output = await run_batch(payload, script)

    assert script.finished == [quick, slow]
    assert [entry.statement for entry in output.statements] == [slow, quick]


async def test_identifiers_are_assigned_and_supplied_ones_are_kept() -> None:
    """A supplied id passes through; an absent one is named by input position."""
    payload = a_payload(
        a_statement("The first claim."),
        a_statement("The second claim.", identifier="given"),
        a_statement("The third claim."),
    )

    output = await run_batch(
        payload, a_script("The first claim.", "The second claim.", "The third claim.")
    )

    assert [entry.id for entry in output.statements] == ["s1", "given", "s3"]


async def test_a_duplicate_identifier_ends_the_run() -> None:
    """An ambiguous output is worse than no output, so nothing is checked."""
    payload = a_payload(
        a_statement("The first claim.", identifier="same"),
        a_statement("The second claim.", identifier="same"),
    )

    with pytest.raises(CheckError) as raised:
        await run_batch(payload, a_script())

    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert "same" in raised.value.message


async def test_an_empty_batch_returns_a_payload_with_zero_counts() -> None:
    """Nothing to check is a successful run that checked nothing."""
    output = await run_batch(a_payload(), a_script())

    assert output.statements == []
    assert output.meta.counts.total == 0
    assert output.meta.counts.checked == 0
    assert output.meta.usage.prompt_tokens == 0


async def test_a_payload_that_fails_the_contract_is_rejected() -> None:
    """A third classification label is a deliberate change, not a pass-through."""
    payload = a_payload(a_statement("The claim.", kind="speculation"))

    with pytest.raises(CheckError) as raised:
        await run_batch(payload, a_script())

    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert "speculation" in raised.value.message


async def test_an_already_validated_input_is_accepted_as_it_stands() -> None:
    """The entry point takes the model as well as the mapping it validates from."""
    payload = CheckerInput.model_validate(a_payload(a_statement("The claim.")))

    output = await run_batch(payload, a_script("The claim."))

    assert output.statements[0].ruling is not None


async def test_a_statement_failure_is_isolated_onto_its_own_entry() -> None:
    """Forty-nine rulings and one failure is a successful run."""
    failed, ruled = "The failing claim.", "The ruled claim."
    payload = a_payload(a_statement(failed), a_statement(ruled))
    script = a_script(
        failed,
        ruled,
        **{
            failed: Plan(
                failure=StatementFailure(ErrorCode.TOOL_ERROR, "the server said no")
            )
        },
    )

    output = await run_batch(payload, script)

    first, second = output.statements
    assert first.ruling is None
    assert first.error is not None
    assert first.error.code is ErrorCode.TOOL_ERROR
    assert first.error.message == "the server said no"
    assert second.ruling is not None


async def test_an_authentication_failure_ends_the_run_with_nothing_partial() -> None:
    """A rejected credential fails every statement alike, so one error says it."""
    rejected, ruled = "The rejected claim.", "The ruled claim."
    payload = a_payload(a_statement(rejected), a_statement(ruled))
    script = a_script(
        rejected,
        ruled,
        **{
            rejected: Plan(failure=AuthenticationFailure("the token was rejected")),
            ruled: Plan(yields=8),
        },
    )

    with pytest.raises(CheckError) as raised:
        await run_batch(payload, script)

    assert raised.value.code is ErrorCode.AUTH_ERROR
    assert "the token was rejected" in raised.value.message
    assert script.finished == []


async def test_a_statement_that_hangs_times_out_while_the_rest_complete() -> None:
    """The timeout catches a hang, and a hang costs one entry, not the batch."""
    hanging, ruled = "The hanging claim.", "The ruled claim."
    payload = a_payload(a_statement(hanging), a_statement(ruled))
    script = a_script(hanging, ruled, **{hanging: Plan(hangs=True)})

    output = await run_batch(payload, script, statement_timeout_seconds=1)

    first, second = output.statements
    assert first.ruling is None
    assert first.error is not None
    assert first.error.code is ErrorCode.TIMEOUT
    assert "1 seconds" in first.error.message
    assert second.ruling is not None


async def test_no_more_checks_run_at_once_than_the_bound_allows() -> None:
    """More statements than permits, and the permits are what decides."""
    texts = [f"The claim numbered {number}." for number in range(9)]
    payload = a_payload(*(a_statement(text) for text in texts))
    script = a_script(*texts, **{text: Plan(yields=2) for text in texts})

    output = await run_batch(payload, script, concurrency=3)

    assert script.peak == 3
    assert len(output.statements) == 9


async def test_the_counts_report_every_outcome_in_the_batch() -> None:
    """Total, checked, skipped and failed, each from the entries themselves."""
    ruled, also_ruled = "The first claim.", "The second claim."
    failed = "The failing claim."
    payload = a_payload(
        a_statement(ruled),
        a_statement("An aesthetic judgement.", kind="opinion"),
        a_statement(failed),
        a_statement(also_ruled),
    )
    script = a_script(
        ruled,
        also_ruled,
        failed,
        **{failed: Plan(failure=StatementFailure(ErrorCode.AGENT_ERROR, "no"))},
    )

    counts = (await run_batch(payload, script)).meta.counts

    assert counts.total == 4
    assert counts.checked == 2
    assert counts.skipped == 1
    assert counts.failed == 1


async def test_the_usage_sums_the_batch_and_takes_searches_from_the_tools() -> None:
    """Tokens are summed across the batch; searches are the toolkit's own count."""
    first, second = "The first claim.", "The second claim."
    payload = a_payload(a_statement(first), a_statement(second))
    script = a_script(
        first,
        second,
        **{
            first: Plan(checking_tokens=(100, 10), ruling_tokens=(200, 20)),
            second: Plan(checking_tokens=(1, 2), ruling_tokens=(3, 4)),
        },
    )

    usage = (await run_batch(payload, script, searches=7)).meta.usage

    assert usage.prompt_tokens == 304
    assert usage.completion_tokens == 36
    assert usage.searches == 7


async def test_the_run_is_stamped_in_the_published_timestamp_format() -> None:
    """The wire format is to the second, in UTC, with a trailing Z."""
    moments = iter(
        [
            datetime(2026, 8, 22, 14, 3, 11, tzinfo=UTC),
            datetime(2026, 8, 22, 14, 5, 47, tzinfo=UTC),
        ]
    )
    payload = a_payload(a_statement("The claim."))

    output = await check_statements(
        payload,
        config=make_config(),
        runtime=a_runtime(a_script("The claim.")),
        now=lambda: next(moments),
    )

    assert output.meta.started_at == "2026-08-22T14:03:11Z"
    assert output.meta.finished_at == "2026-08-22T14:05:47Z"


async def test_the_run_finishes_no_earlier_than_it_started() -> None:
    """Read from the real clock, the pair still reads in order."""
    payload = a_payload(a_statement("The claim."))

    meta = (await run_batch(payload, a_script("The claim."))).meta

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta.started_at)
    assert meta.finished_at >= meta.started_at


async def test_the_run_reports_the_configured_model() -> None:
    """The meta names the model that did the work, not a default."""
    output = await run_batch(a_payload(), a_script())

    assert output.meta.model == A_MODEL


async def test_the_run_opens_its_own_toolkit_when_no_runtime_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an injected seam the service connects, builds, and closes."""
    script = a_script("The claim.")
    toolkit = FakeToolkit([])
    opened: list[tuple[CheckerConfig, RunCache]] = []
    closed: list[Toolkit] = []

    @asynccontextmanager
    async def open_one(
        config: CheckerConfig, cache: RunCache
    ) -> AsyncIterator[Toolkit]:
        opened.append((config, cache))
        try:
            yield toolkit
        finally:
            closed.append(toolkit)

    def build_two(
        config: CheckerConfig, given: Toolkit
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        assert given is toolkit
        return script.checking_model, script.ruling_model

    monkeypatch.setattr(service, "open_toolkit", open_one)
    monkeypatch.setattr(service, "build_models", build_two)
    config = make_config()

    output = await check_statements(a_payload(a_statement("The claim.")), config=config)

    assert output.statements[0].ruling is not None
    assert [held for held, _ in opened] == [config]
    assert closed == [toolkit]


async def test_the_searches_are_read_while_the_toolkit_is_still_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A toolkit that forgets its count on close still reports what it counted."""
    script = a_script("The claim.")
    toolkit = FakeToolkit([])
    toolkit.searches = 4

    @asynccontextmanager
    async def open_one(
        _config: CheckerConfig, _cache: RunCache
    ) -> AsyncIterator[Toolkit]:
        try:
            yield toolkit
        finally:
            toolkit.searches = 0

    monkeypatch.setattr(service, "open_toolkit", open_one)
    monkeypatch.setattr(
        service,
        "build_models",
        lambda _config, _toolkit: (script.checking_model, script.ruling_model),
    )

    output = await check_statements(
        a_payload(a_statement("The claim.")), config=make_config()
    )

    assert output.meta.usage.searches == 4


async def test_one_cache_serves_the_whole_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second statement's search reuses the first's, so there is one cache."""
    texts = ["The first claim.", "The second claim."]
    script = a_script(*texts)
    caches: list[RunCache] = []

    @asynccontextmanager
    async def open_one(
        _config: CheckerConfig, cache: RunCache
    ) -> AsyncIterator[Toolkit]:
        caches.append(cache)
        yield FakeToolkit([])

    monkeypatch.setattr(service, "open_toolkit", open_one)
    monkeypatch.setattr(
        service,
        "build_models",
        lambda _config, _toolkit: (script.checking_model, script.ruling_model),
    )

    await check_statements(
        a_payload(*(a_statement(text) for text in texts)), config=make_config()
    )

    assert len(caches) == 1
    assert isinstance(caches[0], RunCache)


async def test_the_configuration_is_read_from_the_environment_when_none_is_passed(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A caller that passes no configuration gets the environment's."""
    clean_env.setenv("OPENROUTER_API_KEY", "sk-from-the-environment")
    clean_env.setenv("BRIGHTDATA_API_TOKEN", "bd-from-the-environment")
    clean_env.setenv("OPENROUTER_MODEL", "a-vendor/a-model")

    output = await check_statements(a_payload(), runtime=a_runtime(a_script()))

    assert output.meta.model == "a-vendor/a-model"


def info_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the INFO lines the run wrote, in the order it wrote them."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]


async def test_every_statement_gets_one_info_line_naming_its_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ruling, a skip and a failure each read as one line at the default level."""
    caplog.set_level(logging.INFO, logger="fact_checker")
    ruled, failed = "The ruled claim.", "The failing claim."
    payload = a_payload(
        a_statement(ruled),
        a_statement("An aesthetic judgement.", kind="opinion"),
        a_statement(failed),
    )
    script = a_script(
        ruled,
        failed,
        **{
            ruled: Plan(verdict="refuted"),
            failed: Plan(failure=StatementFailure(ErrorCode.TOOL_ERROR, "no")),
        },
    )

    await run_batch(payload, script)

    lines = info_lines(caplog)
    assert len(lines) == 3
    assert any(line.startswith("s1: refuted in ") for line in lines)
    assert any(line.startswith("s2: skipped in ") for line in lines)
    assert any(line.startswith("s3: failed (TOOL_ERROR) in ") for line in lines)


async def test_the_info_line_reports_the_tool_calls_the_check_spent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The line carries what the statement cost in calls, not only its verdict."""
    caplog.set_level(logging.INFO, logger="fact_checker")
    text = "The claim."
    payload = a_payload(a_statement(text))
    script = a_script(text, **{text: Plan(tool_calls=2)})

    await run_batch(payload, script, tool_answers=["a result", "another result"])

    assert info_lines(caplog)[0].endswith("2 tool calls")


async def test_a_failed_check_logs_the_tool_calls_it_had_already_spent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A statement that spends calls and then fails reports what it spent."""
    caplog.set_level(logging.INFO, logger="fact_checker")
    text = "The claim."
    payload = a_payload(a_statement(text))
    script = a_script(text, **{text: Plan(tool_calls=2)})

    await run_batch(
        payload,
        script,
        tool_answers=["a result", StatementFailure(ErrorCode.TOOL_ERROR, "no")],
    )

    line = info_lines(caplog)[0]
    assert line.startswith("s1: failed (TOOL_ERROR) in ")
    assert line.endswith("1 tool calls")


async def test_a_timed_out_check_never_claims_a_tool_call_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cancelled check leaves no count behind, so the line says it does not know."""
    caplog.set_level(logging.INFO, logger="fact_checker")
    text = "The hanging claim."
    payload = a_payload(a_statement(text))
    script = a_script(text, **{text: Plan(hangs=True)})

    await run_batch(payload, script, statement_timeout_seconds=1)

    line = info_lines(caplog)[0]
    assert line.startswith("s1: failed (TIMEOUT) in ")
    assert line.endswith("an unknown number of tool calls")
