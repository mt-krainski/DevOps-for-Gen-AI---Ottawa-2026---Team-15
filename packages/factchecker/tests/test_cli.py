"""Tests for the logging setup and the command-line front end.

No test here opens a connection, calls a model, or reads a real credential. Two
autouse fixtures make that structural rather than a rule each test remembers:
`_isolate_environment` pins the environment the command reads, and `wiring` puts a
fake connection and a fake agent where the command looks for them.
"""

import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool

from factchecker import cli
from factchecker.cache import RunCache
from factchecker.checker import CheckOutcome
from factchecker.config import ConfigurationError, Settings
from factchecker.errors import AuthenticationFailed
from factchecker.logging_setup import configure_logging
from factchecker.models import IdentifiedStatement, Ruling, Verdict
from tests.conftest import wire_statement

PACKAGE_LOGGER = "factchecker"
ENV_FILE_NAME = ".env"
DEFAULT_MODEL = "google/gemma-4-31b-it"

# The two credentials every test's command reads. Neither is a value any service
# would take, so a test that reached a network would fail rather than spend one.
PINNED_ENVIRONMENT = {
    "OPENROUTER_API_KEY": "sk-or-v1-not-a-key",
    "BRIGHTDATA_API_TOKEN": "brd-not-a-token",
}

_OVERRIDE_PREFIX = "FACTCHECKER_"
_PINNED_NAMES = {"LOG_LEVEL", *PINNED_ENVIRONMENT}


def _ruling(verdict: Verdict = "supported") -> Ruling:
    """A ruling the fake agent hands back for every statement it is given."""
    return Ruling(
        verdict=verdict,
        confidence=0.9,
        justification="The sources agree [1].",
        references=[],
    )


def _outcome(verdict: Verdict = "supported") -> CheckOutcome:
    """A check result with usage the output payload can add up."""
    return CheckOutcome(
        ruling=_ruling(verdict),
        prompt_tokens=100,
        completion_tokens=20,
        searches=2,
    )


def _search_tool() -> BaseTool:
    """One tool as `load_tools` hands it over, for the wiring to carry across."""

    async def call(query: str) -> str:
        raise AssertionError("no test calls the tool")

    return StructuredTool(
        name="search_engine",
        description="ask the search engine",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        coroutine=call,
    )


class _Loader:
    """A `load_tools` stand-in that hands back tools and watches its own release."""

    def __init__(self, *, tools: Sequence[BaseTool] = ()) -> None:
        self.tools = list(tools)
        self.failure: Exception | None = None
        self.endpoints: list[object] = []
        self.released = 0

    async def __call__(
        self, endpoint: object
    ) -> tuple[list[BaseTool], Callable[[], Awaitable[None]]]:
        """Open the fake connection, or fail the way the real one fails."""
        self.endpoints.append(endpoint)
        if self.failure is not None:
            raise self.failure
        return self.tools, self._release

    async def _release(self) -> None:
        """Count the release, so a test can see the connection was given back."""
        self.released += 1


class _Agent:
    """An `AgentChecker` stand-in that rules without a model or a connection.

    It is callable, so one object stands in both for the class the command
    constructs and for the checker that construction returns. That is what lets a
    test read the arguments the command built the agent with.
    """

    def __init__(self, outcome: CheckOutcome) -> None:
        self.outcome: CheckOutcome | Exception = outcome
        self.hangs = False
        self.turns = 0
        self.built: list[tuple[object, object, object]] = []
        self.seen: list[str] = []
        self.running = 0
        self.peak = 0

    def __call__(self, model: object, tools: object, settings: object) -> Self:
        """Record what the command built the agent with, and be that agent."""
        self.built.append((model, tools, settings))
        return self

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Hold a slot for as long as the test asked, then answer or raise."""
        self.seen.append(statement.id)
        self.running += 1
        self.peak = max(self.peak, self.running)
        for _ in range(self.turns):
            await asyncio.sleep(0)
        if self.hangs:
            await asyncio.Event().wait()
        self.running -= 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@dataclass
class _Wiring:
    """The fake collaborators the command was given, for a test to steer and read."""

    loader: _Loader
    agent: _Agent


def _opinion() -> dict[str, object]:
    """One statement the run passes through without checking."""
    return wire_statement(
        id="s2",
        statement="Water is the best drink",
        classification={"class": "opinion", "confidence": 0.7},
    )


def _second_fact() -> dict[str, object]:
    """A second factual statement, for the tests that need two checks at once."""
    return wire_statement(id="s2", statement="Water freezes at 0 C")


def _input_file(tmp_path: Path, payload: Mapping[str, object]) -> Path:
    """Write a payload to an input file the command can read."""
    return _input_text(tmp_path, json.dumps(payload))


def _input_text(tmp_path: Path, text: str) -> Path:
    """Write arbitrary text to an input file, whether or not it is JSON."""
    path = tmp_path / "statements.json"
    path.write_text(text, encoding="utf-8")
    return path


def _env_file(directory: Path, **written: str) -> Path:
    """Write an environment file for the command to read, and name its path."""
    path = directory / ENV_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}={value}" for name, value in written.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(input_path: Path, output_path: Path, *flags: str) -> int:
    """Drive the command the way a shell does, and return its exit code.

    Every test names an environment file under its own `tmp_path`, whether or not
    one is written there. Left to the default the command would read the `.env` a
    developer keeps beside the package, and no test may read a real credential.
    """
    return cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--env-file",
            str(input_path.parent / ENV_FILE_NAME),
            *flags,
        ]
    )


@pytest.fixture(autouse=True)
def _isolate_environment() -> Iterator[None]:
    """Give every test the same environment, whatever the developer has exported.

    The command reads `os.environ`, and an environment file is loaded into it, so
    both a value a test writes and a value a test's file supplies outlive the test
    that put it there. The whole environment is restored rather than a named few.
    """
    saved = dict(os.environ)
    for name in list(os.environ):
        if name.startswith(_OVERRIDE_PREFIX) or name in _PINNED_NAMES:
            del os.environ[name]
    os.environ.update(PINNED_ENVIRONMENT)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def wiring(monkeypatch: pytest.MonkeyPatch) -> _Wiring:
    """Put a fake connection and a fake agent where the command looks for them.

    Autouse and unconditional. A test that forgot to wire would open a connection
    to Bright Data, and this is a suite that continuous integration runs with no
    secret of any kind.
    """
    loader = _Loader()
    agent = _Agent(_outcome())
    monkeypatch.setattr(cli, "load_tools", loader)
    monkeypatch.setattr(cli, "AgentChecker", agent)
    return _Wiring(loader=loader, agent=agent)


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Undo the global logging state each test works against.

    A handler left behind writes into a captured stream that pytest has already
    closed, and the level left behind reaches every test that follows.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    level = logger.level
    established = list(logger.handlers)
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for handler in established:
        logger.addHandler(handler)
    logger.setLevel(level)


def test_verbose_raises_the_level_to_debug_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag is the caller's explicit choice, so it beats the variable."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    configure_logging(verbose=True)

    assert logging.getLogger(PACKAGE_LOGGER).level == logging.DEBUG


@pytest.mark.parametrize("named", ["WARNING", "warning"])
def test_log_level_names_the_level_when_verbose_is_absent(
    monkeypatch: pytest.MonkeyPatch, named: str
) -> None:
    """The variable sets the level, whichever case it is written in."""
    monkeypatch.setenv("LOG_LEVEL", named)

    configure_logging(verbose=False)

    assert logging.getLogger(PACKAGE_LOGGER).level == logging.WARNING


@pytest.mark.parametrize("named", [None, "", "chatty"])
def test_an_unset_or_unrecognised_log_level_falls_back_to_info(
    monkeypatch: pytest.MonkeyPatch, named: str | None
) -> None:
    """A level nothing knows must not silence the run it was meant to describe."""
    if named is not None:
        monkeypatch.setenv("LOG_LEVEL", named)

    configure_logging(verbose=False)

    assert logging.getLogger(PACKAGE_LOGGER).level == logging.INFO


def test_one_stderr_handler_is_attached_however_often_it_is_configured() -> None:
    """A second call must replace the handler, not stack a duplicate beside it."""
    configure_logging(verbose=False)
    configure_logging(verbose=True)

    handlers = logging.getLogger(PACKAGE_LOGGER).handlers
    assert len(handlers) == 1
    assert handlers[0].stream is sys.stderr


def test_a_handler_the_host_attached_is_left_where_it_is() -> None:
    """A library that detaches its host's handler breaks the host's own logging."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    host_handler = logging.NullHandler()
    logger.addHandler(host_handler)

    configure_logging(verbose=False)
    configure_logging(verbose=True)

    assert host_handler in logger.handlers
    assert len(logger.handlers) == 2


def test_a_run_writes_the_output_payload_to_the_output_path(tmp_path: Path) -> None:
    """The round trip: a file of statements in, a file of rulings out, exit zero."""
    input_path = _input_file(
        tmp_path, {"statements": [wire_statement(id="s1"), _opinion()]}
    )
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["model"] == DEFAULT_MODEL
    assert written["meta"]["counts"] == {
        "total": 2,
        "checked": 1,
        "skipped": 1,
        "failed": 0,
    }
    assert written["meta"]["usage"] == {
        "promptTokens": 100,
        "completionTokens": 20,
        "searches": 2,
    }
    assert datetime.fromisoformat(written["meta"]["startedAt"]).tzinfo is not None
    checked, skipped = written["statements"]
    assert checked["id"] == "s1"
    assert checked["surroundingContext"] == wire_statement()["surroundingContext"]
    assert checked["ruling"]["verdict"] == "supported"
    assert checked["error"] is None
    assert skipped["ruling"] is None


def test_the_payload_goes_to_the_file_and_only_diagnostics_to_the_streams(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stdout stays empty, stderr carries the record, and neither holds the payload."""
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "s1 " in captured.err
    assert "supported" in captured.err
    assert "surroundingContext" not in captured.err


def test_the_verbose_flag_raises_the_package_logger_to_debug(tmp_path: Path) -> None:
    """The flag reaches `configure_logging`, which is what the level proves."""
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    assert _run(input_path, tmp_path / "rulings.json", "--verbose") == 0

    assert logging.getLogger(PACKAGE_LOGGER).level == logging.DEBUG


def test_malformed_json_exits_two_and_writes_no_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing could be read, so there is no payload and no output file."""
    input_path = _input_text(tmp_path, '{"statements": [')
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 2

    assert not output_path.exists()
    assert str(input_path) in capsys.readouterr().err


def test_a_json_document_that_is_not_an_object_exits_two(tmp_path: Path) -> None:
    """A top-level array parses as JSON and still does not satisfy the contract."""
    input_path = _input_text(tmp_path, "[]")
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 2

    assert not output_path.exists()


def test_an_unrecognised_class_value_exits_two_and_names_the_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A third label must stop the run rather than pass through unchecked."""
    statement = wire_statement(
        id="s1", classification={"class": "guess", "confidence": 0.7}
    )
    input_path = _input_file(tmp_path, {"statements": [statement]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 2

    assert not output_path.exists()
    assert "guess" in capsys.readouterr().err


def test_an_unreadable_input_path_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path that names no file is an input failure, not a crash."""
    output_path = tmp_path / "rulings.json"

    assert _run(tmp_path / "absent.json", output_path) == 2

    assert not output_path.exists()
    assert "absent.json" in capsys.readouterr().err


def test_an_unreadable_input_never_opens_the_connection(
    tmp_path: Path, wiring: _Wiring
) -> None:
    """The read comes first, so a rejected input costs nothing upstream."""
    assert _run(tmp_path / "absent.json", tmp_path / "rulings.json") == 2

    assert wiring.loader.endpoints == []


def test_a_repeated_identifier_from_inside_the_run_exits_two(tmp_path: Path) -> None:
    """`assign_ids` rejects this from inside `run_check`, past the parse."""
    repeated = wire_statement(id="dup")
    input_path = _input_file(tmp_path, {"statements": [repeated, dict(repeated)]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 2

    assert not output_path.exists()


def test_a_statement_that_carries_an_error_still_exits_zero(
    tmp_path: Path, wiring: _Wiring
) -> None:
    """A payload exists, so the run succeeded; the failure is reported inside it."""
    wiring.agent.outcome = RuntimeError("the upstream service said no")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["counts"]["failed"] == 1
    entry = written["statements"][0]
    assert entry["ruling"] is None
    assert entry["error"]["kind"] == "check_failed"


def test_a_rejected_credential_exits_three_and_writes_no_output(
    tmp_path: Path, wiring: _Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """One credential failure fails every statement alike, so no payload exists."""
    wiring.agent.outcome = AuthenticationFailed("openrouter rejected the key")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 3

    assert not output_path.exists()
    assert "rejected the key" in capsys.readouterr().err


def test_an_unwritable_output_path_exits_four(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check produced a payload, so only the write failed, and 2 would be a lie."""
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "absent" / "rulings.json"

    assert _run(input_path, output_path) == 4

    assert not output_path.exists()
    stderr = capsys.readouterr().err
    assert stderr.count(str(output_path)) == 1
    assert "No such file or directory" in stderr
    assert "Traceback" not in stderr


def test_an_absent_required_setting_exits_five_and_names_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A variable nobody filled in is a setup mistake, and gets a code of its own."""
    monkeypatch.delenv("OPENROUTER_API_KEY")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 5

    assert not output_path.exists()
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_a_setting_that_will_not_parse_exits_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A misspelled override is the same operator mistake as an absent one."""
    monkeypatch.setenv("FACTCHECKER_CONCURRENCY", "lots")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    assert _run(input_path, tmp_path / "rulings.json") == 5

    assert "FACTCHECKER_CONCURRENCY" in capsys.readouterr().err


def test_a_server_that_cannot_be_reached_exits_five(
    tmp_path: Path, wiring: _Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """`load_tools` reports a misconfigured server, and the run never starts."""
    wiring.loader.failure = ConfigurationError("the MCP server offers no search_engine")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 5

    assert not output_path.exists()
    assert "offers no search_engine" in capsys.readouterr().err


def test_a_token_the_server_refuses_at_startup_exits_three(
    tmp_path: Path, wiring: _Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """The commonest operator error: a token that was present and was refused.

    Five would send an operator to fill in a variable that is already filled in.
    """
    wiring.loader.failure = AuthenticationFailed("the MCP server returned 401")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 3

    assert not output_path.exists()
    assert "returned 401" in capsys.readouterr().err


def test_the_endpoint_the_settings_carry_is_the_one_the_connection_opens(
    tmp_path: Path, wiring: _Wiring
) -> None:
    """The token reaches the tool layer inside the type that prints itself redacted."""
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    assert _run(input_path, tmp_path / "rulings.json") == 0

    (endpoint,) = wiring.loader.endpoints
    assert endpoint.unredacted_url().endswith(
        PINNED_ENVIRONMENT["BRIGHTDATA_API_TOKEN"]
    )


def test_the_connection_is_released_on_the_successful_path(
    tmp_path: Path, wiring: _Wiring
) -> None:
    """A run that ends well still has to give the connection back."""
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    assert _run(input_path, tmp_path / "rulings.json") == 0

    assert wiring.loader.released == 1


def test_the_connection_is_released_when_a_credential_is_rejected(
    tmp_path: Path, wiring: _Wiring
) -> None:
    """A rejected credential ends the run early, and not with the connection open."""
    wiring.agent.outcome = AuthenticationFailed("openrouter rejected the key")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    assert _run(input_path, tmp_path / "rulings.json") == 3

    assert wiring.loader.released == 1


def test_the_connection_is_released_when_the_run_fails_some_other_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wiring: _Wiring
) -> None:
    """However the run ends, the connection it opened is given back."""
    monkeypatch.setattr(cli, "run_check", _refusing_run_check)
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    with pytest.raises(OSError, match="connection reset"):
        _run(input_path, tmp_path / "rulings.json")

    assert wiring.loader.released == 1


def test_a_dropped_connection_inside_the_run_is_not_an_unreadable_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An `OSError` past the read is not the file, so it must not report as one.

    The read and the parse own that handler. Left spanning the run as well, a
    dropped connection would exit `2` and send an operator to look at their input.
    """
    monkeypatch.setattr(cli, "run_check", _refusing_run_check)
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    with pytest.raises(OSError, match="connection reset"):
        _run(input_path, tmp_path / "rulings.json")


async def _refusing_run_check(*arguments: object) -> object:
    """Stand in for the orchestrator, and fail the way a dropped connection does."""
    raise OSError("connection reset by peer")


def test_the_agent_is_built_once_from_the_model_the_tools_and_the_settings(
    tmp_path: Path, wiring: _Wiring
) -> None:
    """One agent holds no per-statement state, so one serves the whole run."""
    input_path = _input_file(
        tmp_path, {"statements": [wire_statement(id="s1"), _second_fact()]}
    )

    assert _run(input_path, tmp_path / "rulings.json") == 0

    (built,) = wiring.agent.built
    model, tools, settings = built
    assert isinstance(model, BaseChatModel)
    assert tools == []
    assert isinstance(settings, Settings)
    assert settings.model == DEFAULT_MODEL
    assert wiring.agent.seen == ["s1", "s2"]


def test_one_cache_and_one_tool_set_serve_every_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wiring: _Wiring
) -> None:
    """The cache earns its place across statements, so it cannot be per-statement."""
    offered = _search_tool()
    wiring.loader.tools = [offered]
    recorded: list[tuple[object, object, object, object]] = []

    def _recording_instrument(
        tools: object, cache: object, settings: object, sleep: object
    ) -> list[object]:
        recorded.append((tools, cache, settings, sleep))
        return list(tools)  # type: ignore[call-overload]

    monkeypatch.setattr(cli, "instrument", _recording_instrument)
    input_path = _input_file(
        tmp_path, {"statements": [wire_statement(id="s1"), _second_fact()]}
    )

    assert _run(input_path, tmp_path / "rulings.json") == 0

    (tools, cache, settings, sleep) = recorded[0]
    assert len(recorded) == 1
    assert tools == [offered]
    assert isinstance(cache, RunCache)
    assert isinstance(settings, Settings)
    assert sleep is asyncio.sleep


def test_the_configured_model_names_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`meta.model` says what checked the statements, so it reads the setting."""
    monkeypatch.setenv("FACTCHECKER_MODEL", "anthropic/claude-sonnet-5")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["model"] == "anthropic/claude-sonnet-5"


def test_the_configured_concurrency_bounds_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wiring: _Wiring
) -> None:
    """`RunSettings` defaults to this same 8, so only a moved value proves the wire."""
    monkeypatch.setenv("FACTCHECKER_CONCURRENCY", "1")
    wiring.agent.turns = 5
    input_path = _input_file(
        tmp_path, {"statements": [wire_statement(id="s1"), _second_fact()]}
    )

    assert _run(input_path, tmp_path / "rulings.json") == 0

    assert wiring.agent.peak == 1


def test_the_configured_statement_timeout_cancels_a_slow_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wiring: _Wiring
) -> None:
    """`RunSettings` defaults to the same 240.0, so a moved value is the only proof."""
    monkeypatch.setenv("FACTCHECKER_STATEMENT_TIMEOUT_SECONDS", "0.01")
    wiring.agent.hangs = True
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    error = written["statements"][0]["error"]
    assert error["kind"] == "timeout"
    assert "0.01 seconds" in error["message"]


def test_the_environment_file_supplies_a_setting_the_process_lacks(
    tmp_path: Path,
) -> None:
    """This is how an operator supplies the two credentials without exporting them."""
    _env_file(tmp_path, FACTCHECKER_MODEL="anthropic/claude-sonnet-5")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["model"] == "anthropic/claude-sonnet-5"


def test_the_process_environment_beats_the_environment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported value is the more deliberate of the two, so it wins."""
    _env_file(tmp_path, FACTCHECKER_MODEL="anthropic/claude-sonnet-5")
    monkeypatch.setenv("FACTCHECKER_MODEL", "google/gemma-4-31b-it:free")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["model"] == "google/gemma-4-31b-it:free"


def test_the_environment_file_is_the_named_path_and_is_not_searched_for(
    tmp_path: Path,
) -> None:
    """A search would let a developer's own file leak into whatever ran below it."""
    _env_file(tmp_path, FACTCHECKER_MODEL="anthropic/claude-sonnet-5")
    below = tmp_path / "below"
    below.mkdir()
    input_path = _input_file(below, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["model"] == DEFAULT_MODEL


def test_the_environment_file_sets_the_level_the_first_record_is_written_at(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The file is read before logging is set up, or its `LOG_LEVEL` does nothing."""
    _env_file(tmp_path, LOG_LEVEL="WARNING")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    assert _run(input_path, tmp_path / "rulings.json") == 0

    assert logging.getLogger(PACKAGE_LOGGER).level == logging.WARNING
    assert capsys.readouterr().err == ""


def test_verbose_puts_the_traceback_under_the_input_rejection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag's other visible effect, beside the DEBUG records of a real run."""
    output_path = tmp_path / "rulings.json"

    assert _run(tmp_path / "absent.json", output_path, "--verbose") == 2

    stderr = capsys.readouterr().err
    assert "Traceback" in stderr
    assert "FileNotFoundError" in stderr


def test_a_quietening_log_level_cannot_swallow_the_input_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`LOG_LEVEL` quiets the per-statement chatter, not the reason the run ended."""
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    output_path = tmp_path / "rulings.json"

    assert _run(tmp_path / "absent.json", output_path) == 2

    assert "absent.json" in capsys.readouterr().err


def test_a_quietening_log_level_cannot_swallow_the_credential_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wiring: _Wiring,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other fatal reason rides the same rule as the input rejection."""
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    wiring.agent.outcome = AuthenticationFailed("openrouter rejected the key")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 3

    assert "rejected the key" in capsys.readouterr().err


def test_a_quietening_log_level_cannot_swallow_the_setup_mistake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The third fatal reason rides that rule too, and it is the newest one."""
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    monkeypatch.delenv("BRIGHTDATA_API_TOKEN")
    input_path = _input_file(tmp_path, {"statements": [wire_statement(id="s1")]})

    assert _run(input_path, tmp_path / "rulings.json") == 5

    assert "BRIGHTDATA_API_TOKEN" in capsys.readouterr().err


def test_an_input_file_that_is_not_utf_eight_text_exits_two(tmp_path: Path) -> None:
    """A path aimed at a binary file is an input failure, not a crash."""
    input_path = tmp_path / "statements.json"
    # 0x80 is a continuation byte with nothing to continue, so no UTF-8 decoder
    # accepts it. This is what a PDF or an image named on --input looks like.
    input_path.write_bytes(b"\x80\x81")
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 2

    assert not output_path.exists()
