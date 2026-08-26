"""The command line: a file in, a file out, and one code on the way out."""

import json
import logging
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path

import pytest

from fact_checker import cli, service
from fact_checker.cli import main, run
from fact_checker.config import CheckerConfig
from fact_checker.errors import AuthenticationFailure, CheckError, ErrorCode
from fact_checker.tools import SCRAPE_AS_MARKDOWN, SEARCH_ENGINE, open_toolkit
from tests.conftest import (
    BRIGHT_DATA_CREDENTIAL,
    OPENROUTER_CREDENTIAL,
    FakeMCPClient,
    FakeTool,
    Plan,
    Script,
    a_payload,
    a_script,
    a_statement,
    always,
)

A_CLAIM = "The bridge opened in 1937."
A_SEARCH_RESULT = '[{"url": "https://example.test/bridge", "title": "The bridge"}]'


@pytest.fixture(autouse=True)
def package_logger_restored() -> Iterator[None]:
    """Put the package logger back, so one test's handler cannot outlive it."""
    package_logger = logging.getLogger("fact_checker")
    handlers, level = list(package_logger.handlers), package_logger.level
    yield
    package_logger.handlers = handlers
    package_logger.setLevel(level)


@pytest.fixture
def offline(clean_env: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Give the run fake credentials, and keep it away from any real `.env`."""
    clean_env.setenv("OPENROUTER_API_KEY", OPENROUTER_CREDENTIAL)
    clean_env.setenv("BRIGHTDATA_API_TOKEN", BRIGHT_DATA_CREDENTIAL)
    clean_env.delenv("LOG_LEVEL", raising=False)
    clean_env.setattr(cli, "load_dotenv", lambda: None)
    return clean_env


def bright_data_tools() -> list[FakeTool]:
    """Build the two tools the Bright Data server offers, answering fixed text."""
    return [
        FakeTool(SEARCH_ENGINE, always(A_SEARCH_RESULT)),
        FakeTool(SCRAPE_AS_MARKDOWN, always("# The bridge")),
    ]


def connect_to_fakes(monkeypatch: pytest.MonkeyPatch, script: Script) -> None:
    """Point the service at a fake MCP server and the script's two models."""
    monkeypatch.setattr(
        service,
        "open_toolkit",
        partial(
            open_toolkit,
            client_factory=lambda _url: FakeMCPClient(bright_data_tools()),
        ),
    )
    monkeypatch.setattr(
        service,
        "build_models",
        lambda _config, _toolkit: (script.checking_model, script.ruling_model),
    )


def a_file(directory: Path, payload: object) -> Path:
    """Write a payload to `statements.json` and return the path."""
    path = directory / "statements.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def error_object(stderr: str) -> dict[str, str]:
    """Return the one `{code, message}` object the run wrote to stderr."""
    objects = [
        parsed
        for line in stderr.splitlines()
        if (parsed := _as_error(line)) is not None
    ]
    assert len(objects) == 1, f"expected one error object, got {objects}"
    return objects[0]


def _as_error(line: str) -> dict[str, str] | None:
    try:
        parsed = json.loads(line)
    except ValueError:
        return None
    if isinstance(parsed, dict) and parsed.keys() == {"code", "message"}:
        return parsed
    return None


def messages_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return every message the package logged, whatever its level."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("fact_checker")
    ]


def run_over(
    directory: Path, payload: object, *, argv: Sequence[str] | None = None
) -> tuple[int, Path]:
    """Run the command over a payload file, and report the code and output path."""
    input_path = a_file(directory, payload)
    output_path = directory / "rulings.json"
    arguments = (
        list(argv)
        if argv is not None
        else ["--input", str(input_path), "--output", str(output_path)]
    )
    return main(arguments), output_path


def test_a_run_writes_the_payload_to_the_named_file(
    tmp_path: Path, offline: pytest.MonkeyPatch
) -> None:
    """The output goes to the file `--output` names, and the run reports success."""
    connect_to_fakes(offline, a_script(A_CLAIM))

    code, output_path = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 0
    written = output_path.read_text(encoding="utf-8")
    assert written.endswith("\n")
    payload = json.loads(written)
    assert payload["statements"][0]["ruling"]["verdict"] == "supported"
    assert payload["statements"][0]["surroundingContext"].endswith(A_CLAIM)
    assert payload["meta"]["counts"]["total"] == 1
    assert payload["meta"]["usage"]["promptTokens"] == 0
    assert payload["meta"]["startedAt"].endswith("Z")


def test_the_run_writes_nothing_at_all_to_stdout(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Logs go to stderr and the payload goes to a file; stdout stays empty."""
    connect_to_fakes(offline, a_script(A_CLAIM))

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 0
    assert capsys.readouterr().out == ""


def test_an_input_file_that_is_not_there_reports_and_returns_two(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No input is no run, and the reason says which file was wanted."""
    code = main(
        [
            "--input",
            str(tmp_path / "absent.json"),
            "--output",
            str(tmp_path / "rulings.json"),
        ]
    )

    assert code == 2
    assert "absent.json" in error_object(capsys.readouterr().err)["message"]


def test_input_that_is_not_json_reports_and_returns_two(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unparseable input is rejected before any credential is read."""
    input_path = tmp_path / "statements.json"
    input_path.write_text("{not json at all", encoding="utf-8")

    code = main(
        ["--input", str(input_path), "--output", str(tmp_path / "rulings.json")]
    )

    assert code == 2
    assert error_object(capsys.readouterr().err)["code"] == "INVALID_INPUT"


def test_a_payload_that_fails_the_contract_returns_two(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A third classification label is rejected, and the value is named."""
    connect_to_fakes(offline, a_script(A_CLAIM))

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM, kind="speculation")))

    assert code == 2
    reported = error_object(capsys.readouterr().err)
    assert reported["code"] == "INVALID_INPUT"
    assert "speculation" in reported["message"]


def test_json_that_is_not_an_object_at_all_returns_two(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Valid JSON is not a valid payload, and the contract is what decides."""
    connect_to_fakes(offline, a_script(A_CLAIM))

    code, _ = run_over(tmp_path, ["not", "an", "envelope"])

    assert code == 2
    assert error_object(capsys.readouterr().err)["code"] == "INVALID_INPUT"


def test_a_missing_credential_returns_three(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing can run without a key, and the reason names the variable."""
    offline.delenv("OPENROUTER_API_KEY")

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 3
    reported = error_object(capsys.readouterr().err)
    assert reported["code"] == "MISSING_CREDENTIAL"
    assert "OPENROUTER_API_KEY" in reported["message"]


def test_a_rejected_credential_returns_three(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rejected credential ends the run, and no payload is written."""
    connect_to_fakes(
        offline,
        a_script(A_CLAIM, **{A_CLAIM: Plan(failure=AuthenticationFailure("rejected"))}),
    )

    code, output_path = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 3
    assert not output_path.exists()
    assert error_object(capsys.readouterr().err)["code"] == "AUTH_ERROR"


def test_an_output_path_that_cannot_be_written_returns_four(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The payload was built, so the failure to store it is its own code."""
    connect_to_fakes(offline, a_script(A_CLAIM))
    input_path = a_file(tmp_path, a_payload(a_statement(A_CLAIM)))

    code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "no-such-directory" / "rulings.json"),
        ]
    )

    assert code == 4
    assert error_object(capsys.readouterr().err)["code"] == "IO_ERROR"


def test_a_missing_input_argument_reports_under_the_same_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A usage error is an error like any other, in the same JSON shape."""
    code = main(["--output", "rulings.json"])

    assert code == 2
    reported = error_object(capsys.readouterr().err)
    assert reported["code"] == "INVALID_INPUT"
    assert "--input" in reported["message"]


def test_a_missing_output_argument_reports_under_the_same_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both paths are required, so neither has a silent default."""
    code = main(["--input", "statements.json"])

    assert code == 2
    assert "--output" in error_object(capsys.readouterr().err)["message"]


def test_every_non_zero_exit_logs_its_reason_at_critical(
    tmp_path: Path, offline: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No setting of LOG_LEVEL can hide why a run returned a non-zero code."""
    caplog.set_level(logging.CRITICAL, logger="fact_checker")
    offline.setenv("LOG_LEVEL", "CRITICAL")
    offline.delenv("BRIGHTDATA_API_TOKEN")

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 3
    critical = [
        record for record in caplog.records if record.levelno == logging.CRITICAL
    ]
    assert len(critical) == 1
    assert "BRIGHTDATA_API_TOKEN" in critical[0].getMessage()


def test_the_default_level_writes_a_line_per_statement_and_no_tool_line(
    tmp_path: Path, offline: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """INFO is one line per statement; the per-call detail is a level below."""
    caplog.set_level(logging.DEBUG, logger="fact_checker")
    connect_to_fakes(offline, a_script(A_CLAIM, **{A_CLAIM: Plan(tool_calls=1)}))

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 0
    logged = messages_of(caplog)
    assert any(message.startswith("s1: supported in ") for message in logged)
    assert not any(SEARCH_ENGINE in message for message in logged)


def test_the_debug_level_adds_a_line_for_each_tool_call(
    tmp_path: Path, offline: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """At DEBUG the operator sees what was searched and how much came back."""
    caplog.set_level(logging.DEBUG, logger="fact_checker")
    offline.setenv("LOG_LEVEL", "DEBUG")
    connect_to_fakes(offline, a_script(A_CLAIM, **{A_CLAIM: Plan(tool_calls=1)}))

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 0
    assert any(SEARCH_ENGINE in message for message in messages_of(caplog))


def test_an_unrecognised_log_level_falls_back_to_info_and_says_so(
    tmp_path: Path, offline: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A typo in LOG_LEVEL costs a warning, never the run."""
    caplog.set_level(logging.DEBUG, logger="fact_checker")
    offline.setenv("LOG_LEVEL", "chatty")
    connect_to_fakes(offline, a_script(A_CLAIM))

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 0
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "chatty" in warnings[0]
    assert logging.getLogger("fact_checker").level == logging.INFO


def test_no_log_record_anywhere_carries_a_credential(
    tmp_path: Path, offline: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The loudest level the tool offers still says nothing about either secret.

    This sweeps every record captured, not only this package's own. A library
    underneath it that quoted a credential would fail this test too.
    """
    caplog.set_level(logging.DEBUG, logger="fact_checker")
    offline.setenv("LOG_LEVEL", "DEBUG")
    connect_to_fakes(offline, a_script(A_CLAIM, **{A_CLAIM: Plan(tool_calls=1)}))

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    assert code == 0
    assert caplog.records
    for record in caplog.records:
        assert BRIGHT_DATA_CREDENTIAL not in record.getMessage()
        assert OPENROUTER_CREDENTIAL not in record.getMessage()


def test_an_unexpected_crash_returns_one_without_quoting_the_token(
    tmp_path: Path,
    offline: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The crash path is the last rendering surface, so it scrubs like the rest."""
    caplog.set_level(logging.DEBUG, logger="fact_checker")
    offline.setenv("LOG_LEVEL", "DEBUG")

    async def crash(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            f"the transport gave up on "
            f"https://mcp.brightdata.com/mcp?token={BRIGHT_DATA_CREDENTIAL}"
        )

    offline.setattr(cli, "check_statements", crash)

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    captured = capsys.readouterr()
    assert code == 1
    assert BRIGHT_DATA_CREDENTIAL not in captured.err
    assert "***" in error_object(captured.err)["message"]

    logged = messages_of(caplog)
    chain = next(message for message in logged if message.startswith("the run crashed"))
    assert "RuntimeError" in chain
    assert "***" in chain
    for message in logged:
        assert BRIGHT_DATA_CREDENTIAL not in message


def test_a_crash_quoting_the_openrouter_key_is_scrubbed_too(
    tmp_path: Path, offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither credential reaches a reader, whichever one the failure quoted."""

    async def crash(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(f"Bearer {OPENROUTER_CREDENTIAL} was refused")

    offline.setattr(cli, "check_statements", crash)

    code, _ = run_over(tmp_path, a_payload(a_statement(A_CLAIM)))

    captured = capsys.readouterr()
    assert code == 1
    assert OPENROUTER_CREDENTIAL not in captured.err


def test_the_dotenv_file_is_read_before_the_configuration(
    tmp_path: Path, clean_env: pytest.MonkeyPatch
) -> None:
    """A `.env` is only useful if it is in place before the variables are read."""
    order: list[str] = []

    def note_the_configuration_read() -> CheckerConfig:
        order.append("config")
        raise CheckError(ErrorCode.MISSING_CREDENTIAL, "read no further")

    clean_env.setattr(cli, "load_dotenv", lambda: order.append("dotenv"))
    clean_env.setattr(cli, "load_config", note_the_configuration_read)

    code, _ = run_over(tmp_path, a_payload())

    assert code == 3
    assert order == ["dotenv", "config"]


def test_the_console_script_exits_under_whatever_main_returns(
    offline: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run` is the entry point the installed command calls."""
    offline.setattr("sys.argv", ["fact-checker", "--output", "rulings.json"])

    with pytest.raises(SystemExit) as raised:
        run()

    assert raised.value.code == 2
    assert error_object(capsys.readouterr().err)["code"] == "INVALID_INPUT"
