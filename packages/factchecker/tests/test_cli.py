"""Tests for the logging setup and the command-line front end."""

import json
import logging
import sys
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path

import pytest

from factchecker import cli
from factchecker.checker import CheckOutcome
from factchecker.errors import AuthenticationFailed
from factchecker.logging_setup import configure_logging
from factchecker.models import IdentifiedStatement

PACKAGE_LOGGER = "factchecker"


class _FailingChecker:
    """A checker that fails every statement it is given."""

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Raise, so the run records an error entry and carries on."""
        raise RuntimeError("the upstream service said no")


class _RejectingChecker:
    """A checker whose credential the upstream service rejects."""

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Raise, so the run ends before it can assemble a payload."""
        raise AuthenticationFailed("openrouter rejected the key")


def _statement() -> dict[str, object]:
    """One factual statement, as it is written on the wire."""
    return {
        "id": "s1",
        "surroundingContext": "The paragraph around the claim.",
        "statement": "Water boils at 100 C",
        "classification": {"class": "fact", "confidence": 0.7},
    }


def _opinion() -> dict[str, object]:
    """One statement the run passes through without checking."""
    return {
        "id": "s2",
        "surroundingContext": "The paragraph around the claim.",
        "statement": "Water is the best drink",
        "classification": {"class": "opinion", "confidence": 0.7},
    }


def _input_file(tmp_path: Path, payload: Mapping[str, object]) -> Path:
    """Write a payload to an input file the command can read."""
    return _input_text(tmp_path, json.dumps(payload))


def _input_text(tmp_path: Path, text: str) -> Path:
    """Write arbitrary text to an input file, whether or not it is JSON."""
    path = tmp_path / "statements.json"
    path.write_text(text, encoding="utf-8")
    return path


def _run(input_path: Path, output_path: Path, *flags: str) -> int:
    """Drive the command the way a shell does, and return its exit code."""
    return cli.main(["--input", str(input_path), "--output", str(output_path), *flags])


@pytest.fixture(autouse=True)
def _isolate_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo the global logging state and the environment each test works against.

    A handler left behind writes into a captured stream that pytest has already
    closed, and the level left behind reaches every test that follows.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
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
    input_path = _input_file(tmp_path, {"statements": [_statement(), _opinion()]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["model"] == "offline"
    assert written["meta"]["counts"] == {
        "total": 2,
        "checked": 1,
        "skipped": 1,
        "failed": 0,
    }
    assert datetime.fromisoformat(written["meta"]["startedAt"]).tzinfo is not None
    checked, skipped = written["statements"]
    assert checked["id"] == "s1"
    assert checked["surroundingContext"] == _statement()["surroundingContext"]
    assert checked["ruling"]["verdict"] == "unverifiable"
    assert checked["error"] is None
    assert skipped["ruling"] is None


def test_the_payload_goes_to_the_file_and_only_diagnostics_to_the_streams(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stdout stays empty, stderr carries the record, and neither holds the payload."""
    input_path = _input_file(tmp_path, {"statements": [_statement()]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "s1 " in captured.err
    assert "unverifiable" in captured.err
    assert "surroundingContext" not in captured.err


def test_the_verbose_flag_raises_the_package_logger_to_debug(tmp_path: Path) -> None:
    """The flag reaches `configure_logging`, which is what the level proves."""
    input_path = _input_file(tmp_path, {"statements": [_statement()]})

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
    statement = _statement() | {"classification": {"class": "guess", "confidence": 0.7}}
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


def test_a_repeated_identifier_from_inside_the_run_exits_two(tmp_path: Path) -> None:
    """`assign_ids` rejects this from inside `run_check`, past the parse."""
    repeated = _statement() | {"id": "dup"}
    input_path = _input_file(tmp_path, {"statements": [repeated, dict(repeated)]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 2

    assert not output_path.exists()


def test_a_statement_that_carries_an_error_still_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload exists, so the run succeeded; the failure is reported inside it."""
    monkeypatch.setattr(cli, "OfflineChecker", _FailingChecker)
    input_path = _input_file(tmp_path, {"statements": [_statement()]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["meta"]["counts"]["failed"] == 1
    entry = written["statements"][0]
    assert entry["ruling"] is None
    assert entry["error"]["kind"] == "check_failed"


def test_a_rejected_credential_exits_three_and_writes_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One credential failure fails every statement alike, so no payload exists."""
    monkeypatch.setattr(cli, "OfflineChecker", _RejectingChecker)
    input_path = _input_file(tmp_path, {"statements": [_statement()]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 3

    assert not output_path.exists()
    assert "rejected the key" in capsys.readouterr().err


def test_an_unwritable_output_path_exits_four(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check produced a payload, so only the write failed, and 2 would be a lie."""
    input_path = _input_file(tmp_path, {"statements": [_statement()]})
    output_path = tmp_path / "absent" / "rulings.json"

    assert _run(input_path, output_path) == 4

    assert not output_path.exists()
    stderr = capsys.readouterr().err
    assert stderr.count(str(output_path)) == 1
    assert "No such file or directory" in stderr
    assert "Traceback" not in stderr


def test_verbose_puts_the_traceback_under_the_input_rejection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag's visible effect in this build, where nothing yet emits DEBUG."""
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other fatal reason rides the same rule as the input rejection."""
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    monkeypatch.setattr(cli, "OfflineChecker", _RejectingChecker)
    input_path = _input_file(tmp_path, {"statements": [_statement()]})
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 3

    assert "rejected the key" in capsys.readouterr().err


def test_an_input_file_that_is_not_utf_eight_text_exits_two(tmp_path: Path) -> None:
    """A path aimed at a binary file is an input failure, not a crash."""
    input_path = tmp_path / "statements.json"
    # 0x80 is a continuation byte with nothing to continue, so no UTF-8 decoder
    # accepts it. This is what a PDF or an image named on --input looks like.
    input_path.write_bytes(b"\x80\x81")
    output_path = tmp_path / "rulings.json"

    assert _run(input_path, output_path) == 2

    assert not output_path.exists()
