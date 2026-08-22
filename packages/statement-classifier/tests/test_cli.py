"""Tests for the command line in `statement_classifier.cli`."""

import io
import json
from pathlib import Path

import pytest

from statement_classifier import cli
from statement_classifier.errors import ErrorCode
from statement_classifier.models import (
    Classification,
    ClassifiedStatement,
    ClassifierOutput,
)

VALID_INPUT = {
    "statements": [{"surroundingContext": "ctx", "statement": "The sky is blue"}]
}
VALID_TEXT_INPUT = {"text": "The sky is blue"}


def _fake_output() -> ClassifierOutput:
    return ClassifierOutput(
        statements=[
            ClassifiedStatement(
                surrounding_context="ctx",
                statement="The sky is blue",
                classification=Classification(**{"class": "fact", "confidence": 0.9}),
                error=None,
            )
        ]
    )


def _fake_text_output() -> ClassifierOutput:
    return ClassifierOutput(
        statements=[
            ClassifiedStatement(
                surrounding_context="The sky is blue",
                statement="The sky is blue",
                classification=Classification(**{"class": "fact", "confidence": 0.9}),
                error=None,
            )
        ]
    )


def test_valid_file_in_writes_file_out_with_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A readable batch is written where `--output` says, under exit code 0."""
    monkeypatch.setattr(
        cli, "classify_statements_sync", lambda payload, concurrency: _fake_output()
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(VALID_INPUT))

    exit_code = cli.main(
        ["classify", "--input", str(input_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["statements"][0]["classification"]["class"] == "fact"
    assert result["statements"][0]["surroundingContext"] == "ctx"


def test_malformed_json_input_exits_nonzero_with_stderr_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unparseable input is reported as JSON on stderr, not as a traceback."""
    input_path = tmp_path / "input.json"
    input_path.write_text("{not valid json")

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code != 0
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT
    assert "message" in err


def test_invalid_schema_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Valid JSON in the wrong shape is still invalid input."""
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"statements": [{"surroundingContext": "ctx"}]}))

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT


def test_missing_api_key_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing credential is a config failure, told apart from bad input."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(VALID_INPUT))

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code == 3
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.MISSING_API_KEY


def test_stdin_and_stdout_are_used_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With neither flag given the command composes in a shell pipeline."""
    monkeypatch.setattr(
        cli, "classify_statements_sync", lambda payload, concurrency: _fake_output()
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(VALID_INPUT)))

    exit_code = cli.main(["classify"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["statements"][0]["classification"]["class"] == "fact"


def test_bad_flag_value_exits_with_json_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A usage error follows the same stderr contract as every other failure."""
    exit_code = cli.main(["classify", "--concurrency", "notanumber"])

    assert exit_code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT


def test_missing_subcommand_exits_with_json_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Naming no subcommand is a usage error, reported the same way."""
    exit_code = cli.main([])

    assert exit_code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT


def test_non_utf8_input_file_exits_nonzero_with_stderr_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file that will not decode is an IO failure, not a schema one."""
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b"\xff\xfe\x00\x01")

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code != 0
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.IO_ERROR
    assert "message" in err


def test_concurrency_override_is_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--concurrency` reaches the batch call rather than being parsed and dropped."""
    captured = {}

    def fake(payload: object, concurrency: int) -> ClassifierOutput:
        captured["concurrency"] = concurrency
        return _fake_output()

    monkeypatch.setattr(cli, "classify_statements_sync", fake)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(VALID_INPUT))

    exit_code = cli.main(
        ["classify", "--input", str(input_path), "--concurrency", "10"]
    )

    assert exit_code == 0
    assert captured["concurrency"] == 10


def test_classify_text_valid_file_writes_output_with_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text input is split, classified, and written under exit code 0."""
    monkeypatch.setattr(
        cli,
        "classify_text_sync",
        lambda payload, concurrency: _fake_text_output(),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(VALID_TEXT_INPUT))

    exit_code = cli.main(
        [
            "classify-text",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["statements"][0]["classification"]["class"] == "fact"
    assert result["statements"][0]["surroundingContext"] == "The sky is blue"


def test_classify_text_malformed_json_exits_nonzero_with_stderr_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unparseable input is reported as JSON on stderr, not as a traceback."""
    input_path = tmp_path / "input.json"
    input_path.write_text("{not valid json")

    exit_code = cli.main(["classify-text", "--input", str(input_path)])

    assert exit_code != 0
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT
    assert "message" in err


def test_classify_text_missing_api_key_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing credential is a config failure, told apart from bad input."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(VALID_TEXT_INPUT))

    exit_code = cli.main(["classify-text", "--input", str(input_path)])

    assert exit_code == 3
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.MISSING_API_KEY
