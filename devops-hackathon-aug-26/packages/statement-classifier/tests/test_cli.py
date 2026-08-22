import io
import json

from statement_classifier import cli
from statement_classifier.errors import ErrorCode
from statement_classifier.models import ClassifiedStatement, Classification, ClassifierOutput

VALID_INPUT = {"statements": [{"surroundingContext": "ctx", "statement": "The sky is blue"}]}


def _fake_output() -> ClassifierOutput:
    return ClassifierOutput(
        statements=[
            ClassifiedStatement(
                surroundingContext="ctx",
                statement="The sky is blue",
                classification=Classification(**{"class": "fact", "confidence": 0.9}),
                error=None,
            )
        ]
    )


def test_valid_file_in_writes_file_out_with_exit_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "classify_statements_sync", lambda payload, concurrency: _fake_output()
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(VALID_INPUT))

    exit_code = cli.main(["classify", "--input", str(input_path), "--output", str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["statements"][0]["classification"]["class"] == "fact"


def test_malformed_json_input_exits_nonzero_with_stderr_error(tmp_path, capsys):
    input_path = tmp_path / "input.json"
    input_path.write_text("{not valid json")

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code != 0
    err = json.loads(capsys.readouterr().err)
    assert "code" in err
    assert "message" in err


def test_invalid_schema_exits_two(tmp_path, capsys):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"statements": [{"surroundingContext": "ctx"}]}))

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT


def test_missing_api_key_exits_three(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(VALID_INPUT))

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code == 3
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.MISSING_API_KEY


def test_stdin_and_stdout_are_used_by_default(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "classify_statements_sync", lambda payload, concurrency: _fake_output()
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(VALID_INPUT)))

    exit_code = cli.main(["classify"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["statements"][0]["classification"]["class"] == "fact"


def test_bad_flag_value_exits_with_json_error(capsys):
    exit_code = cli.main(["classify", "--concurrency", "notanumber"])

    assert exit_code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT


def test_missing_subcommand_exits_with_json_error(capsys):
    exit_code = cli.main([])

    assert exit_code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == ErrorCode.INVALID_INPUT


def test_non_utf8_input_file_exits_nonzero_with_stderr_error(tmp_path, capsys):
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b"\xff\xfe\x00\x01")

    exit_code = cli.main(["classify", "--input", str(input_path)])

    assert exit_code != 0
    err = json.loads(capsys.readouterr().err)
    assert "code" in err
    assert "message" in err


def test_concurrency_override_is_passed_through(tmp_path, monkeypatch):
    captured = {}

    def fake(payload, concurrency):
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
