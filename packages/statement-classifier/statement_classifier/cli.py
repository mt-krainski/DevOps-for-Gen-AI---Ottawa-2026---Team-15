"""The command-line front end: JSON in, classified JSON out, a code on the way out."""

import argparse
import json
import sys
from pathlib import Path

from statement_classifier.config import DEFAULT_CONCURRENCY
from statement_classifier.errors import ClassifierError, ErrorCode
from statement_classifier.service import classify_statements_sync

EXIT_SUCCESS = 0
EXIT_INTERNAL_ERROR = 1
EXIT_INVALID_INPUT = 2
EXIT_AUTH_ERROR = 3

_EXIT_CODES_BY_ERROR = {
    ErrorCode.INVALID_INPUT: EXIT_INVALID_INPUT,
    ErrorCode.MISSING_API_KEY: EXIT_AUTH_ERROR,
    ErrorCode.AUTH_ERROR: EXIT_AUTH_ERROR,
}


class _JsonErrorArgumentParser(argparse.ArgumentParser):
    """Reports a usage error under the same {code, message} contract as the rest."""

    def error(self, message: str) -> None:
        """Write the usage error to stderr as JSON, then exit."""
        _write_error(ErrorCode.INVALID_INPUT, message)
        raise SystemExit(EXIT_INVALID_INPUT)


def _build_parser() -> argparse.ArgumentParser:
    parser = _JsonErrorArgumentParser(prog="statement-classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify_parser = subparsers.add_parser(
        "classify", help="Classify a batch of statements as fact or opinion"
    )
    classify_parser.add_argument(
        "--input",
        default="-",
        help="Input JSON file path, or '-' for stdin (default: stdin)",
    )
    classify_parser.add_argument(
        "--output",
        default="-",
        help="Output JSON file path, or '-' for stdout (default: stdout)",
    )
    classify_parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max concurrent LLM calls (default: %(default)s)",
    )
    return parser


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _write_output(path: str, content: str) -> None:
    if path == "-":
        sys.stdout.write(content)
    else:
        Path(path).write_text(content, encoding="utf-8")


def _write_error(code: ErrorCode, message: str) -> None:
    json.dump({"code": code, "message": message}, sys.stderr)
    sys.stderr.write("\n")


def _run_classify(args: argparse.Namespace) -> int:
    try:
        raw = _read_input(args.input)
    except (OSError, UnicodeDecodeError) as exc:
        _write_error(ErrorCode.IO_ERROR, f"Could not read input: {exc}")
        return EXIT_INTERNAL_ERROR

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _write_error(ErrorCode.INVALID_INPUT, f"Invalid JSON: {exc}")
        return EXIT_INVALID_INPUT

    try:
        output = classify_statements_sync(payload, concurrency=args.concurrency)
    except ClassifierError as exc:
        _write_error(exc.code, exc.message)
        return _EXIT_CODES_BY_ERROR.get(exc.code, EXIT_INTERNAL_ERROR)

    try:
        _write_output(args.output, output.model_dump_json(indent=2) + "\n")
    except OSError as exc:
        _write_error(ErrorCode.IO_ERROR, f"Could not write output: {exc}")
        return EXIT_INTERNAL_ERROR

    return EXIT_SUCCESS


def main(argv: list[str] | None = None) -> int:
    """Classify the batch the arguments name and write it where they say.

    Args:
        argv: The arguments after the program name. `None` reads `sys.argv`.

    Returns:
        One of the four exit-code constants above. The README's "CLI" section is
        where each code's meaning is published.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_INVALID_INPUT
    return _run_classify(args)


def run() -> None:
    """Console-script entry point: exit under whatever code `main` returns."""
    sys.exit(main())


if __name__ == "__main__":
    run()
