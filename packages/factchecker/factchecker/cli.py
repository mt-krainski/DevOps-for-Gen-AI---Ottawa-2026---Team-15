"""The command-line front end: two paths in, one payload out, a code on the way out."""

import argparse
import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from factchecker.checker import OfflineChecker
from factchecker.errors import AuthenticationFailed, InputValidationError
from factchecker.ingest import parse_input
from factchecker.logging_setup import configure_logging
from factchecker.models import OutputPayload
from factchecker.run import RunSettings, run_check

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INPUT_REJECTED = 2
EXIT_CREDENTIAL_REJECTED = 3
EXIT_OUTPUT_UNWRITABLE = 4

# No checking agent ships with this build, so nothing calls a model and no
# model slug would be true of the run.
_MODEL = "offline"


def main(argv: Sequence[str] | None = None) -> int:
    """Check every statement in the input file and write the rulings to the output file.

    Args:
        argv: The arguments after the program name. `None` reads `sys.argv`.

    Returns:
        One of the four exit-code constants above, whichever the path through this
        function reaches. The README's "Exit codes" section is where each code's
        meaning is published.
    """
    arguments = _parse_arguments(argv)
    configure_logging(arguments.verbose)
    # The run sits inside the handler beside the parse, because `run_check`
    # assigns the identifiers: a repeated one is rejected from in there.
    try:
        output = _check(arguments.input)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        InputValidationError,
    ) as rejection:
        # This handler covers the read, the parse and the whole run, so an
        # unexpected `OSError` reaches it with nothing in the message to say which
        # of the three raised it. `--verbose` is what asks for that traceback.
        logger.critical(
            "input %s cannot be checked: %s",
            arguments.input,
            rejection,
            exc_info=arguments.verbose,
        )
        return EXIT_INPUT_REJECTED
    except AuthenticationFailed as rejection:
        logger.critical("the run stopped: a credential was rejected: %s", rejection)
        return EXIT_CREDENTIAL_REJECTED
    try:
        _write_output(arguments.output, output)
    except OSError as failure:
        # `str(OSError)` repeats the filename, which this line already names.
        logger.critical(
            "output %s cannot be written: %s", arguments.output, failure.strerror
        )
        return EXIT_OUTPUT_UNWRITABLE
    return EXIT_OK


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Read the two paths and the verbosity flag off the command line."""
    parser = argparse.ArgumentParser(
        prog="factchecker",
        description="Check statements an upstream classifier labelled fact or opinion.",
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="the JSON file of statements to check"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="the JSON file to write rulings to"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="log at DEBUG rather than at INFO"
    )
    return parser.parse_args(argv)


def _check(input_path: Path) -> OutputPayload:
    """Read the statements from a file, check them, and return what the run produced."""
    payload = parse_input(_read_statements(input_path))
    return asyncio.run(
        run_check(payload, OfflineChecker(), RunSettings(model=_MODEL), _utc_now)
    )


def _read_statements(input_path: Path) -> Mapping[str, object]:
    """Read the input file as the JSON object the input contract is written over."""
    document = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise InputValidationError(
            "the payload must be a JSON object, and this one is a "
            f"{type(document).__name__}"
        )
    return document


def _write_output(output_path: Path, output: OutputPayload) -> None:
    """Write the payload as JSON, under the camelCase names the contract fixes."""
    output_path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _utc_now() -> datetime:
    """The current time, as the timezone-aware UTC value the output contract needs."""
    return datetime.now(UTC)
