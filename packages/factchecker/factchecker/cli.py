"""The command-line front end: two paths in, one payload out, a code on the way out."""

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from factchecker.agent import AgentChecker
from factchecker.cache import RunCache
from factchecker.config import (
    ConfigurationError,
    Settings,
    build_model,
    load_settings,
)
from factchecker.errors import AuthenticationFailed, InputValidationError
from factchecker.ingest import parse_input
from factchecker.logging_setup import configure_logging
from factchecker.models import InputPayload, OutputPayload
from factchecker.run import RunSettings, run_check
from factchecker.tools import instrument, load_tools

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INPUT_REJECTED = 2
EXIT_CREDENTIAL_REJECTED = 3
EXIT_OUTPUT_UNWRITABLE = 4
EXIT_MISCONFIGURED = 5

DEFAULT_ENV_FILE = Path(".env")


def main(argv: Sequence[str] | None = None) -> int:
    """Check every statement in the input file and write the rulings to the output file.

    Args:
        argv: The arguments after the program name. `None` reads `sys.argv`.

    Returns:
        One of the five exit-code constants above, whichever the path through this
        function reaches. The README's "Exit codes" section is where each code's
        meaning is published.
    """
    arguments = _parse_arguments(argv)
    # Before the logging is configured, because `LOG_LEVEL` is one of the variables
    # an environment file may carry.
    load_dotenv(dotenv_path=arguments.env_file, override=False)
    configure_logging(arguments.verbose)
    try:
        payload = parse_input(_read_statements(arguments.input))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        InputValidationError,
    ) as rejection:
        return _rejected_input(arguments, rejection)
    try:
        output = asyncio.run(_check(payload))
    except InputValidationError as rejection:
        # `run_check` assigns the identifiers, so a repeated one is rejected from
        # in there, past the parse this handler's twin above covers.
        return _rejected_input(arguments, rejection)
    except ConfigurationError as misconfiguration:
        logger.critical("the run cannot start: %s", misconfiguration)
        return EXIT_MISCONFIGURED
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
    """Read the two paths, the environment file, and the verbosity flag."""
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
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=(
            "the file of settings to read before the process environment "
            f"(default: {DEFAULT_ENV_FILE} in the working directory)"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="log at DEBUG rather than at INFO"
    )
    return parser.parse_args(argv)


async def _check(payload: InputPayload) -> OutputPayload:
    """Open the Bright Data connection, check every statement over it, and close it.

    The connection has to outlive every statement, because every tool call runs over
    it, and it has to be given back however the run ends. So the whole of opening it,
    running the statements and closing it sits inside the command's one `asyncio.run`.

    One cache and one instrumented tool set serve the whole run. Statements drawn
    from one document search for overlapping things, and that sharing is what makes
    the cache worth having. Neither the tools nor the agent hold per-statement state.
    """
    settings = load_settings(os.environ)
    model = build_model(settings)
    tools, release = await load_tools(settings.mcp_endpoint)
    try:
        checker = AgentChecker(
            model,
            instrument(tools, RunCache(), settings, asyncio.sleep),
            settings,
        )
        return await run_check(payload, checker, _run_settings(settings), _utc_now)
    finally:
        await release()


def _run_settings(settings: Settings) -> RunSettings:
    """Hand the orchestrator the three bounds the environment set.

    All three are named. `RunSettings` defaults two of them to exactly the values
    `Settings` defaults them to, so a wiring that passed the model alone would leave
    `FACTCHECKER_CONCURRENCY` and `FACTCHECKER_STATEMENT_TIMEOUT_SECONDS` doing
    nothing at all, and every test would still pass.
    """
    return RunSettings(
        model=settings.model,
        concurrency=settings.concurrency,
        statement_timeout_seconds=settings.statement_timeout_seconds,
    )


def _rejected_input(arguments: argparse.Namespace, rejection: Exception) -> int:
    """Report an input the run will not take, and hand back its code."""
    logger.critical(
        "input %s cannot be checked: %s",
        arguments.input,
        rejection,
        exc_info=arguments.verbose,
    )
    return EXIT_INPUT_REJECTED


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
