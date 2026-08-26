"""The command line: read a file, check the batch, write a file, return a code."""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv

from fact_checker.config import CheckerConfig, load_config
from fact_checker.errors import CheckError, ErrorCode
from fact_checker.service import check_statements
from fact_checker.tools import without_the_token

EXIT_SUCCESS = 0
EXIT_CRASHED = 1
EXIT_INVALID_INPUT = 2
EXIT_CREDENTIAL_REFUSED = 3
EXIT_OUTPUT_UNWRITABLE = 4

PACKAGE_LOGGER = "fact_checker"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")

_EXIT_CODES_BY_ERROR = {
    ErrorCode.INVALID_INPUT: EXIT_INVALID_INPUT,
    ErrorCode.MISSING_CREDENTIAL: EXIT_CREDENTIAL_REFUSED,
    ErrorCode.AUTH_ERROR: EXIT_CREDENTIAL_REFUSED,
}

logger = logging.getLogger(__name__)


class _JsonErrorArgumentParser(argparse.ArgumentParser):
    """Reports a usage error under the same contract as every other failure."""

    def error(self, message: str) -> NoReturn:
        """Report the usage error as JSON on stderr, then leave."""
        _report(ErrorCode.INVALID_INPUT, message, EXIT_INVALID_INPUT)
        raise SystemExit(EXIT_INVALID_INPUT)


def main(argv: list[str] | None = None) -> int:
    """Check the batch the arguments name, and write it where they say.

    Args:
        argv: The arguments after the program name. `None` reads `sys.argv`.

    Returns:
        `0` where a payload was written, `1` for an unexpected crash, `2` where
        the input could not be read or failed the contract, `3` where a
        credential was missing or rejected, and `4` where the payload was built
        and could not be stored. The README publishes the same table.
    """
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as leaving:
        return leaving.code if isinstance(leaving.code, int) else EXIT_INVALID_INPUT

    load_dotenv()
    _configure_logging()
    return _run(Path(arguments.input), Path(arguments.output))


def run() -> None:
    """Console-script entry point: exit under whatever `main` returns."""
    sys.exit(main())


def _build_parser() -> argparse.ArgumentParser:
    parser = _JsonErrorArgumentParser(
        prog="fact-checker",
        description="Check classified statements against the web, with citations.",
    )
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    return parser


def _configure_logging() -> None:
    requested = os.environ.get("LOG_LEVEL", "").strip() or DEFAULT_LOG_LEVEL
    recognised = requested.upper() in LOG_LEVELS

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    package_logger = logging.getLogger(PACKAGE_LOGGER)
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(requested.upper() if recognised else DEFAULT_LOG_LEVEL)

    if not recognised:
        logger.warning(
            "LOG_LEVEL %r is not one of %s; using %s",
            requested,
            ", ".join(LOG_LEVELS),
            DEFAULT_LOG_LEVEL,
        )


def _run(input_path: Path, output_path: Path) -> int:
    try:
        raw = input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _report(
            ErrorCode.IO_ERROR,
            f"could not read {input_path}: {exc}",
            EXIT_INVALID_INPUT,
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _report(
            ErrorCode.INVALID_INPUT,
            f"{input_path} does not hold valid JSON: {exc}",
            EXIT_INVALID_INPUT,
        )

    try:
        config = load_config()
    except CheckError as exc:
        return _report(exc.code, exc.message, _exit_for(exc.code))

    return _check_and_write(payload, output_path, config)


def _check_and_write(payload: object, output_path: Path, config: CheckerConfig) -> int:
    try:
        output = asyncio.run(check_statements(payload, config=config))
    except CheckError as exc:
        return _report(exc.code, exc.message, _exit_for(exc.code), config=config)
    except Exception as exc:  # noqa: BLE001 — the crash barrier: a code, not a trace
        _log_the_chain(exc, config)
        return _report(
            ErrorCode.AGENT_ERROR,
            f"{type(exc).__name__}: {exc}",
            EXIT_CRASHED,
            config=config,
        )

    try:
        output_path.write_text(
            output.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return _report(
            ErrorCode.IO_ERROR,
            f"could not write {output_path}: {exc}",
            EXIT_OUTPUT_UNWRITABLE,
            config=config,
        )

    return EXIT_SUCCESS


def _exit_for(code: ErrorCode) -> int:
    return _EXIT_CODES_BY_ERROR.get(code, EXIT_CRASHED)


def _report(
    code: ErrorCode,
    message: str,
    exit_code: int,
    *,
    config: CheckerConfig | None = None,
) -> int:
    reported = _fit_to_report(message, config)
    json.dump({"code": code, "message": reported}, sys.stderr)
    sys.stderr.write("\n")
    logger.critical("exit %d: %s: %s", exit_code, code, reported)
    return exit_code


def _log_the_chain(exc: BaseException, config: CheckerConfig) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    chain = "".join(traceback.format_exception(exc)).rstrip()
    logger.debug("the run crashed:\n%s", _fit_to_report(chain, config))


def _fit_to_report(text: str, config: CheckerConfig | None) -> str:
    # Every raise in this package chains `from exc`, so an upstream failure's
    # own message survives to here as a cause, and that message can quote the
    # endpoint URL the Bright Data token rides in. This is the last surface
    # before a reader sees it. A run whose configuration never loaded holds no
    # credential, and so has none to keep out.
    if config is None:
        return text
    return without_the_token(
        without_the_token(text, config.bright_data.api_token), config.api_key
    )
