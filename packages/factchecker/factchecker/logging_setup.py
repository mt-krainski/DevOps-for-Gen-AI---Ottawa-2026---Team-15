"""Where this package's log records go, and at what level."""

import logging
import os
import sys

_PACKAGE_LOGGER = "factchecker"
_FORMAT = "%(levelname)s %(name)s: %(message)s"


def configure_logging(verbose: bool) -> None:
    """Send this package's records to stderr, at the level the caller asked for.

    The level goes on the `factchecker` logger rather than on the handler alone.
    A handler admits nothing its logger has already discarded, and the root
    logger's default would discard INFO.

    Any handler an earlier call attached is removed, so calling this twice in one
    process writes each record once.

    Args:
        verbose: Log at DEBUG. Where this is false the level is the one
            `LOG_LEVEL` names, and INFO where that variable is unset or names no
            known level.
    """
    logger = logging.getLogger(_PACKAGE_LOGGER)
    for attached in list(logger.handlers):
        logger.removeHandler(attached)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else _named_level())


def _named_level() -> int:
    """Read the level `LOG_LEVEL` names, and INFO where it names nothing known."""
    named = os.environ.get("LOG_LEVEL", "").upper()
    return logging.getLevelNamesMapping().get(named, logging.INFO)
