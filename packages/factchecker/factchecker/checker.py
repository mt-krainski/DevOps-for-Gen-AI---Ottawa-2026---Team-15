"""The checker seam: what a checking agent is asked for, and what it hands back."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from factchecker.models import IdentifiedStatement, Ruling


@dataclass(frozen=True)
class CheckOutcome:
    """One statement's ruling, and what producing it consumed."""

    ruling: Ruling
    prompt_tokens: int
    completion_tokens: int
    searches: int


@runtime_checkable
class StatementChecker(Protocol):
    """The seam a checking agent implements.

    Runtime-checkable so that an implementation can be asserted against the seam by
    `isinstance`. No type checker runs over this package, so that assertion is the
    only thing checking that an implementation exposes the seam at all: it catches a
    missing or renamed `check` and nothing more. The signature stays bound by the
    tests that call `check` for real.
    """

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Check one statement.

        Args:
            statement: The statement to check, with its identifier assigned.

        Returns:
            The ruling, and the tokens and searches that producing it consumed.
        """
        ...
