"""The checker seam: what a checking agent returns, and the offline stand-in."""

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


class OfflineChecker:
    """A checker that rules without searching, so integrators can parse an output.

    The verdict is a deliberate stand-in. The README defines `unverifiable` as a search
    that ran and settled nothing, and here no search ran at all: the justification says
    so plainly rather than leaving a reader to infer it.
    """

    async def check(self, statement: IdentifiedStatement) -> CheckOutcome:
        """Rule `unverifiable` at no confidence, having consumed nothing.

        Args:
            statement: The statement that would have been checked.

        Returns:
            An outcome whose ruling cites nothing and whose usage is zero.
        """
        return CheckOutcome(
            ruling=Ruling(
                verdict="unverifiable",
                confidence=0.0,
                justification=(
                    "No search was performed because no checking agent is configured."
                ),
                references=[],
            ),
            prompt_tokens=0,
            completion_tokens=0,
            searches=0,
        )
