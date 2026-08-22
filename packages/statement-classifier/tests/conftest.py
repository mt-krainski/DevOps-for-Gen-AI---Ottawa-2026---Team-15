"""Helpers the test files in this package share.

The test files import from this module by name. That resolves because pytest imports
a conftest before any test module and registers it in `sys.modules` under its
rootdir-relative dotted name, which `[tool.pytest.ini_options]` pins to this package.
"""

import pytest

from statement_classifier.models import Classification, Statement


class FakeModel:
    """Stands in for the `ChatOpenAI.with_structured_output(...)` runnable."""

    def __init__(self, responses: list[object]) -> None:
        """Queue the responses, one per `ainvoke` call rather than one per statement.

        Args:
            responses: `Classification` instances to return and exceptions to
                raise, consumed in call order.
        """
        self._responses = list(responses)
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str) -> Classification:
        """Record the prompt and answer with the next queued response."""
        self.calls.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def fact_statement() -> Statement:
    """A checkable claim, with the sentences around it as context."""
    return Statement(
        surrounding_context=(
            "We measured it carefully. The Eiffel Tower is 330 meters tall. "
            "Everyone was impressed."
        ),
        statement="The Eiffel Tower is 330 meters tall",
    )


@pytest.fixture
def opinion_statement() -> Statement:
    """A subjective statement, with the sentences around it as context."""
    return Statement(
        surrounding_context=(
            "We visited Paris last summer. The Eiffel Tower is beautiful at night. "
            "We took many photos."
        ),
        statement="The Eiffel Tower is beautiful at night",
    )
