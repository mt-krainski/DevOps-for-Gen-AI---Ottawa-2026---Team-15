import pytest

from statement_classifier.models import Classification, Statement


class FakeModel:
    """Stands in for the `ChatOpenAI.with_structured_output(...)` runnable."""

    def __init__(self, responses):
        # responses: list of Classification instances or Exceptions, consumed
        # in call order (one entry per `ainvoke` call, not per statement).
        self._responses = list(responses)
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str) -> Classification:
        self.calls.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def fact_statement() -> Statement:
    return Statement(
        surroundingContext="We measured it carefully. The Eiffel Tower is 330 meters tall. Everyone was impressed.",
        statement="The Eiffel Tower is 330 meters tall",
    )


@pytest.fixture
def opinion_statement() -> Statement:
    return Statement(
        surroundingContext="We visited Paris last summer. The Eiffel Tower is beautiful at night. We took many photos.",
        statement="The Eiffel Tower is beautiful at night",
    )
