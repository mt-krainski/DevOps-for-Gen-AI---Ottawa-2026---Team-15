"""Fakes standing in for the LangChain chat-model boundary in tests.

Tests mock at this seam (``BaseChatModel.with_structured_output(...).ainvoke(...)``)
so no live network call to OpenRouter is ever made.
"""

from typing import Any

from statement_classifier.models import Classification


class FakeStructuredModel:
    def __init__(self, result: Classification) -> None:
        self._result = result

    async def ainvoke(self, input: Any) -> Classification:
        return self._result


class FakeChatModel:
    def __init__(self, result: Classification) -> None:
        self._result = result

    def with_structured_output(
        self, schema: type[Classification]
    ) -> FakeStructuredModel:
        return FakeStructuredModel(self._result)
