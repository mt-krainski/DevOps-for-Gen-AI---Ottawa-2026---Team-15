from statement_classifier.classifier import classify_statement
from statement_classifier.models import Classification, Statement
from tests.fakes import FakeChatModel


async def test_classifies_a_fact():
    statement = Statement(
        statement="The Eiffel Tower is 330 meters tall.",
        surroundingContext="We visited Paris last summer. The Eiffel Tower is 330 meters tall. It was breathtaking.",
    )
    fake_model = FakeChatModel(Classification(**{"class": "fact", "confidence": 0.92}))

    result = await classify_statement(statement, model=fake_model)

    assert result.class_ == "fact"
    assert 0.0 <= result.confidence <= 1.0


async def test_classifies_an_opinion():
    statement = Statement(
        statement="The Eiffel Tower is the most beautiful landmark in the world.",
        surroundingContext="We visited Paris last summer. The Eiffel Tower is the most beautiful landmark in the world. It was breathtaking.",
    )
    fake_model = FakeChatModel(
        Classification(**{"class": "opinion", "confidence": 0.81})
    )

    result = await classify_statement(statement, model=fake_model)

    assert result.class_ == "opinion"
    assert 0.0 <= result.confidence <= 1.0
