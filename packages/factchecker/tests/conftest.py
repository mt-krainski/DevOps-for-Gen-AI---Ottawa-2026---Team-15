"""Helpers the test files in this package share."""


def wire_statement(**overrides: object) -> dict[str, object]:
    """One factual statement as written on the wire, with the named keys replaced.

    The statement carries no `id`. Identifier assignment is what most callers
    exercise, and a caller that wants one supplies it as an override.
    """
    statement: dict[str, object] = {
        "surroundingContext": "Water is odd. Water boils at 100 C. The tables agree.",
        "statement": "Water boils at 100 C",
        "classification": {"class": "fact", "confidence": 0.7},
    }
    return statement | overrides
