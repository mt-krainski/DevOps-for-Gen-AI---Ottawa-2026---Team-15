"""Helpers the test files in this package share.

The test files import from this module by name. That resolves because pytest imports
a conftest before any test module and registers it in `sys.modules` under its
rootdir-relative dotted name, which `[tool.pytest.ini_options]` pins to this package.
It is pytest behaviour rather than a documented guarantee: should a pytest release
end it, collection fails loudly on every file, and the sanctioned fix is to move
`wire_statement` behind a fixture. Two callers use it at module scope, so that move
cascades into their constants.
"""


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
