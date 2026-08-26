"""The one credentialed test: the whole package against both live services."""

import json
import os
from pathlib import Path
from typing import get_args

import pytest
from dotenv import dotenv_values, find_dotenv

from fact_checker.cli import main
from fact_checker.models import Verdict

REQUIRED_VARIABLES = ("OPENROUTER_API_KEY", "BRIGHTDATA_API_TOKEN")

VERDICTS = get_args(Verdict)

A_FACT = "Water boils at 100 degrees Celsius at one atmosphere of pressure."
AN_OPINION = "Boiling water is the most satisfying way to cook pasta."


def _credentials_are_available() -> bool:
    # The command reads the `.env` file itself, so this reads it too rather
    # than deciding on the environment alone. `dotenv_values` reports what the
    # file holds without putting any of it into this process's environment.
    from_file = dotenv_values(find_dotenv())
    return all(
        os.environ.get(name) or from_file.get(name) for name in REQUIRED_VARIABLES
    )


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _credentials_are_available(),
        reason=f"needs {' and '.join(REQUIRED_VARIABLES)}",
    ),
]


def a_live_payload() -> dict[str, object]:
    """Build one fact with a stable public answer, and one plain opinion."""
    return {
        "statements": [
            {
                "id": "the-fact",
                "surroundingContext": (
                    "The chapter set out the physics behind everyday cooking."
                ),
                "statement": A_FACT,
                "classification": {"class": "fact", "confidence": 0.95},
            },
            {
                "id": "the-opinion",
                "surroundingContext": (
                    "The chapter closed on the author's own preferences."
                ),
                "statement": AN_OPINION,
                "classification": {"class": "opinion", "confidence": 0.9},
            },
        ]
    }


def test_a_live_run_rules_on_the_fact_and_passes_the_opinion_through(
    tmp_path: Path,
) -> None:
    """The whole path, once, against the real gateway and the real MCP server.

    Which verdict comes back is the model's business, and pinning it here would
    make this a quality measure. That is the promptfoo suite's job.
    """
    input_path = tmp_path / "statements.json"
    input_path.write_text(json.dumps(a_live_payload()), encoding="utf-8")
    output_path = tmp_path / "rulings.json"

    code = main(["--input", str(input_path), "--output", str(output_path)])

    assert code == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in written["statements"]}

    assert by_id["the-opinion"]["ruling"] is None
    assert by_id["the-opinion"]["error"] is None
    assert by_id["the-fact"]["error"] is None
    assert by_id["the-fact"]["ruling"]["verdict"] in VERDICTS
    assert written["meta"]["usage"]["searches"] >= 1
