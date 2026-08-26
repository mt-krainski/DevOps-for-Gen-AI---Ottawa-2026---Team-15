"""The promptfoo provider: one test case in, one checked statement out."""

from typing import Any

from dotenv import find_dotenv, load_dotenv

from fact_checker import CheckedStatement, CheckerOutput, check_statements

# promptfoo runs this file under a wrapper of its own, from a working directory
# that is not this package's, so the `.env` the command line reads is located
# from this file rather than from the current directory.
load_dotenv(find_dotenv())


async def call_api(
    prompt: str, options: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Check the case's one statement and report the entry the package wrote.

    Args:
        prompt: promptfoo's rendered prompt. This provider ignores it: the case
            is carried by `vars`, and the prompts the run sends are the
            package's own.
        options: promptfoo's provider configuration. Unused.
        context: The case, whose `vars` hold `statement`, `surroundingContext`,
            `class` and `confidence`.

    Returns:
        The whole checked entry as a JSON string under `output`, beside the
        run's token usage, or the failure under `error` where no entry was
        written.
    """
    variables = context.get("vars", {})
    try:
        output = await check_statements(_one_statement_payload(variables))
    except Exception as exc:  # noqa: BLE001 — promptfoo reports, it does not raise
        return {"error": f"{type(exc).__name__}: {exc}"}
    return _reported(output)


def _one_statement_payload(variables: dict[str, Any]) -> dict[str, Any]:
    return {
        "statements": [
            {
                "surroundingContext": variables["surroundingContext"],
                "statement": variables["statement"],
                "classification": {
                    "class": variables["class"],
                    "confidence": variables["confidence"],
                },
            }
        ]
    }


def _reported(output: CheckerOutput) -> dict[str, Any]:
    entry: CheckedStatement = output.statements[0]
    usage = output.meta.usage
    return {
        "output": entry.model_dump_json(),
        "tokenUsage": {
            "total": usage.prompt_tokens + usage.completion_tokens,
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
        },
    }
