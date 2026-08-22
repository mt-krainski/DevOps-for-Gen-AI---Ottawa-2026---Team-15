"""promptfoo's Python provider: one case in, one ruling out.

promptfoo spawns this file once per case, so each case opens its own connection and
starts with an empty cache. That costs a little at nineteen cases and keeps them
independent: no case can be made cheaper, or easier, by the case that ran before it.

The interpreter promptfoo spawns has to be the package's own, because this imports
`factchecker` and everything under it. `promptfooconfig.yaml` names it.
"""

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

from factchecker.agent import AgentChecker
from factchecker.cache import RunCache
from factchecker.config import build_model, load_settings
from factchecker.models import Classification, IdentifiedStatement
from factchecker.tools import instrument, load_tools

# The same file the command reads, named from this file rather than searched for.
# promptfoo decides the working directory, so a search would find whatever sat above
# wherever the suite was started from.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Every case checks one statement, and its identifier reaches nothing a case asserts
# on. The agent sees it, so it is a word rather than a number.
_CASE_ID = "case"

load_dotenv(dotenv_path=ENV_FILE, override=False)


async def call_api(
    prompt: str, options: Mapping[str, object], context: Mapping[str, object]
) -> dict[str, str]:
    """Check one case's statement, and hand promptfoo back the ruling.

    Args:
        prompt: The prompt promptfoo rendered. This provider writes its own prompts
            from the statement and its context, so the rendered one is not read.
        options: The provider's own configuration. Nothing here reads it.
        context: What promptfoo knows about this case. `vars` holds the case.

    Returns:
        A mapping carrying `output`, which promptfoo requires on every return. A
        case that could not be checked carries an empty output beside an `error`,
        which fails that case and leaves the rest of the suite to run.
    """
    try:
        ruling = await _ruled(_statement(context.get("vars")))
    except Exception as failure:  # noqa: BLE001 — one case's failure is not the suite's
        return {"output": "", "error": f"{type(failure).__name__}: {failure}"}
    return {"output": ruling}


def _statement(variables: object) -> IdentifiedStatement:
    """Read one case's variables as the statement the agent is asked to check.

    Every case is a factual claim, so the classification is written here rather than
    carried by each case. Confidence is the classifier's, and no classifier ran.
    """
    if not isinstance(variables, Mapping):
        raise TypeError(
            f"the case has no variables, it has a {type(variables).__name__}"
        )
    return IdentifiedStatement(
        id=_CASE_ID,
        surrounding_context=str(variables["surroundingContext"]),
        statement=str(variables["statement"]),
        classification=Classification(class_="fact", confidence=1.0),
    )


async def _ruled(statement: IdentifiedStatement) -> str:
    """Open the connection, check the statement over it, and close it again.

    This is the command's own path with the orchestrator left out: one statement
    needs no concurrency bound, and promptfoo applies the timeout.
    """
    settings = load_settings(os.environ)
    model = build_model(settings)
    tools, release = await load_tools(settings.mcp_endpoint)
    try:
        checker = AgentChecker(
            model,
            instrument(tools, RunCache(), settings, asyncio.sleep),
            settings,
        )
        outcome = await checker.check(statement)
    finally:
        await release()
    return outcome.ruling.model_dump_json()
