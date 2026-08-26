# fact-checker

Takes statements that an upstream classifier labelled `fact` or `opinion`, searches the web for each
factual one, and writes a ruling with cited sources. Opinions pass through untouched.

The tool reports what the evidence shows. It does not establish truth, and its verdict vocabulary —
`supported`, `refuted`, `mixed`, `unverifiable` — says so. `SPEC.md` carries the full design.

It reads the output of the `statement-classifier` package, and it is one stage in a pipeline: it
does no extraction and no classification of its own. It trusts the label it is given.

## Accepted limitation

**References are written by the model and are not checked against the retrieved text.** An excerpt
may be a paraphrase rather than a quotation. Read `source` as a pointer to follow, and never read a
citation here as a verified quotation.

## Configuration

Every setting is an environment variable, read once at the start of a run. A `.env` file is read
where one exists. `.env.example` lists every name with its value left blank; copy it to `.env`,
which `.gitignore` excludes.

| Variable | Default | What it sets |
| -------- | ------- | ------------ |
| `OPENROUTER_API_KEY` | **required** | The OpenRouter key both model calls use. |
| `OPENROUTER_MODEL` | `google/gemma-4-31b-it` | The model slug, recorded in `meta.model`. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | The gateway the model calls go to. |
| `BRIGHTDATA_API_TOKEN` | **required** | The Bright Data token, which rides in the endpoint URL. |
| `BRIGHTDATA_MCP_ENDPOINT` | `https://mcp.brightdata.com/mcp` | The hosted MCP server the two tools come from. |
| `FACT_CHECKER_CONCURRENCY` | `8` | How many statements are checked at once. |
| `FACT_CHECKER_TOOL_CALL_BUDGET` | `10` | How many tool calls one statement may spend. |
| `FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS` | `240` | How long one statement's check may run. |
| `FACT_CHECKER_SCRAPE_CHAR_LIMIT` | `100000` | Where a fetched page is cut before it reaches the model. |
| `LOG_LEVEL` | `INFO` | The level of the `fact_checker` logger. |

**A blank variable is not zero.** `.env.example` ships every name with an empty value, and a blank
numeric variable falls back to its default. The guards reject a non-positive concurrency, tool-call
budget or scrape limit — they never see the blank one.

**The timeout has to clear what the budget can spend.** A tool call may reasonably take 24 seconds,
so `FACT_CHECKER_STATEMENT_TIMEOUT_SECONDS` must be at least 24 times
`FACT_CHECKER_TOOL_CALL_BUDGET`. A timeout under that floor is rejected at start-up as
`INVALID_INPUT`: the timeout guards against a hang, and it must never be the limit that shapes
normal work. Raising the budget raises the timeout with it.

`LOG_LEVEL` takes `CRITICAL`, `ERROR`, `WARNING`, `INFO` or `DEBUG`. Anything else is reported as a
warning and the run continues at `INFO`. There is no `--verbose` flag; this variable is the level
control.

## Command line

```bash
fact-checker --input statements.json --output rulings.json
```

Both arguments are required, and both take a path. There is no stdin or stdout mode: the payload is
written only to `--output`, and every log record goes to stderr, so stdout stays clean.

At `INFO` the tool writes one line per statement, naming its id, its outcome, the elapsed time and
the tool calls it spent. At `DEBUG` it also writes one line per tool call and the traceback behind
any failure. The reason a run ended is logged at `CRITICAL`, so no level hides why a non-zero exit
code came back.

## Input

```json
{
  "statements": [
    {
      "id": "s1",
      "surroundingContext": "A note on kettles. Water boils at 100 C at sea level. That is why the kettle clicks.",
      "statement": "Water boils at 100 C at sea level",
      "classification": { "class": "fact", "confidence": 0.7 }
    }
  ]
}
```

- `id` is optional. Where it is absent the tool assigns `s1`, `s2` and so on in input order, and an
  assigned id means nothing outside its own run. Two statements under one id fail the whole batch.
- `surroundingContext` is required. The agent uses it to turn a claim that leans on its surroundings
  into a claim it can search — to resolve a pronoun, a date or a place. It is never ruled on itself.
- `classification.class` is `fact` or `opinion`. Any other value fails validation, and the message
  names the value that failed.
- `classification.confidence` is echoed to the output and never decides whether a statement is
  checked. A statement labelled `fact` is checked whatever its confidence, because a doubtful label
  is where a check is most informative.

## Output

One entry for every input statement, in input order however the checks finish, each repeating the
input fields and adding `ruling` and `error`. Both keys are always present and either may be null,
so every consumer sees one shape.

```json
{
  "meta": {
    "model": "google/gemma-4-31b-it",
    "startedAt": "2026-08-22T14:03:11Z",
    "finishedAt": "2026-08-22T14:05:47Z",
    "counts": { "total": 2, "checked": 1, "skipped": 1, "failed": 0 },
    "usage": { "promptTokens": 8412, "completionTokens": 377, "searches": 3 }
  },
  "statements": [
    {
      "id": "s1",
      "surroundingContext": "A note on kettles. Water boils at 100 C at sea level. That is why the kettle clicks.",
      "statement": "Water boils at 100 C at sea level",
      "classification": { "class": "fact", "confidence": 0.7 },
      "ruling": {
        "verdict": "supported",
        "confidence": 0.92,
        "justification": "At standard pressure water boils at 100 C [1].",
        "references": [
          {
            "id": "1",
            "source": "https://example.org/boiling-point",
            "excerpt": "At 1 atm, water boils at 100 C"
          }
        ]
      },
      "error": null
    },
    {
      "id": "s2",
      "surroundingContext": "The chapter closed on the author's own preferences.",
      "statement": "Boiling water is the most satisfying way to cook pasta",
      "classification": { "class": "opinion", "confidence": 0.9 },
      "ruling": null,
      "error": null
    }
  ]
}
```

### Three distinguishable cases

| Case | `ruling` | `error` | What happened |
| ---- | -------- | ------- | ------------- |
| Opinion | `null` | `null` | The statement was labelled `opinion`. The agent never ran. |
| Checked | the verdict | `null` | The agent searched and ruled. |
| Failed | `null` | the code and message | The check failed, and the rest of the batch carried on. |

Both keys null therefore means "skipped", never "something went wrong quietly".

### What `meta` records

`meta` describes the run, not the statement. `counts` reports `total`, `checked`, `skipped` and
`failed`, and `usage` reports the tokens both models consumed plus `searches`.

**`searches` counts attempts, not distinct queries.** Every search invocation that reached Bright
Data increments it, so a search retried after a transient failure counts each attempt. A search
answered from the run's cache counts none.

`meta` carries no cost estimate: published prices go stale, and a stale number is worse than none.

## The four verdicts

| Verdict | Meaning |
| ------- | ------- |
| `supported` | The evidence backs the claim. |
| `refuted` | The evidence contradicts the claim. |
| `mixed` | The claim is partly right, or the sources disagree with each other. |
| `unverifiable` | The search ran and the evidence does not settle the claim. |

**`unverifiable` is a finding, not a failure.** A consumer that treats it as an error misreads the
output. A statement that could not be checked at all carries an `error` instead, and its `ruling` is
null.

**`ruling.confidence` measures trust in the verdict, not the truth of the statement.** An
`unverifiable` verdict at high confidence says the tool is sure the claim cannot be settled this way.

A statement whose agent spends the whole tool-call budget still gets a ruling, on the evidence it
holds by then. A spent budget is not a failure, and very often the honest verdict is `unverifiable`.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| `0` | A payload was written. Statements inside it may still carry errors. |
| `1` | An unexpected crash, or the Bright Data tools could not be loaded. |
| `2` | The input could not be read, or it failed the contract. |
| `3` | A credential was missing or was rejected. |
| `4` | The payload was built and could not be written. |

A non-zero exit writes one JSON object to stderr: `{"code": ..., "message": ...}`. Note what `0`
does not promise — a per-statement failure is reported inside the payload, not by the exit code, so
a caller that cares must read the `error` field.

## Errors

A per-statement failure is isolated onto that statement: `ruling` is null, `error` names what went
wrong, and every sibling statement still gets checked.

| Code | Reported when |
| ---- | ------------- |
| `TOOL_ERROR` | A tool call failed, was asked for by a name this run does not offer, or came back with no text. |
| `AGENT_ERROR` | A model call failed on every attempt, or a call failed in a way the tool layer does not name. |
| `TIMEOUT` | The check did not finish inside the statement timeout. |
| `PARSE_ERROR` | The model answered with something the ruling schema rejects. |

These end the whole run instead, because nothing partial is worth returning:

| Code | Reported when | Exit |
| ---- | ------------- | ---- |
| `INVALID_INPUT` | The payload is not valid JSON, fails the contract, repeats an id, or a numeric setting is unusable. | `2` |
| `IO_ERROR` | The input could not be read, or the output could not be written. | `2` / `4` |
| `MISSING_CREDENTIAL` | `OPENROUTER_API_KEY` or `BRIGHTDATA_API_TOKEN` is unset or blank. | `3` |
| `AUTH_ERROR` | A credential was rejected. Every statement would fail the same way, so the run stops. | `3` |
| `TOOL_ERROR` | The Bright Data server could not be reached, or does not offer both tools. | `1` |
| `AGENT_ERROR` | An unexpected crash. The exit code is the machine-readable signal. | `1` |

`INVALID_INPUT`, `MISSING_CREDENTIAL`, `AUTH_ERROR` and `TOOL_ERROR` reach a library caller as a
`CheckError`. `IO_ERROR` and `AGENT_ERROR` belong to the command line: reading and writing files is
its job, and the crash barrier is where an unexpected exception stops.

The Bright Data token travels inside the endpoint URL. Every message this package reports, logs or
publishes passes through a scrub that puts `***` in the token's place, and the settings objects keep
both credentials out of their own `repr`.

## Library

```python
from fact_checker import check_statements

output = await check_statements(payload)
```

The entry point is asynchronous and there is no synchronous wrapper. It accepts either a
`CheckerInput` or the mapping that validates into one, returns a `CheckerOutput`, and raises
`CheckError` — carrying an `ErrorCode` — for a run-level failure. `config=` supplies settings
instead of reading the environment, and `runtime=` supplies the connected toolkit and the two
models. Those are the seams the tests drive.

The library logs to the `fact_checker` logger and configures nothing itself. An embedding
application owns its own handlers; only the command line attaches one, and it leaves propagation on.

## Development

Run both checks from `packages/fact-checker/`:

```bash
uv run poe lint
uv run poe test
```

`lint` runs `ruff check` and `ruff format --check`. `test` runs the suite under coverage, and the
coverage report fails below 80 percent. Continuous integration runs the same two commands on every
pull request that touches this package.

**No test in that suite calls a model or a search.** Every test drives both the model and the MCP
toolkit through fakes, so the suite needs no credential and makes no network call. That is why the
CI job carries no secret.

One credentialed end-to-end test exists, marked `integration` and deselected by `poe test`. CI never
runs it. To run it by hand, with `OPENROUTER_API_KEY` and `BRIGHTDATA_API_TOKEN` set:

```bash
uv run poe test-integration
```

### The LLM quality suite

Verdict quality is measured separately, by a promptfoo suite of about twenty cases in
`promptfoo/`, spread across the four verdicts and including opinions that must pass through
unchecked. It is wired to nothing and is run by hand when a prompt changes. Every case spends a real
OpenRouter key and a real Bright Data token.

promptfoo needs Node 22.22.0 or newer. Run it from `packages/fact-checker/`:

```bash
PROMPTFOO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" \
  npx promptfoo@0.122.0 eval -c promptfoo/promptfooconfig.yaml
```

`PROMPTFOO_PYTHON` points promptfoo at this package's own environment; without it the provider
cannot import `fact_checker`. Nothing is added to a `package.json`.

promptfoo caches to `~/.promptfoo/cache`. `--no-cache` bypasses the cache for a run, and
`npx promptfoo@0.122.0 cache clear` empties it. Budget for a rerun costing what the first run cost
until you have measured otherwise.

A case whose assertion fails is a finding about the prompt or the model. It is not a reason to
loosen the assertion.
