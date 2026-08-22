# statement-classifier

Labels each extracted statement `fact` or `opinion` with a confidence score, from one LLM call per
statement routed through OpenRouter. It is one stage in a larger fact-checking pipeline: it does no
web search and reaches no verdict on whether a claim is true. `SPEC.md` carries the full design.

Its output is the input the `factchecker` package reads.

## Configuration

Three environment variables, read at the start of a run:

- `OPENROUTER_API_KEY` — required.
- `OPENROUTER_MODEL` — defaults to `anthropic/claude-sonnet-5`.
- `OPENROUTER_BASE_URL` — defaults to `https://openrouter.ai/api/v1`.

## Command line

```bash
statement-classifier classify [--input <path|->] [--output <path|->] [--concurrency <n>]
```

`--input` and `--output` both default to `-`, meaning stdin and stdout, so the command composes in a
shell pipeline:

```bash
echo '{"statements": [{"surroundingContext": "...", "statement": "..."}]}' \
  | statement-classifier classify
```

`--concurrency` sets the ceiling on LLM calls in flight at once. It defaults to 5.

### Exit codes

| Code | Meaning |
| ---- | ------- |
| `0`  | The batch succeeded. Individual statements may still carry an error. |
| `1`  | The input could not be read, or the output could not be written. |
| `2`  | The input is not valid JSON, or does not match the input shape. |
| `3`  | The credential is missing or was rejected. |

A non-zero exit writes one JSON object to stderr: `{"code": ..., "message": ...}`. Note what `0`
does not promise — a per-statement failure is reported inside the payload, not by the exit code, so
a caller that cares must read the `error` field.

## Input

```json
{
  "statements": [
    {
      "surroundingContext": "We are testing. This is a test. Test is now over.",
      "statement": "This is a test"
    }
  ]
}
```

`surroundingContext` is context, not a subject: it is what resolves an ambiguous reference in the
statement ("this", "it"), and it is never classified itself.

## Output

The same statements, in the order submitted, each carrying either a `classification` or an `error`:

```json
{
  "statements": [
    {
      "surroundingContext": "We are testing. This is a test. Test is now over.",
      "statement": "This is a test",
      "classification": { "class": "fact", "confidence": 0.82 },
      "error": null
    }
  ]
}
```

### Errors

A failure on one statement is isolated onto that statement. The batch keeps going, and the sibling
statements still classify.

| Code | Raised when |
| ---- | ----------- |
| `LLM_ERROR` | The call failed on every attempt. |
| `LLM_TIMEOUT` | The model did not answer inside the call timeout. |
| `PARSE_ERROR` | The model answered with something the schema rejects. |

Four codes end the whole batch instead, because nothing partial is worth returning:
`INVALID_INPUT`, `MISSING_API_KEY`, `AUTH_ERROR`, and `IO_ERROR`.

## Library

```python
from statement_classifier import classify_statements_sync

output = classify_statements_sync(payload, concurrency=5)
```

`classify_statements` is the async form of the same call. Both accept a `ClassifierInput` or the
dict it validates from, and both raise `ClassifierError` for a batch-level failure. Passing
`model=` supplies the runnable to call, which is the seam the tests drive.

## Development

Run both checks from `packages/statement-classifier/`:

```bash
uv run poe lint
uv run poe test
```

`lint` runs `ruff check` and `ruff format --check`. `test` runs the suite under coverage, and the
coverage report fails below 80 percent. Continuous integration runs the same two commands on every
pull request that touches this package.

No test calls a model. Every test drives the classifier through a fake runnable, so the suite needs
no credential and makes no network call.
