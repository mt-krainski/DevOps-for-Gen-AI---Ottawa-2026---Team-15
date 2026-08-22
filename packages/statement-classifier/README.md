# statement-classifier

Labels each statement `fact` or `opinion` with a confidence score, from one LLM call per statement
routed through OpenRouter. It is one stage in a larger fact-checking pipeline: it does no web search
and reaches no verdict on whether a claim is true. `SPEC.md` carries the full design.

Its output is the input the `factchecker` package reads.

Two ways to call it:

- **`classify`** — you've already split the text into statements; this labels each one.
- **`classify-paragraph`** — you hand over a single paragraph; this splits it into statements (a
  second LLM call) and then labels each one, the same way `classify` does.

## Configuration

Three environment variables, read at the start of a run:

- `OPENROUTER_API_KEY` — required.
- `OPENROUTER_MODEL` — defaults to `anthropic/claude-sonnet-5`.
- `OPENROUTER_BASE_URL` — defaults to `https://openrouter.ai/api/v1`.

## Command line

```bash
statement-classifier classify [--input <path|->] [--output <path|->] [--concurrency <n>]
statement-classifier classify-paragraph [--input <path|->] [--output <path|->] [--concurrency <n>]
```

`--input` and `--output` both default to `-`, meaning stdin and stdout, so either command composes
in a shell pipeline:

```bash
echo '{"statements": [{"surroundingContext": "...", "statement": "..."}]}' \
  | statement-classifier classify

echo '{"paragraph": "..."}' | statement-classifier classify-paragraph
```

`--concurrency` sets the ceiling on classification LLM calls in flight at once. It defaults to 5.

### Example

```bash
echo '{"paragraph": "The prime minister said talks had taken place with the United States over the past year, and accelerated in recent days. He suggested his team had been skeptical, but optimistic a deal could be reached."}' \
  | uv run statement-classifier classify-paragraph
```

```json
{
  "statements": [
    {
      "statement": "The prime minister said talks had taken place with the United States over the past year",
      "classification": {
        "class": "fact",
        "confidence": 0.9
      },
      "error": null
    },
    {
      "statement": "and accelerated in recent days.",
      "classification": {
        "class": "fact",
        "confidence": 0.85
      },
      "error": null
    },
    {
      "statement": "He suggested his team had been skeptical,",
      "classification": {
        "class": "fact",
        "confidence": 0.75
      },
      "error": null
    },
    {
      "statement": "but optimistic a deal could be reached.",
      "classification": {
        "class": "opinion",
        "confidence": 0.85
      },
      "error": null
    }
  ]
}
```

### Exit codes

| Code | Meaning |
| ---- | ------- |
| `0`  | The batch succeeded. Individual statements may still carry an error. |
| `1`  | The input could not be read, the output could not be written, or (`classify-paragraph` only) segmentation failed. |
| `2`  | The input is not valid JSON, or does not match the input shape. |
| `3`  | The credential is missing or was rejected. |

A non-zero exit writes one JSON object to stderr: `{"code": ..., "message": ...}`. Note what `0`
does not promise — a per-statement failure is reported inside the payload, not by the exit code, so
a caller that cares must read the `error` field.

## `classify`: pre-split statements

### Input

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

### Output

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

## `classify-paragraph`: one paragraph, split then classified

### Input

```json
{ "paragraph": "Carney confirmed he was “reluctantly” adding tariffs that would add costs in some areas for Canadians, but insisted they were necessary to retaliate against United States President Donald Trump’s levies" }
```

### Output

The statements the paragraph was split into, in reading order. Each carries the whole paragraph as
its `surroundingContext`, because the paragraph is the only context a paragraph-mode caller
supplies. The shape is `classify`'s output, so either mode feeds the next stage:

```json
{
  "statements": [
    {
      "surroundingContext": "Carney confirmed he was “reluctantly” adding tariffs that would add costs in some areas for Canadians, but insisted they were necessary to retaliate against United States President Donald Trump’s levies",
      "statement": "Carney confirmed he was “reluctantly” adding tariffs that would add costs in some areas for Canadians",
      "classification": { "class": "fact", "confidence": 0.95 },
      "error": null
    },
    {
      "surroundingContext": "Carney confirmed he was “reluctantly” adding tariffs that would add costs in some areas for Canadians, but insisted they were necessary to retaliate against United States President Donald Trump’s levies",
      "statement": "but insisted they were necessary to retaliate against United States President Donald Trump’s levies",
      "classification": { "class": "opinion", "confidence": 0.95 },
      "error": null
    }
  ]
}
```

That handoff holds as long as every statement classified. A statement whose classification failed
carries `classification: null`, and the next stage rejects the whole payload over it.

## Errors

A failure on one statement's classification is isolated onto that statement. The batch keeps
going, and the sibling statements still classify.

| Code | Raised when |
| ---- | ----------- |
| `LLM_ERROR` | The classification call failed on every attempt. |
| `LLM_TIMEOUT` | The model did not answer inside the call timeout. |
| `PARSE_ERROR` | The model answered with something the classification schema rejects. |

These codes end the whole call instead, because nothing partial is worth returning:
`INVALID_INPUT`, `MISSING_API_KEY`, `AUTH_ERROR`, and `IO_ERROR`. `classify-paragraph` adds one
more batch-level code: `SEGMENTATION_ERROR`, when splitting the paragraph itself fails on every
attempt (or its output fails schema validation) — there's no per-item granularity to isolate that
onto, since nothing was extracted yet.

## Library

```python
from statement_classifier import classify_statements_sync, classify_paragraph_sync

output = classify_statements_sync(payload, concurrency=5)
paragraph_output = classify_paragraph_sync({"paragraph": "..."}, concurrency=5)
```

`classify_statements` and `classify_paragraph` are the async forms of the same calls. Both accept
either a typed input model or the dict it validates from, and both raise `ClassifierError` for a
batch-level failure. `classify_statements`/`classify_statements_sync` take `model=` to supply the
classification runnable; `classify_paragraph`/`classify_paragraph_sync` take `classifier_model=`
and `segmenter_model=` for the two runnables it calls. These are the seams the tests drive.

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
