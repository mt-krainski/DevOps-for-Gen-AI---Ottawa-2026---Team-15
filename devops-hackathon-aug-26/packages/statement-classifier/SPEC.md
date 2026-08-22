# statement-classifier — Spec

Package: `devops-hackathon-aug-26/packages/statement-classifier/`
Part of the `devops-hackathon-aug-26` hackathon monorepo (a `uv` workspace). This package is one stage in a larger fact-checking pipeline being built across the team; it does **not** perform web search or truthiness verification itself.

## Problem Statement

Downstream in the fact-checking pipeline, a separate tool needs to verify factual statements against the web. Before it can run, every statement extracted from a piece of text needs to be labeled as a `fact` (a checkable claim) or an `opinion` (not checkable), each with a confidence score. Doing this labeling by hand doesn't scale, and folding it into the verification tool itself would couple two independently useful, independently testable stages together.

## Solution

A standalone Python package, `statement-classifier`, that takes a batch of statements (each with its surrounding context) and returns the same batch with a `classification` (`class` + `confidence`) attached to each one, produced by an LLM call per statement via LangChain, routed through OpenRouter as the model gateway. It's usable both as an importable Python function and as a CLI command, so teammates can either call it as a library from their own pipeline code or shell out to it. It fails gracefully: one statement's classification failing never aborts the batch.

## User Stories

1. As the developer of the downstream fact-verification tool, I want to import `statement-classifier` as a Python function, so that I can feed its output directly into my own tool without shelling out.
2. As a hackathon teammate wiring the pipeline together via scripts, I want a CLI command that reads a JSON file and writes a JSON file, so that I can compose pipeline stages without writing Python glue code.
3. As a teammate testing the pipeline manually, I want to pipe JSON through the CLI via stdin/stdout, so that I can quickly try it without creating files.
4. As any caller of this tool, I want each statement classified independently, so that a claim can be judged fact vs. opinion without other statements in the batch influencing it.
5. As any caller of this tool, I want a confidence score alongside the class, so that downstream stages can decide whether to trust a borderline classification.
6. As any caller of this tool, I want the `surroundingContext` field used as context for classification (not classified itself), so that ambiguous statements ("This is a test") get classified correctly using the sentences around them.
7. As any caller of this tool, I want one failing statement (LLM error, timeout, bad model output) to not crash the whole batch, so that a single bad statement doesn't cost me the classifications of everything else I submitted.
8. As any caller of this tool, I want a clear, typed error (with a code) when a whole batch fails outright (bad input shape, missing/invalid API key), so that I can distinguish "my request was malformed" from "one statement had a transient LLM issue."
9. As a caller running many statements at once, I want them classified concurrently, so that a large batch doesn't take one-LLM-call-worth-of-latency times the number of statements.
10. As a teammate integrating this via shell scripts, I want the CLI to exit non-zero only on batch-level failure (not on individual per-statement errors), so that my scripts can reliably detect "the whole call failed" vs. "some individual items need attention."
11. As a developer configuring this package, I want the OpenRouter API key and model to be set via environment variables, so that I don't have to hardcode credentials or model choice.
12. As a maintainer of the hackathon monorepo, I want this package to be a `uv` workspace member under `devops-hackathon-aug-26/packages/`, so that it shares tooling/lockfile conventions with the other subpackages teammates are building alongside it.
13. As a future maintainer, I want the input/output shapes defined as typed schemas (Pydantic), so that the contract between this stage and the rest of the pipeline is explicit and validated, not just documented in prose.

## Implementation Decisions

**Workspace & package layout**
- `devops-hackathon-aug-26/` is the monorepo root: a `uv` workspace (`[tool.uv.workspace]` with `members = ["packages/*"]`), shared across all teammates' subpackages.
- This package lives at `devops-hackathon-aug-26/packages/statement-classifier/`, as its own `uv` workspace member with its own `pyproject.toml`.
- Distribution/import name: `statement_classifier`. CLI command name: `statement-classifier`.

**Scope boundary**
- Input: statements with `surroundingContext`, **no** classification yet.
- Output: same statements with `classification: {class, confidence}` attached.
- No web search, no truthiness ruling, no references — that's a separate downstream package this one's output feeds into.

**Data shapes (Pydantic models)**

Input:
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

Output:
```json
{
  "statements": [
    {
      "surroundingContext": "We are testing. This is a test. Test is now over.",
      "statement": "This is a test",
      "classification": {
        "class": "fact",
        "confidence": 0.7
      },
      "error": null
    }
  ]
}
```

On a per-statement failure, that item's `classification` is `null` and `error` is populated:
```json
{
  "surroundingContext": "...",
  "statement": "...",
  "classification": null,
  "error": { "code": "LLM_TIMEOUT", "message": "..." }
}
```

- `classification.class`: exactly two values for MVP — `"fact"` | `"opinion"`. (Documented as a two-value enum; expanding this later, e.g. adding `not-a-claim`, is a natural but explicitly out-of-scope-for-now extension.)
- `classification.confidence`: float in `[0, 1]`, the model's self-reported confidence, obtained via structured output (not parsed from free text).

**LLM integration**
- LangChain's `ChatOpenAI`, pointed at OpenRouter's OpenAI-compatible endpoint (`base_url="https://openrouter.ai/api/v1"`), since OpenRouter is a drop-in OpenAI-API-compatible gateway — this is the standard LangChain↔OpenRouter integration pattern, no special LangChain OpenRouter package needed.
- Structured output enforced via LangChain's `with_structured_output(...)` against a Pydantic schema (`class: Literal["fact", "opinion"]`, `confidence: float`), rather than parsing free-text completions — avoids brittle regex/JSON-extraction from prose.
- One LLM call per statement (not one call for the whole batch): keeps structured-output parsing simple and per-item, and is what makes per-statement error isolation and concurrency straightforward. The `surroundingContext` is included in that statement's prompt so the model has the context it needs without seeing (or being influenced by classifying) sibling statements.
- Config via environment variables:
  - `OPENROUTER_API_KEY` (required)
  - `OPENROUTER_MODEL` (default: a placeholder, e.g. `anthropic/claude-sonnet-5` — the team should pick/tune this during the hackathon based on cost and accuracy on real inputs; trivially overridable)
  - `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`, overridable for testing against a mock endpoint)

**Concurrency**
- Statements are classified concurrently via `asyncio.gather` bounded by a semaphore (default concurrency: configurable, e.g. `CLASSIFIER_CONCURRENCY` env var or a function/CLI parameter, defaulting to something modest like 5) — independent per-statement LLM calls with no shared state.

**Error handling**
- Per-statement errors (LLM call failure after a small bounded number of retries with backoff, timeout, or the model returning output that fails schema validation) are isolated: that statement's `classification` is `null`, with `error: {code, message}` populated, and the rest of the batch proceeds normally.
- Batch-level errors (malformed input JSON/schema, missing or invalid `OPENROUTER_API_KEY`, auth failure) raise a typed exception (`ClassifierError`) carrying an error code, and abort the whole call — nothing partial is returned.
- A small, explicit error code enum, shared between the per-statement `error.code` field and the batch-level exception:
  - `INVALID_INPUT` (batch-level)
  - `MISSING_API_KEY` / `AUTH_ERROR` (batch-level)
  - `LLM_ERROR` (per-statement, call failed after retries exhausted)
  - `LLM_TIMEOUT` (per-statement)
  - `PARSE_ERROR` (per-statement, model output didn't satisfy the structured schema after retries)

**Python API**
```python
async def classify_statements(
    input: ClassifierInput, *, concurrency: int = 5
) -> ClassifierOutput: ...
```
Plus a sync wrapper (`classify_statements_sync`) for callers not already in an async context.

**CLI**
- `statement-classifier classify --input <path|-> --output <path|->` — reads/writes JSON; `-` (or omission) means stdin/stdout, so it composes in shell pipelines.
- `--concurrency <n>` optional override.
- Exit codes: `0` success (even if individual statements carry per-item errors — that's still a successful batch response), non-zero only for batch-level failure. Suggested mapping: `2` invalid input, `3` auth/config error, `1` unexpected/internal error. On non-zero exit, a JSON error object (`{code, message}`) is written to stderr.

## Testing Decisions

- Tests mock the LangChain chat model boundary (the `ChatOpenAI`/`with_structured_output` runnable) — no live network calls to OpenRouter in CI. This is the seam: everything on either side of "call the LLM, get back a `{class, confidence}`" is deterministic and testable without a real model.
- Test only external behavior (input → output shape and content), not internal call structure.
- Cases to cover:
  - Mixed batch of facts and opinions classifies correctly given a mocked model response.
  - Empty `statements` list returns an empty result without error.
  - One statement's mocked LLM call raising/timing out results in that item alone getting `classification: null` + populated `error`, while sibling statements still classify successfully (proves isolation).
  - Malformed input (missing `statement` field, wrong types) raises `ClassifierError(code="INVALID_INPUT")` at the batch level, before any LLM calls are made.
  - Missing `OPENROUTER_API_KEY` raises `ClassifierError(code="MISSING_API_KEY")` before any LLM calls are made.
  - CLI: valid input file → output file written, exit code `0`.
  - CLI: malformed input JSON → non-zero exit code, JSON error object on stderr.
- No prior art exists yet in this monorepo (this is the first subpackage) — this package's test setup (`pytest` + `pytest-asyncio`, LLM boundary mocked, typed Pydantic fixtures for input/output) is intended as the convention other hackathon subpackages in `devops-hackathon-aug-26/packages/` follow.

## Out of Scope

- Web search / evidence retrieval and truthiness verification (`ruling`, `references`, etc.) — a separate downstream subpackage.
- More than two classification classes (e.g. `not-a-claim`, `unverifiable`) — possible future extension.
- An HTTP API / server — library + CLI only for now.
- Batching multiple statements into a single LLM call.
- Persistence, database, or UI of any kind.
- Non-English input handling / localization.
- A model-eval or accuracy-benchmarking harness for the classifier itself.
- Rate-limit-aware backpressure beyond basic per-call retry with backoff.

## Further Notes

- The `OPENROUTER_MODEL` default in this spec is a placeholder; pick the actual default during implementation based on quick cost/accuracy testing against real sample statements, since OpenRouter model availability and pricing shift often.
- The workspace layout (`devops-hackathon-aug-26/packages/<name>/` as `uv` workspace members) anticipates sibling packages — e.g. a statement-extraction stage and the fact-verification stage — sharing the same root lockfile and tooling conventions.
- This package intentionally does not know anything about the fact-verification stage's needs beyond the documented output shape; keep that boundary as the sole integration contract between the two.
