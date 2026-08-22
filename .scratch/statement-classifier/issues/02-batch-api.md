# 02: Batch API

**What to build:** Given a full batch of statements, get back the full batch with a classification attached to each one, classified concurrently, where one bad statement never takes down the rest of the batch — only a malformed request or missing credentials fails the whole call.

**Blocked by:** 01: Core classification

**Status:** ready-for-agent

- [ ] `ClassifierInput` / `ClassifierOutput` Pydantic models exist matching the spec's JSON shape (a `statements` list, each item carrying `surroundingContext`, `statement`, `classification` (nullable), `error` (nullable))
- [ ] `async def classify_statements(input: ClassifierInput, *, concurrency: int = 5) -> ClassifierOutput` classifies every statement in the batch concurrently, bounded by a semaphore
- [ ] A sync wrapper (`classify_statements_sync`) is available for callers not already in an async context
- [ ] A per-statement LLM failure (timeout, error after retries exhausted, or output that fails schema validation) results in that item's `classification: null` and `error: {code, message}` populated (`LLM_ERROR` / `LLM_TIMEOUT` / `PARSE_ERROR`), while sibling statements in the same batch still classify successfully
- [ ] Malformed input (missing `statement` field, wrong types) raises `ClassifierError(code="INVALID_INPUT")` before any LLM calls are made
- [ ] Missing or invalid `OPENROUTER_API_KEY` raises `ClassifierError(code="MISSING_API_KEY")` (or `AUTH_ERROR`) before any LLM calls are made
- [ ] Unit tests cover: a mixed fact/opinion batch, an empty `statements` list, one forced per-statement failure alongside successful siblings, malformed input, and a missing API key — all against a mocked chat model, no live network calls
