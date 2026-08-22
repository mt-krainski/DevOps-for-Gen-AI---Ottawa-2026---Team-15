# statement-classifier

Classifies extracted statements as `fact` or `opinion` with a confidence score, via an LLM call routed through OpenRouter. One stage in a larger fact-checking pipeline; see `SPEC.md` for the full design.

## Configuration

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_MODEL` (default: `anthropic/claude-sonnet-5`)
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)

## CLI

```
statement-classifier classify [--input <path|->] [--output <path|->] [--concurrency <n>]
```

- `--input` / `--output` default to `-`, meaning stdin/stdout, so it composes in shell pipelines.
- `--concurrency` overrides the default concurrency (5).
- Exit code `0` on a successful batch call, even if individual statements carry per-item errors.
- Non-zero exit codes are for batch-level failure only: `2` invalid input, `3` auth/config error, `1` unexpected/internal error. On non-zero exit, a JSON error object (`{code, message}`) is written to stderr.

```
echo '{"statements": [{"surroundingContext": "...", "statement": "..."}]}' \
  | statement-classifier classify
```
