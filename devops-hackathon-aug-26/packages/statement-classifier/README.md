# statement-classifier

Classifies extracted statements as `fact` or `opinion` with a confidence score, via an LLM call routed through OpenRouter. One stage in a larger fact-checking pipeline; see `SPEC.md` for the full design.

## Configuration

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_MODEL` (default: `anthropic/claude-sonnet-5`)
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
