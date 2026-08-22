# 01: Core classification

**What to build:** Given a single statement and its surrounding context, get back a fact/opinion classification with a confidence score, backed by a real LLM call routed through OpenRouter. This includes standing up the `uv` workspace and package that everything else in this feature builds on.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `devops-hackathon-aug-26/` exists as a `uv` workspace (root `pyproject.toml` with `[tool.uv.workspace]`, `members = ["packages/*"]`)
- [ ] `statement-classifier` package exists at `devops-hackathon-aug-26/packages/statement-classifier/` with its own `pyproject.toml`, installable via `uv`
- [ ] Pydantic models exist for a single statement (`statement`, `surroundingContext`) and its classification (`class: Literal["fact", "opinion"]`, `confidence: float`)
- [ ] A function classifies one statement via `ChatOpenAI` configured for OpenRouter (`base_url`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` read from environment variables) using `with_structured_output` against the classification schema
- [ ] `surroundingContext` is passed into the prompt as context for the statement being classified, not classified itself
- [ ] A unit test with a mocked chat model returns a correct `fact`/`opinion` classification with `confidence` in `[0, 1]` for at least one fact example and one opinion example
- [ ] No live network calls are made in tests
