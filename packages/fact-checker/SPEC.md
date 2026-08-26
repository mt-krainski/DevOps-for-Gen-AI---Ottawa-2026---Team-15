# fact-checker — Spec

Package: `packages/fact-checker/`

Part of the team's hackathon monorepo. This package is stage two of the fact-checking pipeline: it
reads the classifier's output and rules on each factual statement against public web evidence. It
does **not** extract statements from a document, and it does not classify them.

## Problem Statement

An upstream classifier can say which sentences in a piece of text carry a checkable claim. Knowing
that is not the same as knowing what the public evidence says about each one. Doing that reading by
hand does not scale: every claim needs its own searches, its own pages, and its own judgement about
whether the sources actually settle it.

The obvious failure mode is worse than doing nothing. A tool that answers "true" or "false" invites
a reader to stop reading. What a person standing behind generated text needs is the evidence, a
verdict that refuses to overclaim, and a plain statement of the cases the evidence does not settle.

## Solution

A standalone Python package, `fact-checker`, that takes a batch of classified statements and returns
the same batch with a `ruling` or an `error` attached to each one. Each factual statement gets its
own agent run: the agent searches the web, reads the pages it chooses, and rules with one of four
verdicts and the references behind it. Opinions are never checked and pass through untouched.

It is both an importable async function and a command-line tool, so a teammate can call it from
their own pipeline code or shell out to it. It fails gracefully: one statement's failure never
aborts the batch, and a batch of forty-nine rulings and one error is a successful run reporting an
honest fact.

The vocabulary is the design. `unverifiable` is a first-class finding rather than an error, and
`confidence` measures trust in the verdict rather than the truth of the claim.

## Contract

### Input

```json
{
  "statements": [
    {
      "id": "s1",
      "surroundingContext": "We are testing. This is a test. Test is now over.",
      "statement": "This is a test",
      "classification": { "class": "fact", "confidence": 0.7 }
    }
  ]
}
```

- `id` is optional. Where it is absent the tool assigns `s1`, `s2` and so on in input order, and an
  assigned id means nothing outside its own run. A supplied id passes through unchanged. Two
  statements under one id fail the whole batch: an ambiguous output helps no consumer.
- `surroundingContext` is required. The agent uses it to turn a claim that depends on its
  surroundings into a claim that can be searched. It is never ruled on itself.
- `classification.class` is `fact` or `opinion`. Any other value fails validation, and the message
  names the value that failed. A classifier that grows a third label must be met with a deliberate
  change here, not with statements passing through unchecked.
- `classification.confidence` is echoed to the output and never decides whether a statement is
  checked. A statement labelled `fact` is checked whatever its confidence, because a doubtful label
  is where a check is most informative.

### Output

One entry per input statement, in input order, repeating the input fields and adding `ruling` and
`error`. Both keys are always present and either may be null, so every consumer sees one shape.

```json
{
  "meta": {
    "model": "google/gemma-4-31b-it",
    "startedAt": "2026-08-22T14:03:11Z",
    "finishedAt": "2026-08-22T14:05:47Z",
    "counts": { "total": 50, "checked": 31, "skipped": 19, "failed": 0 },
    "usage": { "promptTokens": 184203, "completionTokens": 9877, "searches": 74 }
  },
  "statements": [
    {
      "id": "s1",
      "surroundingContext": "...",
      "statement": "...",
      "classification": { "class": "fact", "confidence": 0.7 },
      "ruling": {
        "verdict": "supported",
        "confidence": 0.92,
        "justification": "At standard pressure water boils at 100 C [1].",
        "references": [
          { "id": "1", "source": "https://...", "excerpt": "At 1 atm, water boils at 100 C" }
        ]
      },
      "error": null
    }
  ]
}
```

Three cases are distinguishable, and that is deliberate:

- **Opinion.** `ruling` and `error` are both null. The agent never ran.
- **Checked.** `ruling` holds the verdict.
- **Failed.** `error` names what went wrong and `ruling` is null.

`meta` records the run, not the statement. `searches` counts search invocations that reached Bright
Data, so a search retried after a transient failure counts each attempt and a search answered from
the run's cache counts none. `meta` carries no cost estimate: published prices go stale, and a stale
number is worse than none.

### The four verdicts

- `supported` — the evidence backs the claim.
- `refuted` — the evidence contradicts the claim.
- `mixed` — the claim is partly right, or the sources disagree with each other.
- `unverifiable` — the search ran and the evidence does not settle the claim.

`unverifiable` is a finding. A consumer that treats it as a failure misreads the output. A statement
that could not be checked at all carries an `error` instead.

`confidence` measures trust in the verdict, not the truth of the statement. An `unverifiable`
verdict at high confidence says the tool is sure the claim cannot be settled this way.

## Implementation Decisions

**Package layout**

- The package lives at `packages/fact-checker/`, alongside the other stages. It is self-contained:
  its own `pyproject.toml`, its own `uv.lock`, its own `.python-version`, and its own path-filtered
  CI job. A teammate's change never queues, or fails, its checks.
- Distribution and import name: `fact_checker`. Command name: `fact-checker`.
- `ruff` for lint and format, `pytest` under `coverage` for tests, both run through `poe`. The green
  bar is `poe lint` and `poe test`, and the coverage report fails below 80 percent.
- JSON on the wire is camelCase and Python is snake_case. Pydantic's `to_camel` alias generator
  bridges the two, configured the same way the classifier configures it, so the two stages agree on
  the shape of a field name without sharing code.

**The checking loop**

- One agent run per factual statement, and runs are independent of each other. A statement's failure
  reaches one entry in the output and nothing else.
- Two model calls with different jobs: a tool-bound model that decides what to fetch next, and a
  structured-output model that turns the finished transcript into a `Ruling`. Splitting them keeps
  the ruling schema off every turn of the search loop, and keeps a parse failure attributable.
- The tool-call budget is ten calls per statement, shared between the two tools, and the agent
  decides the mix. Every message states how many calls remain, so the agent plans against the number
  rather than being cut off by it. When the budget runs out the agent rules on the evidence it
  holds, and that ruling is legitimate — very often `unverifiable`.
- The system prompt states that text the tools return is retrieved web content: evidence to weigh,
  never instruction to follow. That is a statement of intent, not an injection control, and the
  package claims no more than that.

**Tools**

- Bright Data's hosted MCP server supplies both tools, reached with `langchain-mcp-adapters`. The
  client filters what the server offers down to exactly two: `search_engine`, which returns each
  result's URL, title and description, and `scrape_as_markdown`, which returns one page as markdown.
  A server that does not offer both ends the run rather than starting it half-equipped.
- The adapter is built with `handle_tool_errors=False`. Under the default a tool failure comes back
  as content, and every layer above would read the failure text as evidence.
- The result count is not adjustable: the search tool takes a query, an engine and a pagination
  cursor, and Bright Data withdrew the count parameter.
- A fetched page is cut at 100,000 characters and the cut is marked, so the agent knows it read a
  fragment. This guards against the pathological page that would overflow the model's context and
  kill a run that was working. It is not a content control.

**Model access**

- OpenRouter is the gateway, reached through LangChain's `ChatOpenAI` against its OpenAI-compatible
  endpoint. The model is configuration, and the default is `google/gemma-4-31b-it`.
- The default names an explicit version rather than a `-latest` alias, so an evaluation can tell a
  prompt regression from a model swap underneath it.
- An agent loop punishes a weak model twice: it searches vaguely and needs more turns, and it
  mistakes a source that shares vocabulary with a claim for a source that confirms it. The second
  failure is the one this tool exists to avoid, so the model is worth tuning.

**Concurrency and ordering**

- Statements run concurrently under asyncio, bounded by a semaphore that defaults to eight. The work
  is almost all waiting on two networked services, which is what asyncio suits.
- Output order follows input order whatever order the runs finish in.
- The public entry point is asynchronous, and there is no synchronous wrapper. A caller that needs
  one owns its own event-loop policy.

**Caching**

- One run shares a cache keyed by search arguments and by fetched URL. Statements drawn from one
  document search for overlapping things, so the cache spares the second and third statement a call
  the first already paid for.
- The cache shares fetched material, never reasoning: two statements that read the same page still
  reach their own verdicts.
- It is single-flight, so concurrent statements wanting the same page wait on one call rather than
  making three. A statement that times out cancels its own work without cancelling that shared call,
  and a failed call is not kept, so a later statement may try again.

**Errors**

- A per-statement failure produces an `error` entry and the run continues. The codes are
  `TOOL_ERROR`, `AGENT_ERROR`, `TIMEOUT` and `PARSE_ERROR`.
- A statement that exceeds 240 seconds fails with `TIMEOUT`. The timeout catches a hang; it is never
  the constraint that shapes normal work. The tool-call budget is the deliberate limit, so the
  timeout is rejected at start-up unless it clears 24 seconds for every call the budget allows.
  Raising the budget raises the timeout with it.
- Transient failures are retried three times with exponential backoff and jitter. A transient
  failure is a 429, a 5xx, or a dropped connection. A 400 is not retried: nothing about waiting
  makes a malformed request valid. The MCP client stack runs on task groups, so a failure can arrive
  wrapped in an `ExceptionGroup` and the classification looks inside one.
- **An authentication failure ends the run.** A rejected credential fails every statement the same
  way, so fifty per-statement errors would cost four minutes to say what one error says in three
  seconds.
- A run-level failure returns nothing partial. The library raises `CheckError` carrying an
  `ErrorCode` — `INVALID_INPUT`, `MISSING_CREDENTIAL`, `AUTH_ERROR` or `TOOL_ERROR` — and the
  command line adds two of its own: `IO_ERROR` where it could not read the input or write the
  output, and `AGENT_ERROR` at the crash barrier.
- The command's exit codes are `0` a payload was written, `1` an unexpected crash, `2` the input
  could not be read or failed the contract, `3` a credential was missing or rejected, and `4` the
  payload was built and could not be written. An unexpected crash is reported as `AGENT_ERROR`; the
  exit code is the machine-readable signal, and the error vocabulary was deliberately not widened to
  carry a second meaning.

**Credentials and logging**

- Credentials come from the environment, and a `.env` file is read where one exists. `.env.example`
  records the names with no values, and real credentials never enter the repository. A blank
  variable is treated as unset: a required one is rejected, and an optional one falls back to its
  default.
- **The Bright Data token travels inside the endpoint URL**, which makes the URL itself a
  credential. The settings object builds the real URL only when a caller asks for it by that name,
  its `repr` shows the redacted form, and every message the package reports, logs or publishes
  passes through a scrub that puts `***` in the token's place. The OpenRouter key is kept out of the
  settings `repr` for the same reason: the test task runs `pytest --showlocals`.
- The command line configures the `fact_checker` logger rather than the root logger, and leaves
  propagation on. The library configures nothing itself, so an application embedding it owns its own
  handlers.
- Logs go to stderr and never to stdout. At `INFO` there is one line per statement; at `DEBUG` one
  line per tool call and the traceback behind any failure. The reason a run ended is logged at
  `CRITICAL`, so no `LOG_LEVEL` setting hides why a non-zero code came back. `LOG_LEVEL` is the
  level control, and there is no `--verbose` flag.

## Testing Decisions

- **The default suite makes no network call and needs no credential.** The seams are the two model
  runnables and the MCP toolkit, and every test in it drives them through fakes. That is why the CI
  job carries no secret and why a check never depends on two upstream services being up. The one
  test that does call them is deselected, and is described below.
- Tests cover external behaviour — payload in, payload out — rather than internal call structure.
  The named areas: input validation and its rejections, identifier assignment, the output envelope
  and its ordering, the split between a ruling and an error, the retry and timeout paths, the
  concurrency bound, the cache and its single-flight behaviour, the command-line surface and every
  exit code, and the redaction of the endpoint URL.
- Coverage floor is 80 percent, enforced by the test task rather than by review.
- One credentialed end-to-end test exercises the whole path against live OpenRouter and live Bright
  Data. It is marked `integration` and deselected by `poe test`, so the default suite stays offline;
  CI never invokes it. It asserts the shape and the pass-through of an opinion, never which verdict
  came back — pinning a verdict there would make it a quality measure.
- **LLM quality is measured separately**, by a promptfoo suite of about twenty cases in
  `promptfoo/`, spread across the four verdicts and including opinions that must pass through
  unchecked and a claim that cannot be settled from public sources. It runs by hand when a prompt
  changes and is wired to nothing. Its assertions are deterministic — a JSON Schema over the whole
  entry, and a JavaScript expression over the verdict — so no grading credential is needed. A case
  that fails is a finding about the prompt or the model, not a reason to loosen the assertion.

## Accepted Limitation

**References are written by the model and are not checked against the retrieved text.** An excerpt
may be a paraphrase rather than a quotation, and a citation here is not a verified quotation. This
was accepted deliberately in exchange for speed. The package README states it plainly, so that no
reader downstream mistakes a citation for a verified quotation.

## Out of Scope

- Extraction of statements from a document, and classification of them — the two neighbouring
  stages.
- Verifying an excerpt against the page it came from. See the accepted limitation above.
- An HTTP API or a user interface. Library and command line only.
- Any store of past runs: the tool reads a file, writes a file, and keeps nothing.
- A cost estimate in `meta`.
- A destination allowlist or a private-address blocklist on the fetch path, and any control that
  detects or neutralises an injection attempt in a fetched page.
- Adversarial and load testing.
- Non-English input.
- A cap on total spend per run, and rate-limit-aware backpressure beyond per-call retry with
  backoff.
