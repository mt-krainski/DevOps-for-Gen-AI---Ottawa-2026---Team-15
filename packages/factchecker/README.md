# factchecker

Takes statements that an upstream classifier has labelled `fact` or `opinion`. Each factual
statement is checked against what the web says about it, and comes back with a verdict and the
sources behind it. Opinions pass through untouched.

This document publishes the tool's contract: the input it reads, the output it writes, and the seam
a checking agent plugs into. Every section below describes that contract. A statement is searched
with Bright Data and ruled on by a model reached through OpenRouter, so a run needs a credential for
each, and `meta.model` names the model that did the work.

The tool reports what the evidence shows. It does not establish truth, and its vocabulary says so.

## Command line

```bash
uv run factchecker --input statements.json --output rulings.json
```

Run it from `packages/factchecker/`, where the checks under Development are run too. Where the
package is installed, the same command is on the path as `factchecker`.

Both paths are required. `--input` names a JSON file holding the input described below, and
`--output` names the file the rulings are written to. `--env-file` names the settings file, and
defaults to `.env` in the working directory. `--verbose` raises the log level to DEBUG.

### Credentials

A run reads two credentials, and neither has a default:

- `OPENROUTER_API_KEY` — the key the chat model authenticates with.
- `BRIGHTDATA_API_TOKEN` — the token the search and page tools authenticate with.

Copy `.env.example` to `.env` in this directory, fill both in, and run the command from here. Git
ignores `.env`. Every other setting is optional, and `.env.example` names each one beside the
default that applies while it is unset.

The file is read from the path `--env-file` names and nowhere else. Nothing searches upward from the
working directory, so a run started somewhere else picks up nothing it was not pointed at. A value
already exported in the environment beats the file's, so a one-off override needs no edit.

The Bright Data token travels inside the endpoint URL, because that is how Bright Data
authenticates. The tool holds the token in a type that prints itself with `REDACTED` in the token's
place, and it filters the log records that the libraries underneath it write, so no log line and no
error message carries the token. The OpenRouter key is kept out of every printed form of the
settings for the same reason.

### Exit codes

- `0` — an output payload was written. Statements inside it may still carry errors.
- `2` — the input could not be read, or it did not satisfy the contract.
- `3` — a credential was rejected. Get a new one.
- `4` — the payload was built, and it could not be written to the output path.
- `5` — a setting is absent, or it will not parse, or the search server could not be reached. Look
  at the environment.

A failed statement is a result the payload carries, so it leaves the exit code at zero. Each of the
other codes names a different reason no output file holds a payload: the input, the credential, the
write, or the setup. An unexpected crash exits `1`, so a failed write has a code of its own rather
than that one.

`3` and `5` are worth telling apart. A token that was supplied and refused is `3`, and a variable
nobody filled in is `5`. They ask an operator for different things.

### Limits

Every limit below is a default, and `.env.example` names the variable that moves it.

A run checks eight statements at once. A check that is still running after 240 seconds is cancelled,
and that statement comes back with a `timeout` error quoting the limit. One check may make ten tool
calls before it must rule on what it has, and reaching that ceiling is not a failure: the searching
stops there, and the agent is asked to rule on the evidence it holds. At most 100,000 characters of
any one fetched page reach the model, and a page cut there says so where it was cut. A call that
fails for a reason another try might fix is retried three times, with a jittered doubling wait
between attempts.

### What a run costs

Two meters run at once: OpenRouter charges for tokens, and Bright Data charges for requests.

The request side is bounded and easy to read. One statement makes at most ten Bright Data requests,
and a run of fifty statements at most five hundred. A run caches what it fetched, so a search or a
page a later statement repeats costs nothing. Two statements that ask for the same thing at the same
moment both pay for it, because neither has recorded an answer yet when the other looks.

The token side is dominated by pages rather than searches. Every turn resends the whole
conversation, so a fetched page is paid for again on each turn that follows it. A statement that
reads two long pages can spend a few hundred thousand prompt tokens, which is a few cents at the
default model's $0.08 per million. Lowering `FACTCHECKER_PAGE_CHARACTER_CEILING` is the lever that
moves this most.

### Logging

Every log record goes to stderr. The output file is the only place the payload is written, and
stdout stays empty.

At INFO the tool writes one line for each statement, naming the statement, the elapsed time, and
what became of it. `--verbose` raises the level to DEBUG, and prints the traceback under a rejected
input. Without the flag the `LOG_LEVEL` environment variable names the level, and the level is INFO
where that variable is unset or names no level the standard library knows.

The reason a run ended is written at CRITICAL, so no setting of `LOG_LEVEL` can hide why a non-zero
exit code was returned.

A run that a rejected credential ends writes no line for the statements it cancelled. Those checks
never finished, so there is nothing to report for them. That is the design, and not a missing
record.

## Input

One JSON object with a `statements` array. Field names are camelCase on the wire.

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

`id` is optional. Where it is absent the tool assigns `s1`, `s2`, and so on, by position in the
input. A supplied `id` is used as it stands. An assigned `id` means nothing outside its own run.

Identifiers must be unique across the payload, assigned and supplied alike. A supplied `s1` on the
second statement therefore collides with the identifier the first statement was assigned, and the
run exits `2` naming the repeated identifier.

`surroundingContext` is required. It turns a claim that depends on its surroundings into a claim
that can be searched.

`classification.class` is `fact` or `opinion`. Any other value fails validation, and the message
names the value that failed.

`classification.confidence` is echoed to the output and never decides whether a statement is
checked. A statement labelled `fact` is checked whatever its confidence.

A statement may carry fields this contract does not name. They are dropped. The values of the named
fields are validated strictly.

## Output

One entry for every input statement, in input order. Each entry repeats the input fields and adds
`ruling` and `error`. Both keys are always present, and either may be null, so every consumer reads
one shape.

```json
{
  "meta": {
    "model": "<model name>",
    "startedAt": "2026-08-22T14:03:11Z",
    "finishedAt": "2026-08-22T14:05:47Z",
    "counts": { "total": 50, "checked": 30, "skipped": 19, "failed": 1 },
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
    },
    {
      "id": "s2",
      "surroundingContext": "...",
      "statement": "...",
      "classification": { "class": "fact", "confidence": 0.9 },
      "ruling": null,
      "error": {
        "kind": "timeout",
        "message": "the check exceeded the per-statement limit of 240.0 seconds"
      }
    }
  ]
}
```

Three cases are distinguishable:

- **Opinion.** `ruling` and `error` are both null. Nothing was checked.
- **Checked.** `ruling` holds the verdict.
- **Failed.** `error` names what went wrong and `ruling` is null.

`meta` records the run, not the statement: the model, the start and end times, the counts, and the
observed token and search usage. `startedAt` and `finishedAt` are timezone-aware and serialize in
ISO 8601, which writes a zero UTC offset as `Z`.

### Verdicts

- `supported` — the evidence backs the claim.
- `refuted` — the evidence contradicts the claim.
- `mixed` — the claim is partly right, or the sources disagree with each other.
- `unverifiable` — the search ran and the evidence does not settle the claim.

`unverifiable` is a finding. A consumer that treats it as a failure misreads the output.

`ruling.confidence` measures trust in the verdict, not the truth of the statement. An `unverifiable`
verdict at high confidence says the tool is sure the claim cannot be settled this way.

### Errors

`error` carries two keys. `kind` names the failure and `message` describes it in prose:

- `timeout` — the check was still running at the per-statement limit.
- `malformed_ruling` — the model's ruling did not validate, and nor did the one it wrote after being
  shown why. A cheap model produces this most often.
- `check_failed` — the checker raised something it did not name itself.

A checking agent may add kinds, so branch on the kinds you know and leave a default branch for the
rest.

## Limitations

Two are worth knowing before you read an output payload.

**References are the model's own work.** Nothing compares them against the text that was retrieved,
so an excerpt may be a paraphrase rather than a quotation. This was accepted deliberately in
exchange for speed. Read `source` as a pointer to follow, not as a promise that `excerpt` appears
there word for word.

**`meta.usage` under-reports cost.** A statement that ends in an error contributes nothing to the
totals. That is worst where it matters most: a `malformed_ruling` spends the whole tool-call budget
before the ruling fails to validate, so it is the most expensive outcome there is and it reports
zero. Read `meta.usage` as a floor, and read it beside `meta.counts.failed`.

## The checker seam

A checking agent plugs into one seam, `factchecker/checker.py`:

- `StatementChecker` is the protocol a checking agent implements. Its one method, `check`, takes a
  statement with its identifier already assigned and returns a `CheckOutcome`.
- `CheckOutcome` carries the ruling and what producing it consumed: the prompt tokens, the
  completion tokens, and the searches. The run adds those up into `meta.usage`.
- `AgentChecker`, in `factchecker/agent.py`, is the implementation this build runs. It searches,
  reads the pages it chooses, and rules with sources. One instance serves every statement of a run
  at once, because it holds nothing that changes.

The orchestrator holds an implementation to three rules:

- Raising `AuthenticationFailed`, from `factchecker/errors.py`, ends the whole run, because a
  rejected credential fails every statement alike.
- Raising `CheckFailed`, from the same module, fails that one statement under the `kind` the
  exception carries, rather than under `check_failed`. Reach for it when the caller should read one
  failure differently from another: a ruling that will not parse asks for a different response than
  a dropped connection does. Write the message yourself. It reaches the output payload, so nothing
  an upstream library wrote belongs in it.
- Any other exception becomes that one statement's `check_failed` error, and the run carries on
  with the statements that are left.

Your `check` coroutine is also what gets cancelled at the per-statement limit under Limits. It must
tolerate cancellation partway, and the statement it was checking comes back under `timeout` rather
than under either kind above.

## Development

Run both checks from `packages/factchecker/`:

```bash
uv run poe lint
uv run poe test
```

`lint` runs `ruff check` and `ruff format --check`. `test` runs the suite under coverage, and the
coverage report fails below 80 percent. Continuous integration runs the same two commands on every
pull request that touches this package.

No test reaches a network or reads a credential. Every one of them drives a fake model and a fake
tool, which is what lets continuous integration run the suite with no secret of any kind.

## Evaluation suite

`eval/` holds nineteen cases and scores the tool on one thing: does it reach the right verdict.
Sixteen are ordinary claims across the four verdicts, and three are traps — a claim whose sources
share names, numbers and vocabulary with it, so that a careless read takes topical overlap for
confirmation.

It runs by hand and never on a push, because every case spends real money at OpenRouter and real
requests at Bright Data.

promptfoo is an npm tool and needs Node 22.22 or newer. The package itself needs no Node. Run
`uv sync` first: the provider imports `factchecker`, and `promptfooconfig.yaml` points promptfoo at
`../.venv/bin/python` to get it.

```bash
cd eval
npx promptfoo@latest eval
npx promptfoo@latest view
```

The credentials come from the same `.env` the command reads, one directory up. Each case runs in its
own process, so each opens its own connection and starts with an empty cache.

Every case asserts two things and no more: the output parses as JSON, and its `verdict` equals the
case's expected one. The scoring is exact and deterministic, with no model-graded assertion
anywhere. A verdict is one of four words, which is what makes it checkable by comparison rather than
by judgement.

Nothing asserts on the justification, the references, or the confidence. That keeps the suite small
and cheap, and it leaves one blind spot worth remembering when you read a score: a case that guesses
the right verdict with no evidence behind it scores exactly like one that did the work.

The two `mixed` cases sit on a genuine judgement boundary, where a defensible `refuted` scores zero.
Read those failures rather than counting them.

A full run costs under a dollar at the default model, and roughly twenty-five times that at
`anthropic/claude-sonnet-5`. promptfoo caches its results, so a case that has not changed costs
nothing on a rerun.
