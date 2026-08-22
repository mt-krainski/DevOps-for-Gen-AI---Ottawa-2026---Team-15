# factchecker

Takes statements that an upstream classifier has labelled `fact` or `opinion`. Each factual
statement is checked against what the web says about it, and comes back with a verdict and the
sources behind it. Opinions pass through untouched.

This document publishes the tool's contract: the input it reads, the output it writes, and the seam
a checking agent plugs into. Every section below describes that contract. This build ships no
checking agent, so nothing searches yet: every factual statement comes back `unverifiable` with a
justification that says no search ran, no reference is cited, and `meta.model` reads `offline`.

The tool reports what the evidence shows. It does not establish truth, and its vocabulary says so.

## Command line

```bash
uv run factchecker --input statements.json --output rulings.json
```

Run it from `packages/factchecker/`, where the checks under Development are run too. Where the
package is installed, the same command is on the path as `factchecker`.

Both paths are required. `--input` names a JSON file holding the input described below, and
`--output` names the file the rulings are written to. `--verbose` raises the log level to DEBUG.

### Exit codes

- `0` — an output payload was written. Statements inside it may still carry errors.
- `2` — the input could not be read, or it did not satisfy the contract.
- `3` — a credential was rejected.
- `4` — the payload was built, and it could not be written to the output path.

A failed statement is a result the payload carries, so it leaves the exit code at zero. Each of the
other codes names a different reason no output file holds a payload: the input, the credential, or
the write. An unexpected crash exits `1`, so a failed write has a code of its own rather than that
one.

### Limits

A run checks eight statements at once. A check that is still running after 240 seconds is cancelled,
and that statement comes back with a `timeout` error quoting the limit.

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
- `check_failed` — the checker raised.

A checking agent may add kinds, so branch on the kinds you know and leave a default branch for the
rest.

## References are not verified quotations

The references come from the checker. Nothing compares them against the text that was retrieved, so
an excerpt may be a paraphrase rather than a quotation. This was accepted deliberately in exchange
for speed. Read `source` as a pointer to follow, not as a promise that `excerpt` appears there word
for word.

## The checker seam

A checking agent plugs into one seam, `factchecker/checker.py`:

- `StatementChecker` is the protocol a checking agent implements. Its one method, `check`, takes a
  statement with its identifier already assigned and returns a `CheckOutcome`.
- `CheckOutcome` carries the ruling and what producing it consumed: the prompt tokens, the
  completion tokens, and the searches. The run adds those up into `meta.usage`.
- `OfflineChecker` is the stand-in this build runs in place of a checking agent, with the behaviour
  described at the top of this document.

The orchestrator holds an implementation to two rules:

- Raising `AuthenticationFailed`, from `factchecker/errors.py`, ends the whole run, because a
  rejected credential fails every statement alike. Any other exception becomes that one statement's
  `check_failed` error, and the run carries on with the statements that are left.
- Your `check` coroutine is what gets cancelled at the per-statement limit under Limits, so it must
  tolerate cancellation partway.

## Development

Run both checks from `packages/factchecker/`:

```bash
uv run poe lint
uv run poe test
```

`lint` runs `ruff check` and `ruff format --check`. `test` runs the suite under coverage, and the
coverage report fails below 80 percent. Continuous integration runs the same two commands on every
pull request that touches this package.
