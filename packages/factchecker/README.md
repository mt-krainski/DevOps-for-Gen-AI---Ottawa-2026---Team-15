# factchecker

Takes statements that an upstream classifier has labelled `fact` or `opinion`. Each factual
statement is checked against what the web says about it, and comes back with a verdict and the
sources behind it. Opinions pass through untouched.

The tool reports what the evidence shows. It does not establish truth, and its vocabulary says so.

## Command line

```bash
factchecker --input statements.json --output rulings.json
```

Both paths are required. `--input` names a JSON file holding the input described below, and
`--output` names the file the rulings are written to. `--verbose` raises the log level to DEBUG.

This build ships no checking agent. Every factual statement comes back `unverifiable` with a
justification that says no search ran, and `meta.model` reads `offline`.

### Exit codes

- `0` — an output payload was written. Statements inside it may still carry errors.
- `2` — the input could not be read, or it did not satisfy the contract.
- `3` — a credential was rejected.
- `4` — the payload was built, and it could not be written to the output path.

A failed statement is a result the payload carries, so it leaves the exit code at zero. Each of the
other codes names a different reason no output file holds a payload: the input, the credential, or
the write. An unexpected crash exits `1`, so a failed write has a code of its own rather than that
one.

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

`id` is optional. Where it is absent the tool assigns `s1`, `s2`, and so on, in input order. A
supplied `id` passes through unchanged. An assigned `id` means nothing outside its own run.

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
    "model": "<openrouter model slug>",
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
        "justification": "At standard pressure water boils at 100C [1].",
        "references": [
          { "id": "1", "source": "https://...", "excerpt": "At 1 atm, water boils at 100 C" }
        ]
      },
      "error": null
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
- `unverifiable` — the search ran and the evidence does not settle the claim. In a build that ships
  no checking agent, as this one does, no search runs and every factual statement comes back
  `unverifiable`.

`unverifiable` is a finding. A consumer that treats it as a failure misreads the output.

`ruling.confidence` measures trust in the verdict, not the truth of the statement. An `unverifiable`
verdict at high confidence says the tool is sure the claim cannot be settled this way.

## References are not verified quotations

The references come from the model. Nothing checks them against the text that was retrieved, so an
excerpt may be a paraphrase rather than a quotation. This was accepted deliberately in exchange for
speed. Read `source` as a pointer to follow, not as a promise that `excerpt` appears there word for
word.

## Development

Run both checks from `packages/factchecker/`:

```bash
uv run poe lint
uv run poe test
```

`lint` runs `ruff check` and `ruff format --check`. `test` runs the suite under coverage and prints
the report. Continuous integration runs the same two commands on every pull request that touches
this package.
