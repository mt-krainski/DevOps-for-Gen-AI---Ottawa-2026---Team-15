# factchecker

Takes statements that an upstream classifier has labelled `fact` or `opinion`. Each factual
statement is checked against what the web says about it, and comes back with a verdict and the
sources behind it. Opinions pass through untouched.

The tool reports what the evidence shows. It does not establish truth, and its vocabulary says so.

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
- `unverifiable` — the search ran and the evidence does not settle the claim.

`unverifiable` is a finding. A consumer that treats it as a failure misreads the output.

`ruling.confidence` measures trust in the verdict, not the truth of the statement. An `unverifiable`
verdict at high confidence says the tool is sure the claim cannot be settled this way.

## References are not verified quotations

The references come from the model. Nothing checks them against the text that was retrieved, so an
excerpt may be a paraphrase rather than a quotation. This was accepted deliberately in exchange for
speed. Read `source` as a pointer to follow, not as a promise that `excerpt` appears there word for
word.

## Development

```bash
uv run poe lint
uv run poe test
```
