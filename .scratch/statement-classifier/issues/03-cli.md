# 03: CLI

**What to build:** Run `statement-classifier classify` against a JSON file or stdin and get classified JSON back on a file or stdout, with predictable exit codes so it composes reliably in shell scripts and pipelines.

**Blocked by:** 02: Batch API

**Status:** ready-for-agent

- [ ] `statement-classifier classify --input <path|-> --output <path|->` reads a `ClassifierInput` JSON document and writes a `ClassifierOutput` JSON document
- [ ] Omitting `--input` (or passing `-`) reads from stdin; omitting `--output` (or passing `-`) writes to stdout
- [ ] `--concurrency <n>` optionally overrides the default concurrency
- [ ] Exit code `0` on a successful batch call, even when individual statements carry per-item errors
- [ ] Exit code `2` on invalid input, `3` on auth/config error, `1` on unexpected/internal error, matching the `ClassifierError` codes from ticket 02
- [ ] On non-zero exit, a JSON error object (`{code, message}`) is written to stderr
- [ ] CLI tests cover: valid file in → file out with exit `0`; malformed JSON in → non-zero exit with a stderr JSON error; missing API key → exit `3`
