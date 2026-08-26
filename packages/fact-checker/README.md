# fact-checker

Takes statements that an upstream classifier labelled `fact` or `opinion`, searches the web for each
factual one, and writes a ruling with cited sources. Opinions pass through untouched.

The tool reports what the evidence shows. It does not establish truth, and its verdict vocabulary —
`supported`, `refuted`, `mixed`, `unverifiable` — says so.

## Accepted limitation

References are written by the model and are not checked against the retrieved text. An excerpt may
be a paraphrase rather than a quotation. Do not read a citation here as a verified quotation.
