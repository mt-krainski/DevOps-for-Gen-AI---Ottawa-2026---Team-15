# fact-checker — three-minute talk track

Five slides, roughly thirty-five seconds each. The deck is `index.html` in this
directory; arrow keys or a click move between slides.

Read it once with a timer before you present. If you are running long, slide 4's
second half — the context point — is the one to compress. Never cut slide 4
entirely: it is the only slide that says why the product is built this way.

---

## Slide 1 — Title · 0:00 to 0:20

> We are Team 15, and our theme is AI reliability engineering — hallucination
> detection.
>
> Our project is fact-checker. Text goes in. Every claim in it comes back with a
> verdict and the sources behind it.

## Slide 2 — Problem · 0:20 to 0:55

> Generating text got free. Standing behind it did not.
>
> Look at this sentence. Two checkable claims and one opinion, tangled together in
> a single line. Before anyone can pull one source, someone has to untangle them.
>
> A model drafts a page in seconds. Checking that same page still costs a person an
> afternoon — so today it is done by hand, or it is not done at all.

## Slide 3 — Architecture · 0:55 to 1:30

> So we split it three ways.
>
> Stage one labels each statement fact or opinion. Stage two takes only the facts,
> searches the web, and returns a verdict with references — opinions pass straight
> through. Stage three lays every claim out beside its evidence.
>
> Three separate programs, joined by a JSON contract and nothing else. Any one of
> them can be rewritten without touching the other two.
>
> And note the fourth verdict. Unverifiable is a finding, not a failure — the
> search ran, and the evidence did not settle it. We report what the evidence
> shows. We do not claim to establish truth.

## Slide 4 — The insight · 1:30 to 2:15

*The slide the pitch turns on. Slow down here.*

> Here is why the first stage exists at all.
>
> Most sentences in real writing cannot be fact-checked. They are opinions,
> framing, predictions. And searching is where all the money goes — every claim
> costs a tool-using loop. Labelling costs one small call.
>
> So the cheap call decides whether to spend the expensive one. An opinion never
> reaches the network.
>
> The second thing: a claim on its own stops being checkable. "The figure was
> fixed in 1954." You cannot search that — it lost what it pointed at. So every
> claim travels with the text around it, and we use that context to read the
> claim, never to judge in its place.
>
> That is the whole shape. Each stage adds to the record and rewrites nothing.

## Slide 5 — What you get · 2:15 to 3:00

> And this is what lands in front of you. The claim, the verdict, the reasoning,
> and the sources — close enough together that you can check the checker.
>
> That last part is the point. We are not asking you to trust a label. We are
> putting the evidence next to the claim so you can judge it yourself.
>
> We report what the evidence shows. Where it settles nothing, the tool says so
> rather than guessing — and that answer is a result, not an error.

---

## If a judge asks

**"What is mocked right now?"** Not the searching agent. Every factual statement
gets its own agent run against Bright Data's hosted MCP server, spends a
tool-call budget searching and scraping pages, and rules on what it read with
cited references. The classifier is live too. The one stand-in left is the
display: it renders the true ruling shape from a fixed sample, because the file
picker is not built. The limitation worth naming instead is the references. They
are the model's own, and nothing checks an excerpt against the page it came
from — so a citation is a pointer to follow, not a verified quotation. All of
this is in the README's demo notes.

**"How do you know it works?"** Not as well as you would like. Stage two ships a
hand-run quality suite: cases spread across all four verdicts, each held to a
deterministic assertion — no model grading a model. It catches a prompt
regression on the claims it covers. What it is not is a labelled set, and there
is still no accuracy baseline. We measure cost, throughput and failure rate off
every run's `meta` block, and we would rather tell you that than show you an
accuracy number we made up.

**"What breaks first at 10x?"** The model gateway. Concurrency is a semaphore in
one process, so scale means more processes against a shared rate-limit budget we
have not built yet. The stages themselves fan out cleanly, because the contract
between them is a file.

**"How do you handle secrets?"** Environment only, nothing committed, and no CI
job carries one — the tests fake the model boundary, so they need no credential.
Our search provider puts its token inside the URL, which makes the URL itself a
credential, so it is wrapped in a type that only yields the real URL when a caller
asks for it by name. Print it or log it and you get `token=***`.

**"What are the risks of getting this wrong?"** A wrong verdict that someone
trusts. That is why the output is advisory and terminal — nothing acts on a
verdict, and the last stage exists so a person reads the claim beside the evidence
and decides. The threat model and the system card in the README go through this
properly.

**"Who owns it when it fails?"** Nobody yet. No on-call, no escalation, no
incident process. It is a gap, and it is listed as one.
