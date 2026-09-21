# Contributing to ragroast

Thanks for looking! ragroast is small on purpose — the whole point is a benchmark you
can read and trust. Contributions that keep it that way are very welcome.

## Setup

```bash
pip install -e ".[dense,dev]"
python -m pytest -q
```

The core is pure standard library — `ragroast demo` runs with zero third-party packages.
Anything heavy (torch, sentence-transformers, API clients) stays behind the optional
`dense` extra and is imported lazily, so `import ragroast` never needs them.

## The one rule

The comparison has to be **fair** — a real, competitive embedding model, a real public
benchmark, and every parameter printed with the results. See [`AGENTS.md`](AGENTS.md) for
the full guardrails. If a change could make the benchmark less fair or less reproducible,
it doesn't land. (The numbers are deadpan; the only place with personality is the roast in
`report.py`, and it never touches the numbers.)

## Adding things

- **A retriever** — implement the `Retriever` protocol in `retrievers.py`: `.index(docs)`
  and `.search(query, k, query_id) -> [doc_id, ...]`. Ship a ranking-sanity test.
- **An embedding backend** — add an adapter in `embeddings.py`, behind the `dense` extra,
  degrading gracefully (a clear message) when the extra isn't installed.
- **A metric** — write it from scratch in `metrics.py` (no off-the-shelf IR libraries) with a
  test that checks it against a hand-computed or otherwise independently verifiable value.
  Correctness *is* the product.

## Style

Small pure functions, docstrings on modules and public functions, `from __future__ import
annotations` + type hints, Python >= 3.9, readability over cleverness. Keep
`python -m pytest -q` green.

## Tests

Any change to `metrics.py` or `retrievers.py` must keep the suite green, and new
metrics/retrievers ship with their own tests. Run `python -m pytest -q` before opening a PR.
