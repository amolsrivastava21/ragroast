# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. Read this
first, then keep changes consistent with the conventions below.

## What this project is

`ragroast` answers one question: **does your RAG actually beat BM25?** It runs a
fair, reproducible retrieval showdown — hand-written Okapi BM25 vs a dense
(MiniLM) retriever vs an RRF hybrid — and scores them with from-scratch nDCG@k,
Recall@k, and MRR. The output is a single verdict table with every parameter
printed, so results can be reproduced or challenged.

The credibility of the whole project rests on the benchmark being **fair**. See
"The one rule" below — treat it as a hard constraint, not a suggestion.

## Setup & commands

Core is **pure Python standard library** — the baseline demo needs no install:

```bash
python -m ragroast demo            # BM25 baseline on the bundled sample set
```

Dense + hybrid contenders require the optional extra (downloads MiniLM once):

```bash
pip install -e ".[dense]"          # sentence-transformers
python scripts/build_demo_vectors.py   # precompute vectors -> ragroast/_sample/vectors.json
python -m ragroast demo            # now shows BM25 / Dense / Hybrid
```

Run on your own BEIR-compatible data:

```bash
ragroast run --corpus corpus.jsonl --queries queries.jsonl \
             --qrels qrels.jsonl --dense minilm --k 10
```

Tests (install dev extra first with `pip install -e ".[dev]"`):

```bash
python -m pytest -q
```

## Repo layout

```
ragroast/
  __main__.py     # enables `python -m ragroast`
  cli.py          # argparse entrypoint: `demo` | `run` (console script -> ragroast.cli:main)
  metrics.py      # nDCG@k, Recall@k, MRR — from scratch, unit-tested
  retrievers.py   # Retriever protocol, BM25Retriever, DenseRetriever, RRFHybrid
  embeddings.py   # precomputed-vector loader + optional sentence-transformers adapter
  data.py         # BEIR-compatible corpus/queries/qrels loaders + bundled sample loader
  runner.py       # runs all retrievers, computes metrics, assembles the verdict
  report.py       # dependency-free aligned verdict table
  _sample/        # bundled sample corpus/queries/qrels (+ vectors.json once built)
tests/
  test_metrics.py # metrics vs hand-computed values
  test_bm25.py    # BM25 ranking sanity
scripts/
  build_demo_vectors.py  # one-time offline precompute of demo embeddings
```

## Conventions

- **Python >= 3.9.** Use `from __future__ import annotations` and type hints.
- **Core stays stdlib-only.** Never add a runtime dependency to `[project].dependencies`.
  Anything heavy (torch, sentence-transformers, API clients) goes under an optional
  extra in `[project.optional-dependencies]` and must be imported lazily so that
  `import ragroast` and `ragroast demo` work with zero third-party packages.
- **Do not `import rank_bm25`** (or any off-the-shelf BM25/metrics library). BM25 and
  all metrics are hand-written on purpose — a baseline you can't read is one you
  can't trust. Improve the implementations in place instead of swapping in a package.
- **New retrievers** implement the `Retriever` protocol in `retrievers.py`.
- **New embedding backends** are added as adapters in `embeddings.py`, kept behind an
  optional extra, and must degrade gracefully when the extra isn't installed.
- The demo must run **instantly and offline** with no ML dependencies — that is why
  demo vectors are precomputed to `_sample/vectors.json` rather than embedded at runtime.
- Match the existing style: small pure functions, docstrings on modules/public
  functions, no clever one-liners over readability.

## Testing expectations

- Any change to `metrics.py` or `retrievers.py` must keep `python -m pytest -q` green.
- New metrics/retrievers ship with tests that check against hand-computed or
  otherwise independently verifiable values — correctness is the product.

## The one rule: the fight has to be fair

A benchmark that rigs the baseline is worse than no benchmark. Non-negotiable:

- the dense side uses a **real, competitive** embedding model, never a toy or hashing
  embedding;
- the demo dataset is a **real public IR benchmark** subset, not hand-crafted docs;
- **every parameter is printed** with the results (`k1`, `b`, RRF `k`, cutoff `k`,
  model name, dataset);
- BM25 is allowed to win — that is the point. Do not tune the comparison to force a
  predetermined outcome.

If a change could make the comparison less fair or less reproducible, don't make it.
