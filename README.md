# ragroast 🔥

[![CI](https://github.com/amolsrivastava21/ragroast/actions/workflows/ci.yml/badge.svg)](https://github.com/amolsrivastava21/ragroast/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![core: pure stdlib](https://img.shields.io/badge/core-pure%20stdlib-orange.svg)](pyproject.toml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Does your RAG actually beat BM25?**

Everyone reaches for a vector database. Almost no one checks whether their fancy
dense retrieval actually beats a 30-year-old keyword baseline on *their own* data.
`ragroast` runs the fight and hands you the receipts.

```
ragroast · retrieval showdown on 'scifact'  (5183 docs · 300 queries · k=10)

  method                nDCG@10   Recall@10     MRR      latency
  ──────────────────────────────────────────────────────────────
  BM25 (baseline)         0.664       0.782   0.634      7.36 ms
  Dense (MiniLM)          0.645       0.783   0.605     74.19 ms
  Hybrid (RRF)            0.685       0.821   0.646     82.44 ms
  ──────────────────────────────────────────────────────────────
  VERDICT  BM25 beats your dense retriever by +2.9% nDCG@10.  🔥
           Hybrid (RRF) wins overall (+3.2% over BM25).
  🔥 Photo finish — and the 30-year-old baseline still edged it.

  method: Okapi BM25 (k1=1.5, b=0.75) · dense: all-MiniLM-L6-v2 · fusion: RRF (k=60) · metrics computed from scratch
```

<sub>*(real run on the BEIR **scifact** test split — 5183 docs, 300 queries. Reproduce with `ragroast run` on its corpus/queries/qrels. Yes, it roasts you — pass `--plain` for sober output.)*</sub>

## Why

Modern RAG treats retrieval as a black box: embed everything, stuff a vector DB,
ship it. But the 2026 benchmarks keep finding the same thing — **plain BM25 is a
brutally strong baseline**, and the highest-impact upgrade to a pure-vector
pipeline is usually just *adding BM25 back*. If you can't beat the baseline, you
don't need the vector DB; you need to know that before it's in production.

`ragroast` is the missing sanity check: a fair, reproducible head-to-head so the
"just use embeddings" instinct has to earn it.

## Quickstart

No dependencies for the baseline — it's pure standard library:

```bash
git clone https://github.com/amolsrivastava21/ragroast
cd ragroast
python -m ragroast demo
```

Add the dense + hybrid contenders (downloads MiniLM once, ~90 MB):

```bash
pip install -e ".[dense]"
python scripts/build_demo_vectors.py   # precompute once → offline forever
python -m ragroast demo
```

Once published: `pip install ragroast` → `ragroast demo`.

## What you'll need

ragroast scores rankings against **relevance labels** (`qrels`) — no labels, nothing to
measure. Pick the path that matches what you have:

- **Just curious / no data of your own** → `ragroast demo` (or `python -m ragroast run ragroast/_sample`). Zero setup.
- **A labeled set** (corpus + queries + qrels, BEIR-style) → `ragroast run` (below).
- **An existing retrieval pipeline** → `ragroast score --run …`; score its output directly.
- **Documents but no labels** → the hard case. Auto-labeling (LLM-as-judge) is on the roadmap;
  until then, hand-labeling even a few dozen queries already gives a real signal.

## Run it on your own data

Point it at a **directory** laid out BEIR-style (`corpus.jsonl`, `queries.jsonl`,
and `qrels/test.tsv` or `qrels.jsonl`) — one path, no flags to remember:

```bash
ragroast run ./my-dataset/ --dense minilm --k 10
```

…or pass the three files explicitly:

```bash
ragroast run --corpus corpus.jsonl --queries queries.jsonl --qrels qrels.jsonl --dense minilm
```

Formats (BEIR-compatible):

- corpus  — `{"_id": "...", "title": "...", "text": "..."}`
- queries — `{"_id": "...", "text": "..."}`
- qrels   — `{"qid": "...", "docid": "...", "rel": 1}`  (or BEIR `.tsv`)

### Already have a retrieval pipeline? Just score its output.

No need to reimplement your retriever in ragroast — hand it a **run file** (your
pipeline's ranked results) and it scores them against your qrels:

```bash
ragroast score --run my_run.trec --qrels qrels.jsonl --k 10
```

A run file is either TREC format (`qid Q0 docid rank score tag`) or JSONL
(`{"qid": "...", "docids": ["d1", "d2", ...]}`, already ranked).

## How it works

- **BM25** — Okapi BM25 (`k1=1.5`, `b=0.75`), written from scratch in
  [`retrievers.py`](ragroast/retrievers.py). No `rank_bm25` import: a baseline you
  can't read is not a baseline you can trust.
- **Dense** — cosine similarity over `all-MiniLM-L6-v2` embeddings (swap in any
  embedder).
- **Hybrid** — Reciprocal Rank Fusion (`k=60`) over the two, the standard
  score-normalization-free way to combine ranked lists.
- **Metrics** — nDCG@k, Recall@k, MRR, [from scratch](ragroast/metrics.py) and
  unit-tested against hand-computed values.

## Why not just use BEIR / ranx / pytrec_eval?

Use them — they're excellent, and if you're writing a paper you should. They're
also a research workflow: TREC tooling, a metrics library you wire into a harness
yourself, a notebook, a GPU to embed a corpus. `ragroast` is the five-minute
version — one command, zero dependencies for the baseline, three retrievers already
wired up, and an opinionated **verdict** instead of a dataframe. It's the check you
run *before* the vector database ships, to find out whether you even need it. When
you want publication-grade rigor, graduate to BEIR + `ir_measures`.

## The one rule: the fight has to be fair

A benchmark that rigs the baseline is worse than no benchmark. So:

- the dense side uses a **real, competitive** embedding model, never a toy;
- every parameter (`k1`, `b`, RRF `k`, cutoff `k`, model) is printed with the
  results, so anyone can reproduce or challenge them;
- BM25 sometimes loses here — that's the point. When your embeddings win, you'll
  have earned the receipt.

## Limitations (read these)

- The bundled `demo` set is tiny and illustrative — a mechanism check, not a
  benchmark. Point `ragroast run` at [BEIR](https://github.com/beir-cellar/beir)
  (`scifact`, `nfcorpus`, `fiqa`, …) or your own labeled data for real numbers.
- Metrics need relevance judgements (qrels). No labels, no scores — that's a
  feature, not a bug.

## Troubleshooting

- **`run` on a real corpus pauses before results** — it's embedding every document once.
  Progress prints per stage (`scoring …%  ~ETA left`); bigger corpora just take longer.
- **First dense run downloads MiniLM (~90 MB).** Offline or behind a strict proxy? Pre-cache
  the model once, then run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to skip network checks.
- **`Dense retrieval needs the optional dependency`** → `pip install 'ragroast[dense]'`.
- **No qrels?** Nothing to score — see [What you'll need](#what-youll-need).
- **Machine-readable output?** `--plain` drops the roast and colors; the table stays.

## Roadmap

- [x] `ragroast run ./dataset/` — auto-detect corpus/queries/qrels
- [x] `ragroast score --run …` — score any pipeline's ranked output (TREC/JSONL)
- [ ] `pip install ragroast` on PyPI
- [ ] one-command BEIR dataset fetch (`ragroast bench scifact`)
- [ ] folder / CSV ingestion (md · pdf · txt · csv → corpus)
- [ ] LLM-as-judge to auto-generate qrels for unlabeled corpora
- [ ] cross-encoder rerank stage
- [ ] OpenAI / Cohere / Voyage embedding adapters (with $ cost column)
- [ ] HTML report

## License

MIT © 2026 Amol Srivastava. Built because three years of running search at scale
made "just use embeddings" hard to hear without a benchmark attached.
