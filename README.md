# ragroast 🔥

**Does your RAG actually beat BM25?**

Everyone reaches for a vector database. Almost no one checks whether their fancy
dense retrieval actually beats a 30-year-old keyword baseline on *their own* data.
`ragroast` runs the fight and hands you the receipts.

```
ragroast · retrieval showdown on 'scifact'  (5183 docs · 300 queries · k=10)

  method             nDCG@10   Recall@10     MRR      latency
  ──────────────────────────────────────────────────────────
  BM25 (baseline)      0.641      0.780      0.612      1.3 ms
  Dense (MiniLM)       0.598      0.710      0.560      0.9 ms
  Hybrid (RRF)         0.701      0.860      0.668      2.1 ms
  ──────────────────────────────────────────────────────────
  VERDICT  BM25 beats your dense retriever by +7.2% nDCG@10.  🔥
           Hybrid (RRF) wins overall (+9.4% over BM25).

  method: Okapi BM25 (k1=1.5, b=0.75) · dense: all-MiniLM-L6-v2 · fusion: RRF (k=60)
```

<sub>*(illustrative output — run `ragroast demo` for live numbers on the bundled set)*</sub>

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

## Run it on your own data

BEIR-compatible formats:

```bash
ragroast run \
  --corpus corpus.jsonl \      # {"_id": "...", "title": "...", "text": "..."}
  --queries queries.jsonl \    # {"_id": "...", "text": "..."}
  --qrels qrels.jsonl \        # {"qid": "...", "docid": "...", "rel": 1}  (or BEIR .tsv)
  --dense minilm --k 10
```

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

## Roadmap

- [ ] `pip install ragroast` on PyPI
- [ ] one-command BEIR dataset fetch (`ragroast bench scifact`)
- [ ] cross-encoder rerank stage
- [ ] LLM-as-judge to auto-generate qrels for unlabeled corpora
- [ ] OpenAI / Cohere / Voyage embedding adapters (with $ cost column)
- [ ] HTML report

## License

MIT © 2026 Amol Srivastava. Built because three years of running search at scale
made "just use embeddings" hard to hear without a benchmark attached.
