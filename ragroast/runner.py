"""Run the showdown: index each retriever, score every query, average metrics."""
from __future__ import annotations

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data import Doc, Query
from .metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from .retrievers import BM25Retriever, DenseRetriever, RRFHybrid


def _progress(msg: str) -> None:
    """Write a progress message to stderr, only in an interactive terminal.

    Progress goes to stderr on purpose: stdout stays a clean, pipeable table.
    Carriage-return updates and phase lines are silent when output is captured.
    """
    if sys.stderr.isatty():
        sys.stderr.write(msg)
        sys.stderr.flush()


def _fmt_eta(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


@dataclass
class Result:
    name: str
    ndcg: float
    recall: float
    mrr: float
    latency_ms: float
    n_queries: int
    k: int


def _evaluate(
    name: str,
    retriever,
    queries: Sequence[Query],
    qrels: Dict[str, Dict[str, int]],
    k: int,
    progress: bool = True,
) -> Result:
    scored = [q for q in queries if qrels.get(q.id)]
    total = len(scored)
    # Persistent start line (newline-terminated, so it survives in scrollback and
    # renders even where an in-place \r counter would not); the \r line below is a
    # live bonus on capable terminals.
    if progress and total:
        _progress(f"  scoring {name} ({total} queries) …\n")
    ndcgs: List[float] = []
    recalls: List[float] = []
    rrs: List[float] = []
    t0 = time.perf_counter()
    for i, q in enumerate(scored, 1):
        qrels_q = qrels[q.id]
        ranking = retriever.search(q.text, k=k, query_id=q.id)
        ndcgs.append(ndcg_at_k(ranking, qrels_q, k))
        recalls.append(recall_at_k(ranking, qrels_q, k))
        rrs.append(reciprocal_rank(ranking, qrels_q))
        if progress and (i % 5 == 0 or i == total):
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (total - i)
            _progress(f"\r    {100.0 * i / total:3.0f}%  ({i}/{total})  ~{_fmt_eta(eta)} left      ")
    if progress and total:
        # Overwrite the live counter with a persistent, newline-terminated summary.
        _progress(f"\r    done — {total} queries in {_fmt_eta(time.perf_counter() - t0)}                    \n")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n = max(1, len(ndcgs))
    return Result(
        name=name,
        ndcg=mean(ndcgs),
        recall=mean(recalls),
        mrr=mean(rrs),
        latency_ms=elapsed_ms / n,
        n_queries=len(ndcgs),
        k=k,
    )


def run_showdown(
    docs: Sequence[Doc],
    queries: Sequence[Query],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
    dense_vectors: Optional[Dict] = None,
    embedder=None,
    rrf_k: int = 60,
    k1: float = 1.5,
    b: float = 0.75,
    progress: bool = True,
) -> "OrderedDict[str, Result]":
    results: "OrderedDict[str, Result]" = OrderedDict()

    _progress(f"  indexing BM25 ({len(docs)} docs) …\n")
    bm25 = BM25Retriever(k1=k1, b=b).index(docs)
    results["BM25 (baseline)"] = _evaluate("BM25 (baseline)", bm25, queries, qrels, k, progress)

    dense = None
    if dense_vectors is not None:
        dense = DenseRetriever(
            doc_vectors=dense_vectors.get("docs"),
            query_vectors=dense_vectors.get("queries"),
            label="Dense (MiniLM)",
        ).index(docs)
    elif embedder is not None:
        _progress(f"  embedding {len(docs)} docs with the dense model …\n")
        dense = DenseRetriever(embedder=embedder, label="Dense (MiniLM)").index(docs)
        # Pre-embed all judged queries once, in a single batch. Otherwise each
        # query re-embeds on the fly during search — and again inside the hybrid
        # pass — which is slow and buries the progress line under per-item model
        # bars. Batched embedding yields identical vectors, so metrics are unchanged.
        scored_q = [q for q in queries if qrels.get(q.id)]
        if scored_q:
            _progress(f"  embedding {len(scored_q)} queries …\n")
            qvecs = embedder([q.text for q in scored_q])
            dense.query_vectors = {q.id: v for q, v in zip(scored_q, qvecs)}

    if dense is not None:
        results["Dense (MiniLM)"] = _evaluate("Dense (MiniLM)", dense, queries, qrels, k, progress)
        hybrid = RRFHybrid([bm25, dense], rrf_k=rrf_k)
        results["Hybrid (RRF)"] = _evaluate("Hybrid (RRF)", hybrid, queries, qrels, k, progress)

    return results


def score_runs(
    runs: Dict[str, Dict[str, List[str]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> "OrderedDict[str, Result]":
    """Score one or more pre-computed runs against qrels.

    Each run is ``{query_id: [doc_id, ...ranked]}`` — e.g. the output of an
    existing retrieval pipeline. No retrieval happens here; we only apply the
    from-scratch metrics, so any pipeline can be scored without adopting
    ragroast's retrievers. ``latency_ms`` is left at 0 (not measured here).
    """
    results: "OrderedDict[str, Result]" = OrderedDict()
    for name, ranking in runs.items():
        ndcgs: List[float] = []
        recalls: List[float] = []
        rrs: List[float] = []
        for qid, rel in qrels.items():
            if not rel:
                continue
            docids = ranking.get(qid, [])
            ndcgs.append(ndcg_at_k(docids, rel, k))
            recalls.append(recall_at_k(docids, rel, k))
            rrs.append(reciprocal_rank(docids, rel))
        results[name] = Result(
            name=name,
            ndcg=mean(ndcgs),
            recall=mean(recalls),
            mrr=mean(rrs),
            latency_ms=0.0,
            n_queries=len(ndcgs),
            k=k,
        )
    return results
