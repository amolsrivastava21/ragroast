"""Run the showdown: index each retriever, score every query, average metrics."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data import Doc, Query
from .metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from .retrievers import BM25Retriever, DenseRetriever, RRFHybrid


@dataclass
class Result:
    name: str
    ndcg: float
    recall: float
    mrr: float
    latency_ms: float
    n_queries: int
    k: int


def _evaluate(name: str, retriever, queries: Sequence[Query], qrels: Dict[str, Dict[str, int]], k: int) -> Result:
    ndcgs: List[float] = []
    recalls: List[float] = []
    rrs: List[float] = []
    t0 = time.perf_counter()
    for q in queries:
        qrels_q = qrels.get(q.id, {})
        if not qrels_q:
            continue
        ranking = retriever.search(q.text, k=k, query_id=q.id)
        ndcgs.append(ndcg_at_k(ranking, qrels_q, k))
        recalls.append(recall_at_k(ranking, qrels_q, k))
        rrs.append(reciprocal_rank(ranking, qrels_q))
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
) -> "OrderedDict[str, Result]":
    results: "OrderedDict[str, Result]" = OrderedDict()

    bm25 = BM25Retriever(k1=k1, b=b).index(docs)
    results["BM25 (baseline)"] = _evaluate("BM25 (baseline)", bm25, queries, qrels, k)

    dense = None
    if dense_vectors is not None:
        dense = DenseRetriever(
            doc_vectors=dense_vectors.get("docs"),
            query_vectors=dense_vectors.get("queries"),
            label="Dense (MiniLM)",
        ).index(docs)
    elif embedder is not None:
        dense = DenseRetriever(embedder=embedder, label="Dense (MiniLM)").index(docs)

    if dense is not None:
        results["Dense (MiniLM)"] = _evaluate("Dense (MiniLM)", dense, queries, qrels, k)
        hybrid = RRFHybrid([bm25, dense], rrf_k=rrf_k)
        results["Hybrid (RRF)"] = _evaluate("Hybrid (RRF)", hybrid, queries, qrels, k)

    return results
