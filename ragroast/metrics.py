"""Information-retrieval metrics, implemented from scratch.

The correctness of these three functions *is* the credibility of ragroast, so
they are written plainly and covered by tests in ``tests/test_metrics.py``.

Conventions
-----------
- A *ranking* is a list of document ids, most-relevant first.
- *qrels* for a query is a dict ``{doc_id: relevance}`` where relevance is an
  int ``>= 0`` (0 = not relevant). Binary or graded judgements both work.
"""
from __future__ import annotations

from math import log2
from typing import Dict, Iterable, Sequence


def recall_at_k(ranking: Sequence[str], qrels_q: Dict[str, int], k: int) -> float:
    """Fraction of all relevant documents that appear in the top ``k``."""
    relevant = {d for d, r in qrels_q.items() if r > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranking[:k] if d in relevant)
    return hits / len(relevant)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain over the first ``k`` graded gains.

    Position ``p`` (1-indexed) is discounted by ``log2(p + 1)``.
    """
    return sum(g / log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranking: Sequence[str], qrels_q: Dict[str, int], k: int) -> float:
    """Normalized DCG@k: DCG of the ranking over the DCG of the ideal ranking."""
    gains = [qrels_q.get(d, 0) for d in ranking[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = sorted(qrels_q.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranking: Sequence[str], qrels_q: Dict[str, int]) -> float:
    """1 / rank of the first relevant document (0 if none retrieved)."""
    relevant = {d for d, r in qrels_q.items() if r > 0}
    for i, d in enumerate(ranking):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
