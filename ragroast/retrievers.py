"""Retrievers: BM25 (from scratch), Dense (cosine), and RRF hybrid.

BM25 is deliberately hand-written rather than pulled from ``rank_bm25`` — a
faithful baseline is the whole point of this tool, so it should be readable and
auditable, not a black box.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Sequence

from .data import Doc
from .embeddings import cosine

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_tokenizer(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25.

    score(q, d) = sum over t in q of
        idf(t) * ( f(t,d) * (k1 + 1) ) / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    idf(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )   # Lucene-style, always > 0
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Callable[[str], List[str]] = default_tokenizer):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.doc_ids: List[str] = []
        self.doc_tf: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def index(self, docs: Sequence[Doc]) -> "BM25Retriever":
        self.doc_ids, self.doc_tf, self.doc_len, self.df = [], [], [], {}
        for d in docs:
            toks = self.tokenizer(d.text)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_ids.append(d.id)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()
        }
        return self

    def scores(self, query: str) -> Dict[str, float]:
        q_terms = self.tokenizer(query)
        out: Dict[str, float] = {}
        for i, doc_id in enumerate(self.doc_ids):
            tf = self.doc_tf[i]
            dl = self.doc_len[i]
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + norm)
            if s != 0.0:
                out[doc_id] = s
        return out

    def search(self, query: str, k: int = 10, query_id: Optional[str] = None) -> List[str]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return [d for d, _ in ranked[:k]]


class DenseRetriever:
    """Rank documents by cosine similarity of embeddings.

    Either provide an ``embedder`` (a callable ``texts -> list[vector]``) or
    precomputed ``doc_vectors`` / ``query_vectors`` dicts (the offline demo path).
    """

    def __init__(
        self,
        embedder: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        doc_vectors: Optional[Dict[str, List[float]]] = None,
        query_vectors: Optional[Dict[str, List[float]]] = None,
        label: str = "Dense",
    ):
        self.embedder = embedder
        self.doc_vectors: Dict[str, List[float]] = dict(doc_vectors or {})
        self.query_vectors: Dict[str, List[float]] = dict(query_vectors or {})
        self.name = label
        self.doc_ids: List[str] = []

    def index(self, docs: Sequence[Doc]) -> "DenseRetriever":
        self.doc_ids = [d.id for d in docs]
        if self.embedder is not None and not self.doc_vectors:
            vecs = self.embedder([d.text for d in docs])
            self.doc_vectors = {d.id: v for d, v in zip(docs, vecs)}
        return self

    def _query_vector(self, query_id: Optional[str], query_text: str) -> List[float]:
        if query_id is not None and query_id in self.query_vectors:
            return self.query_vectors[query_id]
        if self.embedder is not None:
            return self.embedder([query_text])[0]
        raise RuntimeError(
            "Dense retriever has neither an embedder nor a precomputed vector "
            "for this query. Build vectors or install ragroast[dense]."
        )

    def search(self, query: str, k: int = 10, query_id: Optional[str] = None) -> List[str]:
        qv = self._query_vector(query_id, query)
        sims = [
            (doc_id, cosine(qv, self.doc_vectors[doc_id]))
            for doc_id in self.doc_ids
            if doc_id in self.doc_vectors
        ]
        sims.sort(key=lambda kv: (-kv[1], kv[0]))
        return [d for d, _ in sims[:k]]


class RRFHybrid:
    """Reciprocal Rank Fusion over several already-indexed retrievers.

    fused(d) = sum over retrievers of 1 / (rrf_k + rank_r(d))   (rank 1-indexed)
    """

    def __init__(self, retrievers: List, rrf_k: int = 60, candidates: int = 100, label: str = "Hybrid (RRF)"):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.name = label

    def index(self, docs: Sequence[Doc]) -> "RRFHybrid":
        for r in self.retrievers:
            r.index(docs)
        return self

    def search(self, query: str, k: int = 10, query_id: Optional[str] = None) -> List[str]:
        fused: Dict[str, float] = {}
        for r in self.retrievers:
            ranking = r.search(query, k=self.candidates, query_id=query_id)
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return [d for d, _ in ranked[:k]]
