"""Command line interface: ``ragroast demo`` and ``ragroast run``."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

from . import __version__
from .data import load_corpus_jsonl, load_qrels, load_queries_jsonl, load_sample
from .embeddings import load_vectors, minilm_embedder
from .report import render
from .runner import run_showdown

_PARAMS = (
    "method: Okapi BM25 (k1=1.5, b=0.75) · dense: all-MiniLM-L6-v2 · "
    "fusion: RRF (k=60) · metrics computed from scratch"
)


def _dense_setup(dense_arg: Optional[str], vectors_path: Optional[str]) -> Tuple[Optional[dict], Optional[object]]:
    """Resolve the dense path. Prefer offline precomputed vectors; fall back to MiniLM."""
    if dense_arg in (None, "none", "off"):
        vecs = load_vectors(vectors_path)
        return (vecs, None) if vecs else (None, None)
    if dense_arg == "minilm":
        vecs = load_vectors(vectors_path)
        if vecs:
            return vecs, None
        return None, minilm_embedder()
    raise SystemExit(f"unknown --dense value: {dense_arg!r} (use 'minilm' or 'none')")


def _counted_queries(queries, qrels) -> int:
    return len([q for q in queries if qrels.get(q.id)])


def cmd_demo(args: argparse.Namespace) -> int:
    docs, queries, qrels, vectors_path = load_sample()
    dense_vectors, embedder = _dense_setup(args.dense, vectors_path)
    results = run_showdown(docs, queries, qrels, k=args.k, dense_vectors=dense_vectors, embedder=embedder)
    print(render(results, "sample (demo)", len(docs), _counted_queries(queries, qrels), args.k, _PARAMS))
    if dense_vectors is None and embedder is None:
        print("  tip: that was BM25 only. Light up Dense + Hybrid:")
        print("       pip install 'ragroast[dense]'   &&   ragroast demo --dense minilm\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    docs = load_corpus_jsonl(args.corpus)
    queries = load_queries_jsonl(args.queries)
    qrels = load_qrels(args.qrels)
    # No precomputed vectors for arbitrary corpora; embed on the fly if asked.
    _, embedder = _dense_setup(args.dense, None)
    results = run_showdown(
        docs, queries, qrels, k=args.k, embedder=embedder, rrf_k=args.rrf_k, k1=args.k1, b=args.b
    )
    dataset = args.corpus.rsplit("/", 1)[-1]
    print(render(results, dataset, len(docs), _counted_queries(queries, qrels), args.k, _PARAMS))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragroast",
        description="Does your RAG actually beat BM25? A fair retrieval showdown.",
    )
    p.add_argument("--version", action="version", version=f"ragroast {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the bundled showdown")
    d.add_argument("--dense", default=None, help="'minilm' to include dense/hybrid (needs ragroast[dense])")
    d.add_argument("-k", "--k", type=int, default=10, help="cutoff k for nDCG/Recall (default 10)")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run on your own corpus/queries/qrels")
    r.add_argument("--corpus", required=True, help="corpus JSONL (BEIR format)")
    r.add_argument("--queries", required=True, help="queries JSONL (BEIR format)")
    r.add_argument("--qrels", required=True, help="qrels JSONL or BEIR TSV")
    r.add_argument("--dense", default=None, help="'minilm' to include dense/hybrid (needs ragroast[dense])")
    r.add_argument("-k", "--k", type=int, default=10)
    r.add_argument("--rrf-k", type=int, default=60, dest="rrf_k")
    r.add_argument("--k1", type=float, default=1.5)
    r.add_argument("--b", type=float, default=0.75)
    r.set_defaults(func=cmd_run)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
