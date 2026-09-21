"""Command line interface: ``ragroast demo`` / ``run`` / ``score``."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

from . import __version__
from .data import (
    load_corpus_jsonl,
    load_qrels,
    load_queries_jsonl,
    load_run,
    load_sample,
    resolve_dataset_dir,
)
from .embeddings import load_vectors, minilm_embedder
from .report import render
from .runner import run_showdown, score_runs

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


def _roast_level(args: argparse.Namespace) -> int:
    """0 = plain (no roast), 1 = mild (default), 2 = spicy (--roast)."""
    if getattr(args, "plain", False):
        return 0
    if getattr(args, "roast", False):
        return 2
    return 1


def cmd_demo(args: argparse.Namespace) -> int:
    level = _roast_level(args)
    docs, queries, qrels, vectors_path = load_sample()
    dense_vectors, embedder = _dense_setup(args.dense, vectors_path)
    results = run_showdown(
        docs, queries, qrels, k=args.k, dense_vectors=dense_vectors, embedder=embedder, roast_level=level
    )
    print(render(results, "sample (demo)", len(docs), _counted_queries(queries, qrels), args.k, _PARAMS, roast_level=level))
    if dense_vectors is None and embedder is None:
        print("  tip: that was BM25 only. Light up Dense + Hybrid:")
        print("       pip install 'ragroast[dense]'   &&   ragroast demo --dense minilm\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.path:
        corpus_path, queries_path, qrels_path = resolve_dataset_dir(args.path)
        dataset = os.path.basename(os.path.normpath(args.path)) or args.path
    elif args.corpus and args.queries and args.qrels:
        corpus_path, queries_path, qrels_path = args.corpus, args.queries, args.qrels
        dataset = os.path.basename(args.corpus)
    else:
        raise SystemExit(
            "give a dataset directory (e.g. `ragroast run ./data/`) "
            "or all of --corpus / --queries / --qrels"
        )
    docs = load_corpus_jsonl(corpus_path)
    queries = load_queries_jsonl(queries_path)
    qrels = load_qrels(qrels_path)
    # No precomputed vectors for arbitrary corpora; embed on the fly if asked.
    level = _roast_level(args)
    _, embedder = _dense_setup(args.dense, None)
    results = run_showdown(
        docs, queries, qrels, k=args.k, embedder=embedder,
        rrf_k=args.rrf_k, k1=args.k1, b=args.b, roast_level=level,
    )
    print(render(results, dataset, len(docs), _counted_queries(queries, qrels), args.k, _PARAMS, roast_level=level))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    qrels = load_qrels(args.qrels)
    runs = {}
    for run_path in args.run:
        name = args.name if (args.name and len(args.run) == 1) else os.path.basename(run_path)
        runs[name] = load_run(run_path)
    results = score_runs(runs, qrels, k=args.k)
    n_queries = next(iter(results.values())).n_queries if results else 0
    params = f"metrics computed from scratch · nDCG@{args.k}, Recall@{args.k}, MRR · qrels: {os.path.basename(args.qrels)}"
    print(render(results, "your run", 0, n_queries, args.k, params, roast_level=_roast_level(args)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragroast",
        description="Does your RAG actually beat BM25? A fair retrieval showdown.",
        epilog=(
            "examples:\n"
            "  ragroast demo                        # bundled showdown (BM25 vs Dense vs Hybrid)\n"
            "  ragroast demo --dense minilm         # once the dense extra is installed\n"
            "  ragroast run ./my-dataset/ --dense minilm --k 10\n"
            "  ragroast run --corpus c.jsonl --queries q.jsonl --qrels r.jsonl\n"
            "  ragroast score --run my_run.trec --qrels qrels.jsonl\n"
            "\nmetrics need qrels (relevance labels) — no labels, nothing to score.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"ragroast {__version__}")
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("demo", help="run the bundled showdown")
    d.add_argument("--dense", default=None, help="'minilm' to include dense/hybrid (needs ragroast[dense])")
    d.add_argument("-k", "--k", type=int, default=10, help="cutoff k for nDCG/Recall (default 10)")
    d.add_argument("--plain", action="store_true", help="sober output, no roast (good for CI / parsing)")
    d.add_argument("--roast", action="store_true", help="extra-spicy roast")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run on your own data (a dataset dir, or explicit files)")
    r.add_argument("path", nargs="?", help="BEIR-style dataset dir (corpus.jsonl, queries.jsonl, qrels)")
    r.add_argument("--corpus", help="corpus JSONL (BEIR format); omit if passing a dataset dir")
    r.add_argument("--queries", help="queries JSONL (BEIR format)")
    r.add_argument("--qrels", help="qrels JSONL or BEIR TSV")
    r.add_argument("--dense", default=None, help="'minilm' to include dense/hybrid (needs ragroast[dense])")
    r.add_argument("-k", "--k", type=int, default=10)
    r.add_argument("--rrf-k", type=int, default=60, dest="rrf_k")
    r.add_argument("--k1", type=float, default=1.5)
    r.add_argument("--b", type=float, default=0.75)
    r.add_argument("--plain", action="store_true", help="sober output, no roast (good for CI / parsing)")
    r.add_argument("--roast", action="store_true", help="extra-spicy roast")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="score an existing pipeline's run file against qrels")
    s.add_argument("--run", action="append", required=True, metavar="RUN",
                   help="run file (TREC or JSONL); repeat --run to compare several")
    s.add_argument("--qrels", required=True, help="qrels JSONL or BEIR TSV")
    s.add_argument("--name", default=None, help="label for a single run (defaults to the filename)")
    s.add_argument("-k", "--k", type=int, default=10)
    s.add_argument("--plain", action="store_true", help="sober output, no roast (good for CI / parsing)")
    s.add_argument("--roast", action="store_true", help="extra-spicy roast")
    s.set_defaults(func=cmd_score)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not getattr(args, "func", None):  # bare `ragroast` -> show help, don't error
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
