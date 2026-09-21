"""Corpus / query / qrels loading.

File formats (BEIR-compatible)
------------------------------
- corpus  : JSONL, one doc per line: ``{"_id": "...", "title": "...", "text": "..."}``
- queries : JSONL, one query per line: ``{"_id": "...", "text": "..."}``
- qrels   : either
      * JSONL: ``{"qid": "...", "docid": "...", "rel": 1}``  (used by the demo), or
      * TSV  : BEIR-style ``query-id<TAB>corpus-id<TAB>score`` (header row tolerated).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Doc:
    id: str
    text: str
    title: str = ""


@dataclass
class Query:
    id: str
    text: str


def _lines(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_corpus_jsonl(path: str) -> List[Doc]:
    docs: List[Doc] = []
    for line in _lines(path):
        o = json.loads(line)
        doc_id = str(o.get("_id", o.get("id")))
        title = (o.get("title") or "").strip()
        text = (o.get("text") or "").strip()
        full = (title + " " + text).strip() if title else text
        docs.append(Doc(id=doc_id, text=full, title=title))
    return docs


def load_queries_jsonl(path: str) -> List[Query]:
    queries: List[Query] = []
    for line in _lines(path):
        o = json.loads(line)
        queries.append(Query(id=str(o.get("_id", o.get("id"))), text=(o.get("text") or "").strip()))
    return queries


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    if str(path).endswith(".tsv"):
        for i, line in enumerate(_lines(path)):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            if i == 0 and not score.strip().lstrip("-").isdigit():
                continue  # header row
            qrels[qid][did] = int(float(score))
    else:
        for line in _lines(path):
            o = json.loads(line)
            qrels[str(o["qid"])][str(o["docid"])] = int(o.get("rel", 1))
    return dict(qrels)


def _sample_base() -> str:
    return os.path.join(os.path.dirname(__file__), "_sample")


def load_sample() -> Tuple[List[Doc], List[Query], Dict[str, Dict[str, int]], str]:
    """Return ``(docs, queries, qrels, vectors_path)`` for the bundled demo set."""
    base = _sample_base()
    return (
        load_corpus_jsonl(os.path.join(base, "corpus.jsonl")),
        load_queries_jsonl(os.path.join(base, "queries.jsonl")),
        load_qrels(os.path.join(base, "qrels.jsonl")),
        os.path.join(base, "vectors.json"),
    )
