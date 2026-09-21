#!/usr/bin/env python3
"""Precompute MiniLM embeddings for the bundled demo set.

Run this ONCE (needs the optional dense extra). It writes
``ragroast/_sample/vectors.json`` so that ``ragroast demo`` can show the full
BM25 / Dense / Hybrid verdict fully offline, with no torch at runtime.

    pip install -e ".[dense]"
    python scripts/build_demo_vectors.py
"""
from __future__ import annotations

import json
import os

from ragroast.data import load_sample
from ragroast.embeddings import minilm_embedder

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    docs, queries, _, vectors_path = load_sample()
    embed = minilm_embedder(MODEL)

    doc_vecs = dict(zip([d.id for d in docs], embed([d.text for d in docs])))
    qry_vecs = dict(zip([q.id for q in queries], embed([q.text for q in queries])))

    payload = {"model": MODEL, "docs": doc_vecs, "queries": qry_vecs}
    with open(vectors_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size_kb = os.path.getsize(vectors_path) / 1024
    print(f"wrote {vectors_path}  ({len(doc_vecs)} docs, {len(qry_vecs)} queries, {size_kb:.0f} KB)")
    print("now run:  ragroast demo")


if __name__ == "__main__":
    main()
