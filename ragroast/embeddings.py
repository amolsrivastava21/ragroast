"""Embedding helpers. Pure-stdlib cosine; MiniLM is an optional extra."""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def minilm_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Callable[[Sequence[str]], List[List[float]]]:
    """Return an embed function backed by sentence-transformers (optional dep)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "Dense retrieval needs the optional dependency:\n"
            "    pip install 'ragroast[dense]'\n"
            "(then all-MiniLM-L6-v2 downloads once, ~90 MB)."
        )
    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> List[List[float]]:
        vecs = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=False)
        return [[float(x) for x in v] for v in vecs]

    return embed


def load_vectors(path: Optional[str]) -> Optional[Dict]:
    """Load a precomputed vectors file: {"model", "docs": {...}, "queries": {...}}."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
