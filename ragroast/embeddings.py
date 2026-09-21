"""Embedding helpers. Pure-stdlib cosine; MiniLM is an optional extra."""
from __future__ import annotations

import json
import math
import os
import sys
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
    if sys.stderr.isatty():
        sys.stderr.write(f"  loading {model_name} (first run downloads ~90 MB) …\n")
        sys.stderr.flush()
    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> List[List[float]]:
        # sentence-transformers ships its own batch progress bar; show it only for
        # a real batch in an interactive terminal, so piped/captured output stays
        # clean and stray single-item encodes don't stomp on the scoring line.
        texts = list(texts)
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=sys.stderr.isatty() and len(texts) >= 16,
        )
        return [[float(x) for x in v] for v in vecs]

    return embed


def load_vectors(path: Optional[str]) -> Optional[Dict]:
    """Load a precomputed vectors file: {"model", "docs": {...}, "queries": {...}}."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
