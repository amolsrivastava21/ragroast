"""ragroast — does your RAG actually beat BM25?

A fair, reproducible retrieval showdown: BM25 vs your dense embeddings vs a
hybrid (RRF) fusion, scored with from-scratch IR metrics (nDCG, Recall, MRR).

The core is pure standard library. Dense retrieval is an optional extra:
    pip install "ragroast[dense]"
"""

__version__ = "0.1.0"

from .retrievers import BM25Retriever, DenseRetriever, RRFHybrid  # noqa: E402,F401
from .runner import Result, run_showdown  # noqa: E402,F401

__all__ = [
    "__version__",
    "BM25Retriever",
    "DenseRetriever",
    "RRFHybrid",
    "Result",
    "run_showdown",
]
