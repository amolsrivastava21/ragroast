"""The roast is personality only — it must never touch the numbers.

These tests pin the outcome→line mapping, the on/off levels, and (critically)
that a plain render keeps the exact same metric values as a roasted one.
"""
from __future__ import annotations

from ragroast.report import _roast_bucket, render, roast
from ragroast.runner import Result


def _r(name: str, ndcg: float) -> Result:
    return Result(name=name, ndcg=ndcg, recall=0.5, mrr=0.5, latency_ms=1.0, n_queries=10, k=10)


def test_buckets_track_the_outcome():
    blowout = {"BM25 (baseline)": _r("BM25 (baseline)", 0.70), "Dense (MiniLM)": _r("Dense (MiniLM)", 0.55)}
    loss = {"BM25 (baseline)": _r("BM25 (baseline)", 0.68), "Dense (MiniLM)": _r("Dense (MiniLM)", 0.60)}
    slim = {"BM25 (baseline)": _r("BM25 (baseline)", 0.664), "Dense (MiniLM)": _r("Dense (MiniLM)", 0.645)}
    win = {"BM25 (baseline)": _r("BM25 (baseline)", 0.60), "Dense (MiniLM)": _r("Dense (MiniLM)", 0.63)}
    assert _roast_bucket(blowout) == "dense_blowout"   # -21%
    assert _roast_bucket(loss) == "dense_loss"         # -12%
    assert _roast_bucket(slim) == "dense_slim"         # -3%
    assert _roast_bucket(win) == "dense_win"           # +5%
    assert _roast_bucket({"BM25 (baseline)": _r("BM25 (baseline)", 0.7)}) == "bm25_only"
    assert _roast_bucket({"my-run": _r("my-run", 0.8)}) == "scored"
    assert _roast_bucket({}) is None


def test_levels_off_mild_spicy():
    res = {"BM25 (baseline)": _r("BM25 (baseline)", 0.70), "Dense (MiniLM)": _r("Dense (MiniLM)", 0.55)}
    assert roast(res, level=0) is None            # plain
    assert roast(res, level=1) is not None        # mild
    assert roast(res, level=2) is not None        # spicy
    assert roast(res, 1) != roast(res, 2)         # mild and spicy differ


def test_roast_is_deterministic():
    res = {"BM25 (baseline)": _r("BM25 (baseline)", 0.70), "Dense (MiniLM)": _r("Dense (MiniLM)", 0.55)}
    assert roast(res, 1) == roast(res, 1)


def test_plain_render_keeps_identical_numbers():
    res = {"BM25 (baseline)": _r("BM25 (baseline)", 0.664), "Dense (MiniLM)": _r("Dense (MiniLM)", 0.645)}
    plain = render(res, "d", 10, 10, 10, "params", roast_level=0)
    roasted = render(res, "d", 10, 10, 10, "params", roast_level=1)
    # the metric values are present and identical in both renders
    for token in ("0.664", "0.645"):
        assert token in plain and token in roasted
    # the roast line appears only when roasting is on
    assert "Photo finish" in roasted
    assert "Photo finish" not in plain
