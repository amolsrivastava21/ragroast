"""Render the showdown table and the verdict."""
from __future__ import annotations

import os
import sys
from typing import Dict, List

from .runner import Result


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _paint(s: str, code: str, on: bool) -> str:
    return f"\033[{code}m{s}\033[0m" if on else s


def verdict(results: Dict[str, Result], k: int) -> List[str]:
    bm = results.get("BM25 (baseline)")
    de = results.get("Dense (MiniLM)")
    hy = results.get("Hybrid (RRF)")
    out: List[str] = []

    def pct(a: float, b: float) -> float:
        return (a - b) / b * 100.0 if b > 0 else 0.0

    if bm is not None and de is not None:
        d = pct(de.ndcg, bm.ndcg)
        if de.ndcg < bm.ndcg:
            out.append(f"VERDICT  BM25 beats your dense retriever by +{abs(d):.1f}% nDCG@{k}.  \U0001f525")
        elif de.ndcg > bm.ndcg:
            out.append(f"VERDICT  Your dense retriever beats BM25 by +{d:.1f}% nDCG@{k}.")
        else:
            out.append(f"VERDICT  Dense and BM25 tie on nDCG@{k}.")
        if hy is not None:
            hd = pct(hy.ndcg, bm.ndcg)
            if hy.ndcg >= max(bm.ndcg, de.ndcg):
                out.append(f"         Hybrid (RRF) wins overall (+{hd:.1f}% over BM25).")
            else:
                out.append(f"         Hybrid (RRF): {hd:+.1f}% vs BM25.")
        return out

    if bm is not None:
        out.append("VERDICT  Only the BM25 baseline ran.")
        out.append("         See if your embeddings can beat it:  ragroast demo --dense minilm")
        return out

    # Generic case: scoring external run(s), no BM25 baseline present.
    if not results:
        return out
    ranked = sorted(results.values(), key=lambda r: -r.ndcg)
    best = ranked[0]
    if len(ranked) == 1:
        out.append(
            f"VERDICT  {best.name}: nDCG@{k} {best.ndcg:.3f} · Recall@{k} {best.recall:.3f} · "
            f"MRR {best.mrr:.3f}  (over {best.n_queries} queries)"
        )
    else:
        runner_up = ranked[1]
        lead = pct(best.ndcg, runner_up.ndcg)
        out.append(f"VERDICT  {best.name} wins: nDCG@{k} {best.ndcg:.3f} (+{lead:.1f}% over {runner_up.name}).")
    return out


def render(results: Dict[str, Result], dataset: str, n_docs: int, n_queries: int, k: int, params: str) -> str:
    color = _use_color()
    best_ndcg = max((r.ndcg for r in results.values()), default=0.0)

    header = f"  {'method':<20}{'nDCG@' + str(k):>9}{'Recall@' + str(k):>12}{'MRR':>8}{'latency':>13}"
    rule = "  " + "─" * (len(header) - 2)

    scope = (
        f"{n_docs} docs · {n_queries} queries · k={k}"
        if n_docs > 0
        else f"{n_queries} queries · k={k}"
    )
    lines: List[str] = [
        "",
        f"  ragroast · retrieval showdown on '{dataset}'  ({scope})",
        "",
        header,
        rule,
    ]
    for name, r in results.items():
        cell = f"{r.ndcg:.3f}".rjust(9)
        if color and r.ndcg == best_ndcg and best_ndcg > 0:
            cell = _paint(cell, "1;32", True)
        latency = f"{r.latency_ms:>10.2f} ms" if r.latency_ms > 0 else f"{'—':>13}"
        lines.append(
            f"  {name:<20}{cell}{r.recall:>12.3f}{r.mrr:>8.3f}{latency}"
        )
    lines.append(rule)
    for vl in verdict(results, k):
        lines.append("  " + vl)
    lines.append("")
    lines.append(_paint("  " + params, "2", color))
    lines.append("")
    return "\n".join(lines)
