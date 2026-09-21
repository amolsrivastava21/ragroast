"""Tests for run-file loading, dataset-dir resolution, and external-run scoring."""
from __future__ import annotations

import json
from math import log2

from ragroast.data import load_run, resolve_dataset_dir
from ragroast.runner import score_runs


def test_load_run_trec(tmp_path):
    p = tmp_path / "run.trec"
    p.write_text(
        "q1 Q0 a 1 5.0 sys\n"
        "q1 Q0 b 2 4.0 sys\n"
        "q1 Q0 x 3 3.0 sys\n"
        "q2 Q0 c 1 9.0 sys\n",
        encoding="utf-8",
    )
    run = load_run(str(p))
    assert run == {"q1": ["a", "b", "x"], "q2": ["c"]}


def test_load_run_jsonl(tmp_path):
    p = tmp_path / "run.jsonl"
    p.write_text(
        json.dumps({"qid": "q1", "docids": ["a", "b", "x"]}) + "\n"
        + json.dumps({"qid": "q2", "docids": ["c"]}) + "\n",
        encoding="utf-8",
    )
    assert load_run(str(p)) == {"q1": ["a", "b", "x"], "q2": ["c"]}


def test_load_run_trec_out_of_order_sorts_by_rank(tmp_path):
    p = tmp_path / "run.trec"
    # deliberately shuffled rank order in the file
    p.write_text("q1 Q0 b 2 4.0 sys\nq1 Q0 a 1 5.0 sys\n", encoding="utf-8")
    assert load_run(str(p)) == {"q1": ["a", "b"]}


def test_resolve_dataset_dir(tmp_path):
    (tmp_path / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    qdir = tmp_path / "qrels"
    qdir.mkdir()
    (qdir / "test.tsv").write_text("query-id\tcorpus-id\tscore\n", encoding="utf-8")

    corpus, queries, qrels = resolve_dataset_dir(str(tmp_path))
    assert corpus.endswith("corpus.jsonl")
    assert queries.endswith("queries.jsonl")
    assert qrels.endswith("test.tsv")


def test_score_runs_matches_hand_computed():
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"c": 1}}
    run = {"q1": ["a", "x", "b"], "q2": ["y", "c"]}

    results = score_runs({"mine": run}, qrels, k=10)
    r = results["mine"]

    # q1: dcg = 1/log2(2) + 1/log2(4) = 1.5 ; idcg = 1/log2(2) + 1/log2(3)
    ndcg_q1 = 1.5 / (1.0 + 1.0 / log2(3))
    # q2: dcg = 1/log2(3) ; idcg = 1/log2(2) = 1
    ndcg_q2 = (1.0 / log2(3)) / 1.0
    expected_ndcg = (ndcg_q1 + ndcg_q2) / 2

    assert abs(r.ndcg - expected_ndcg) < 1e-9
    assert abs(r.recall - 1.0) < 1e-9          # both relevant sets fully retrieved in top-10
    assert abs(r.mrr - (1.0 + 0.5) / 2) < 1e-9  # first-relevant at ranks 1 and 2
    assert r.n_queries == 2
    assert r.latency_ms == 0.0


def test_score_runs_missing_query_scores_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": ["a"]}  # q2 absent from the run
    r = score_runs({"partial": run}, qrels, k=10)["partial"]
    # q1 perfect (1.0), q2 nothing retrieved (0.0) -> mean 0.5 across the board
    assert abs(r.ndcg - 0.5) < 1e-9
    assert abs(r.recall - 0.5) < 1e-9
    assert abs(r.mrr - 0.5) < 1e-9
