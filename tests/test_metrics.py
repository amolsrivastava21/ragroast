from math import log2

from ragroast.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_recall_at_k():
    ranking = ["a", "b", "c", "d"]
    qrels = {"a": 1, "c": 1}
    assert recall_at_k(ranking, qrels, 2) == 0.5   # only 'a' in top-2
    assert recall_at_k(ranking, qrels, 4) == 1.0
    assert recall_at_k(ranking, {}, 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "c"], {"a": 1}) == 0.5
    assert reciprocal_rank(["a"], {"a": 1}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a": 1}) == 0.0


def test_ndcg_known_value():
    ranking = ["a", "b", "c", "d"]
    qrels = {"a": 1, "c": 1}
    # dcg  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # idcg = 1/log2(2) + 1/log2(3) = 1 + 0.6309297535714575
    expected = 1.5 / (1.0 + 1.0 / log2(3))
    assert approx(ndcg_at_k(ranking, qrels, 4), expected)


def test_ndcg_perfect_and_empty():
    assert approx(ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 2), 1.0)
    assert ndcg_at_k(["a"], {}, 5) == 0.0


def test_ndcg_graded_prefers_higher_gain_first():
    # putting the rel=2 doc first must score higher than putting rel=1 first
    q = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], q, 2) > ndcg_at_k(["b", "a"], q, 2)
