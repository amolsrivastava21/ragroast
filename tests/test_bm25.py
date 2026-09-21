from ragroast.data import Doc
from ragroast.retrievers import BM25Retriever


def test_bm25_matches_documents_with_query_terms():
    docs = [
        Doc("d1", "the quick brown fox"),
        Doc("d2", "lazy dog sleeps all day"),
        Doc("d3", "quick fox runs"),
    ]
    bm = BM25Retriever().index(docs)
    r = bm.search("quick fox", k=3)
    assert set(r) == {"d1", "d3"}      # d2 shares no query term, scores 0
    assert "d2" not in r


def test_bm25_length_normalization_prefers_shorter_doc():
    # both docs contain 'quick' and 'fox' once; the shorter doc should win
    docs = [
        Doc("short", "quick fox"),
        Doc("long", "quick fox " + "padding word " * 20),
    ]
    bm = BM25Retriever().index(docs)
    assert bm.search("quick fox", k=2)[0] == "short"


def test_bm25_idf_prefers_rarer_term():
    docs = [Doc("d1", "apple apple apple"), Doc("d2", "apple banana")]
    bm = BM25Retriever().index(docs)
    assert bm.search("banana", k=1) == ["d2"]


def test_bm25_empty_query_returns_nothing():
    bm = BM25Retriever().index([Doc("d1", "hello world")])
    assert bm.search("", k=5) == []
