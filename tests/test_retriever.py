"""Unit tests for the BM25 retriever and tokenizer."""
from src.corpus import Passage
from src.retriever import BM25Retriever, tokenize


def test_tokenize_lowercases_and_extracts_words():
    assert tokenize("Hello, World! 2024.") == ["hello", "world"]


def test_tokenize_keeps_intra_word_hyphen_and_underscore():
    assert tokenize("scikit-learn and gpt_4") == ["scikit-learn", "and", "gpt_4"]


def test_tokenize_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("...!?,:;") == []


def _toy_corpus() -> list[Passage]:
    return [
        Passage("a", "Cats", "Cats are small carnivorous mammals."),
        Passage("b", "Dogs", "Dogs are domesticated descendants of wolves."),
        Passage("c", "Wolves", "Wolves are wild canids; dogs are descended from them."),
    ]


def test_bm25_returns_at_most_k():
    r = BM25Retriever(_toy_corpus())
    assert len(r.search("dogs", k=2)) == 2
    assert len(r.search("dogs", k=10)) == 3


def test_bm25_top_result_is_relevant():
    r = BM25Retriever(_toy_corpus())
    top = r.search("cats", k=1)
    assert top[0][0].pid == "a"


def test_bm25_scores_non_increasing():
    r = BM25Retriever(_toy_corpus())
    scores = [s for _, s in r.search("dogs wolves", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_bm25_search_returns_passages():
    r = BM25Retriever(_toy_corpus())
    for passage, score in r.search("wolves", k=3):
        assert isinstance(passage, Passage)
        assert isinstance(score, float)
