"""Unit tests for the SQuAD-style scoring helpers."""
from src.corpus import Passage
from src.eval import _normalize_answer, exact_match, f1_score, retrieval_recall


def test_normalize_strips_articles_and_punctuation_and_case():
    assert _normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
    assert _normalize_answer("  A   cat. ") == "cat"


def test_exact_match_handles_normalization():
    assert exact_match("The Cat", "a cat") == 1.0
    assert exact_match("Dogs", "dog") == 0.0


def test_f1_perfect_overlap_is_one():
    assert f1_score("Guido van Rossum", "Guido van Rossum") == 1.0


def test_f1_zero_when_no_overlap():
    assert f1_score("Yann LeCun", "Geoffrey Hinton") == 0.0


def test_f1_partial_overlap():
    # pred "He et al at Microsoft Research" vs gold "He et al"
    # normalized: ["he", "et", "al", "at", "microsoft", "research"] vs ["he", "et", "al"]
    # overlap = 3 tokens; precision = 3/6 = 0.5; recall = 3/3 = 1.0; F1 = 2 * 0.5 / 1.5 = 0.667
    f1 = f1_score("He et al. at Microsoft Research", "He et al.")
    assert abs(f1 - 2 / 3) < 1e-6


def test_f1_empty_prediction_against_nonempty_gold():
    assert f1_score("", "something") == 0.0


def _ranked(pids: list[str]) -> list[tuple[Passage, float]]:
    return [(Passage(pid=p, title="", text=""), 0.0) for p in pids]


def test_retrieval_recall_full_when_all_gold_in_topk():
    out = _ranked(["p01", "p02", "p03"])
    assert retrieval_recall(out, gold_pids={"p01", "p02"}) == 1.0


def test_retrieval_recall_partial():
    out = _ranked(["p01", "p99"])
    assert retrieval_recall(out, gold_pids={"p01", "p02"}) == 0.5


def test_retrieval_recall_zero_when_gold_missing():
    out = _ranked(["p99", "p98"])
    assert retrieval_recall(out, gold_pids={"p01"}) == 0.0


def test_retrieval_recall_with_empty_gold_is_one():
    assert retrieval_recall(_ranked(["p01"]), gold_pids=set()) == 1.0
