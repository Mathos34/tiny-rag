"""Sanity tests for the bundled corpus."""
from src.corpus import PASSAGES


def test_corpus_size_is_30():
    assert len(PASSAGES) == 30


def test_passage_ids_are_unique():
    ids = [p.pid for p in PASSAGES]
    assert len(ids) == len(set(ids))


def test_every_passage_has_non_empty_title_and_text():
    for p in PASSAGES:
        assert p.title.strip() != ""
        assert len(p.text) > 20


def test_joined_concatenates_title_and_text():
    p = PASSAGES[0]
    j = p.joined
    assert p.title in j
    assert p.text in j
