"""BM25 retriever over the tiny corpus."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Passage

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self._tokens = [tokenize(p.joined) for p in passages]
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.passages[i], float(scores[i])) for i in top_idx]
