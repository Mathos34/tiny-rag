"""Evaluation metrics for tiny-rag: retrieval recall and answer F1."""
from __future__ import annotations

import re
import string
from collections import Counter

from .corpus import Passage


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 used by SQuAD."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_recall(retrieved: list[tuple[Passage, float]], gold_pids: set[str]) -> float:
    if not gold_pids:
        return 1.0
    hit = sum(1 for p, _ in retrieved if p.pid in gold_pids)
    return min(1.0, hit / len(gold_pids))
