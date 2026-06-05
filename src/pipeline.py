"""End-to-end RAG pipeline: retrieve passages, then run extractive QA."""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import Passage
from .qa import Answer, ExtractiveQA
from .retriever import BM25Retriever


@dataclass
class RAGOutput:
    question: str
    retrieved: list[tuple[Passage, float]]
    answer: Answer
    chosen_passage: Passage


class RAG:
    def __init__(self, retriever: BM25Retriever, qa: ExtractiveQA):
        self.retriever = retriever
        self.qa = qa

    def __call__(self, question: str, k: int = 3) -> RAGOutput:
        retrieved = self.retriever.search(question, k=k)
        best_ans: Answer | None = None
        best_passage: Passage | None = None
        for passage, _ in retrieved:
            ans = self.qa.answer(question, passage.joined)
            if best_ans is None or ans.score > best_ans.score:
                best_ans = ans
                best_passage = passage
        assert best_ans is not None and best_passage is not None
        return RAGOutput(
            question=question,
            retrieved=retrieved,
            answer=best_ans,
            chosen_passage=best_passage,
        )
