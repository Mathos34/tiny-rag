"""Extractive question answering on top of a retrieved passage.

We load a small pretrained QA head (`deepset/tinyroberta-squad2`, around 80 MB)
and run span extraction manually. We avoid `transformers.pipeline` because the
"question-answering" task was removed from the pipeline registry in recent
transformers releases; using `AutoModelForQuestionAnswering` directly is more
robust across versions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULT_MODEL = "deepset/tinyroberta-squad2"


@dataclass
class Answer:
    text: str
    score: float
    start: int
    end: int


class ExtractiveQA:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 384):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def answer(self, question: str, context: str) -> Answer:
        enc = self.tokenizer(
            question, context,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)
        out = self.model(**enc)
        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]

        # Mask out positions that are not in the context (question tokens, specials).
        mask = torch.tensor([s != 1 for s in sequence_ids], dtype=torch.bool)
        start_logits = start_logits.masked_fill(mask, -1e9)
        end_logits = end_logits.masked_fill(mask, -1e9)

        # Pick best span (start <= end, length <= 30 tokens).
        start_p = F.softmax(start_logits, dim=-1)
        end_p = F.softmax(end_logits, dim=-1)
        best_score = -1.0
        best_s, best_e = 0, 0
        n = start_logits.shape[0]
        for s in range(n):
            if mask[s]:
                continue
            for e in range(s, min(s + 30, n)):
                if mask[e]:
                    continue
                score = float(start_p[s] * end_p[e])
                if score > best_score:
                    best_score = score
                    best_s, best_e = s, e

        if best_score <= 0:
            return Answer(text="", score=0.0, start=0, end=0)

        char_start = offsets[best_s][0]
        char_end = offsets[best_e][1]
        text = context[char_start:char_end].strip()
        return Answer(text=text, score=best_score, start=char_start, end=char_end)
