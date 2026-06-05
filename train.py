"""Build the BM25 index over the hand-curated corpus, then run the RAG eval.

There is no training step in classical RAG: the retriever is BM25 (no learnable
parameters) and the QA head is a pretrained `deepset/tinyroberta-squad2`. We
still call this `train.py` to match the portfolio convention; in practice it is
"build and evaluate".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.corpus import PASSAGES
from src.eval import exact_match, f1_score, retrieval_recall
from src.pipeline import RAG
from src.qa import ExtractiveQA
from src.retriever import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3,
                        help="Number of passages to retrieve per query.")
    parser.add_argument("--qa-pairs", type=str, default="data/qa_pairs.json")
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Corpus: {len(PASSAGES)} passages")
    retriever = BM25Retriever(PASSAGES)

    with open(args.qa_pairs, encoding="utf-8") as f:
        qa = json.load(f)
    print(f"QA pairs: {len(qa)}")

    print("Loading extractive QA model (deepset/tinyroberta-squad2)...")
    t0 = time.time()
    qa_model = ExtractiveQA()
    print(f"  loaded in {time.time()-t0:.1f}s")
    rag = RAG(retriever, qa_model)

    per_query = []
    recalls, ems, f1s = [], [], []
    print("Running end-to-end eval...")
    for item in qa:
        out = rag(item["question"], k=args.k)
        gold_set = set(item["gold_pids"])
        recall = retrieval_recall(out.retrieved, gold_set)
        em = exact_match(out.answer.text, item["answer"])
        f1 = f1_score(out.answer.text, item["answer"])
        recalls.append(recall)
        ems.append(em)
        f1s.append(f1)
        per_query.append({
            "question": item["question"],
            "gold_answer": item["answer"],
            "gold_pids": item["gold_pids"],
            "retrieved": [{"pid": p.pid, "title": p.title, "score": s} for p, s in out.retrieved],
            "chosen_pid": out.chosen_passage.pid,
            "predicted_answer": out.answer.text,
            "qa_score": out.answer.score,
            "recall_at_k": recall,
            "exact_match": em,
            "token_f1": f1,
        })
        marker = "OK" if em else ("partial" if f1 > 0.5 else "MISS")
        print(f"  [{marker}] q='{item['question']}' -> '{out.answer.text}' (gold='{item['answer']}', F1={f1:.2f})")

    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    metrics = {
        "k": args.k,
        "n_queries": len(qa),
        "mean_retrieval_recall_at_k": mean(recalls),
        "mean_exact_match": mean(ems),
        "mean_token_f1": mean(f1s),
        "per_query": per_query,
    }
    out_path = out_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")
    print(f"Recall@{args.k} = {metrics['mean_retrieval_recall_at_k']*100:.1f}%, "
          f"EM = {metrics['mean_exact_match']*100:.1f}%, "
          f"F1 = {metrics['mean_token_f1']*100:.1f}%")


if __name__ == "__main__":
    main()
