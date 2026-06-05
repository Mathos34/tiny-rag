"""Generate assets/result.png from runs/metrics.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    runs = ROOT / "runs"
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    with open(runs / "metrics.json", encoding="utf-8") as f:
        m = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    names = ["Recall@k", "Exact Match", "Token F1"]
    vals = [m["mean_retrieval_recall_at_k"] * 100,
            m["mean_exact_match"] * 100,
            m["mean_token_f1"] * 100]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("score (%)")
    ax.set_title(f"tiny-rag, k={m['k']}, n_queries={m['n_queries']}")
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    per_q = m["per_query"]
    qs = [f"Q{i+1}" for i in range(len(per_q))]
    f1s = [q["token_f1"] for q in per_q]
    rec = [q["recall_at_k"] for q in per_q]
    x = list(range(len(qs)))
    ax.bar([i - 0.2 for i in x], rec, width=0.4, label="Recall@k", color="#1f77b4")
    ax.bar([i + 0.2 for i in x], f1s, width=0.4, label="Token F1", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(qs, rotation=0, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-query scores")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("tiny-rag: BM25 retrieval + extractive QA on a 30-passage corpus", fontsize=13)
    fig.tight_layout()
    out_path = assets / "result.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
