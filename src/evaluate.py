import json
import time
from pathlib import Path
import argparse
from retriever import Retriever

EVAL_PATH = Path("eval/questions.jsonl")
KS = (1, 3, 5, 10, 20)


def is_hit(chunk, item):
    if chunk["doc"] != item["doc"]:
        return False
    return any(chunk["page_start"] <= p <= chunk["page_end"] for p in item["pages"])


def first_hit_rank(results, item):
    for rank, (chunk, score) in enumerate(results, start=1):
        if is_hit(chunk, item):
            return rank
    return None


if __name__ == "__main__":
    with open(EVAL_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    items = [it for it in items if it.get("answerable", True)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=20)

    parser.add_argument("--model", default="e5-small")
    parser.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--rerank", default=None, choices=["minilm", "bge-m3"])
    parser.add_argument("--index", default="brute", choices=["brute", "lsh"])
    parser.add_argument("--lsh-bits", type=int, default=8)
    parser.add_argument("--lsh-tables", type=int, default=8)
    args = parser.parse_args()

    retriever = Retriever(args.model, args.mode, rerank_key=args.rerank,
                          candidates=args.candidates, index=args.index,
                          lsh_bits=args.lsh_bits, lsh_tables=args.lsh_tables)
    ranks = []
    times = []
    for it in items:
        t0 = time.perf_counter()
        results = retriever.search(it["question"], k=max(KS))
        times.append(time.perf_counter() - t0)
        rank = first_hit_rank(results, it)
        ranks.append(rank)
        status = f"rank {rank}" if rank else "MISS"
        print(f"[{it['id']}] {status:8} ({it.get('type', '-')}) {it['question']}")

    n = len(ranks)
    print(f"\n{n} questions")
    for k in KS:
        recall = sum(1 for r in ranks if r and r <= k) / n
        print(f"Recall@{k:<2}: {recall:.2f}")
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"MRR     : {mrr:.3f}")
    print(f"Latency : {1000 * sum(times) / len(times):.0f} ms/question")