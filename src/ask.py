import argparse

from generate import answer_question
from retriever import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", default="bge-m3", choices=["minilm", "bge-m3"])
    parser.add_argument("--candidates", type=int, default=10)
    args = parser.parse_args()

    retriever = Retriever("e5-small", "hybrid", rerank_key=args.rerank,
                          candidates=args.candidates)

    while True:
        question = input("\nQuestion (Enter to quit): ").strip()
        if not question:
            break

        r = answer_question(retriever, question)
        print(f"\n{r['answer']}\n")

        for n in r["cited"]:
            chunk, _ = r["results"][n - 1]
            print(f"  [{n}] {chunk['doc']} pp.{chunk['page_start']}-{chunk['page_end']}")

        if r["invalid_citations"]:
            print(f"  ⚠ Invented citation numbers: {r['invalid_citations']}")
        if not r["cited"] and not r["refused"]:
            print("  ⚠ Answer has no citations — treat with caution")

        print(f"\n  retrieval {r['retrieval_s']:.1f}s | generation {r['generation_s']:.1f}s "
              f"| {r['tokens_per_s']:.0f} tokens/s")


if __name__ == "__main__":
    main()