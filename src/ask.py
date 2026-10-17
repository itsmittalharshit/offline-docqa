import argparse

from generate import NOT_FOUND, answer_question
from retriever import Retriever
from verify import ClaimVerifier

MARKERS = {"verified": "✅", "weak": "⚠️ ", "unsupported": "❌"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", default="bge-m3", choices=["minilm", "bge-m3"])
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    retriever = Retriever("e5-small", "hybrid", rerank_key=args.rerank,
                          candidates=args.candidates)
    verifier = None if args.no_verify else ClaimVerifier()

    while True:
        question = input("\nQuestion (Enter to quit): ").strip()
        if not question:
            break

        r = answer_question(retriever, question)
        if r["refused"]:
            print(f"\n{NOT_FOUND}")
            continue

        print(f"\n{r['answer']}\n")
        for n in r["cited"]:
            chunk, _ = r["results"][n - 1]
            print(f"  [{n}] {chunk['doc']} pp.{chunk['page_start']}-{chunk['page_end']}")
        if r["invalid_citations"]:
            print(f"  ⚠ Invented citation numbers: {r['invalid_citations']}")

        if verifier:
            print("\n  Claim check:")
            for c in verifier.verify(r["answer"], r["results"]):
                print(f"  {MARKERS[c['level']]} {c['score']:.2f}  {c['claim'][:100]}")

        print(f"\n  retrieval {r['retrieval_s']:.1f}s | generation {r['generation_s']:.1f}s "
              f"| {r['tokens_per_s']:.0f} tokens/s")


if __name__ == "__main__":
    main()