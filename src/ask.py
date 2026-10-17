import argparse

from audit import log_interaction
from generate import LLM_MODEL, NOT_FOUND, answer_question
from privacy import Redactor
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
    redactor = Redactor()
    config = {"llm": LLM_MODEL, "prompt": "v3", "embedder": "e5-small",
              "rerank": args.rerank, "candidates": args.candidates,
              "verify": verifier is not None}

    while True:
        question = input("\nQuestion (Enter to quit): ").strip()
        if not question:
            break

        r = answer_question(retriever, question)
        claims = []

        if r["refused"]:
            print(f"\n{NOT_FOUND}")
        else:
            print(f"\n{r['answer']}\n")
            for n in r["cited"]:
                chunk, _ = r["results"][n - 1]
                print(f"  [{n}] {chunk['doc']} pp.{chunk['page_start']}-{chunk['page_end']}")
            if r["invalid_citations"]:
                print(f"  ⚠ Invented citation numbers: {r['invalid_citations']}")
            if verifier:
                claims = verifier.verify(r["answer"], r["results"])
                print("\n  Claim check:")
                for c in claims:
                    print(f"  {MARKERS[c['level']]} {c['score']:.2f}  {c['claim'][:100]}")
            print(f"\n  retrieval {r['retrieval_s']:.1f}s | generation {r['generation_s']:.1f}s "
                  f"| {r['tokens_per_s']:.0f} tokens/s")

        log_interaction(redactor.redact(question)[0], redactor.redact(r["answer"])[0],
                        r, claims, config)


if __name__ == "__main__":
    main()