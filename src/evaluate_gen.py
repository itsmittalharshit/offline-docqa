import argparse
import json
from pathlib import Path
from generate import answer_question, main_script
from check_eval import key_terms
from evaluate import EVAL_PATH, is_hit
from retriever import Retriever
from verify import ClaimVerifier

OUT_DIR = Path("data/processed")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v1", choices=["v1", "v2", "v3"])
    parser.add_argument("--rerank", default="bge-m3", choices=["minilm", "bge-m3"])
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    with open(EVAL_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    items = items[:args.limit]

    retriever = Retriever("e5-small", "hybrid", rerank_key=args.rerank,
                          candidates=args.candidates)
    verifier = ClaimVerifier() if args.verify else None
    answer_question(retriever, items[0]["question"], prompt_version=args.prompt)
    rows = []
    for it in items:
        r = answer_question(retriever, it["question"], prompt_version=args.prompt)
        answerable = it.get("answerable", True)
        row = {"id": it["id"], "type": it.get("type"), "answerable": answerable,
               "question": it["question"], "gold": it.get("answer"),
               "answer": r["answer"], "refused": r["refused"],
               "generation_s": r["generation_s"]}

        if answerable and not r["refused"]:
               terms = key_terms(it["answer"])
               text = r["answer"].lower().replace(",", "")
               row["correct"] = (any(t.lower().replace(",", "") in text for t in terms)
                                 if terms else None)
               row["language_ok"] = main_script(r["answer"]) == main_script(it["question"])
               row["invalid_citations"] = bool(r["invalid_citations"])

               cited = [r["results"][n - 1][0] for n in r["cited"]]
               hits = [is_hit(c, it) for c in cited]
               row["cites_correct_page"] = any(hits)
               row["citation_precision"] = sum(hits) / len(hits) if hits else 0.0
               if verifier:
                   claims = verifier.verify(r["answer"], r["results"])
                   row["claims"] = claims
                   row["has_unsupported"] = any(not c["supported"] for c in claims)

        rows.append(row)

        if r["refused"]:
            flag = "REFUSED"
        elif row.get("language_ok") is False:
            flag = "LANG"
        elif row.get("has_unsupported"):
            flag = "UNSUP"
        else:
            flag = "ok" if row.get("correct") else "check"
        print(f"[{it['id']}] {flag:8} ({row['type']}) {r['answer'][:90]!r}")
        for c in row.get("claims", []):
            if not c["supported"]:
                print(f"      ✗ {c['score']:.2f}  {c['claim'][:100]}")

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable_rows if not r["refused"]]

    print(f"\nprompt={args.prompt} | {len(answerable_rows)} answerable, "
          f"{len(unanswerable_rows)} unanswerable")
    print(f"False-refusal rate        : {mean([r['refused'] for r in answerable_rows]):.2f}")
    if unanswerable_rows:
        print(f"Hallucination (unanswer.) : "
              f"{mean([not r['refused'] for r in unanswerable_rows]):.2f}")
    print(f"Answer correctness        : {mean([r.get('correct') for r in answered]):.2f}")
    print(f"Cites a correct page      : {mean([r.get('cites_correct_page') for r in answered]):.2f}")
    print(f"Citation precision        : {mean([r.get('citation_precision') for r in answered]):.2f}")
    print(f"Language matches question : {mean([r.get('language_ok') for r in answered]):.2f}")
    print(f"Answers w/ invented cites : {mean([r.get('invalid_citations') for r in answered]):.2f}")
    if args.verify:
        print(f"Answers w/ unsupported    : {mean([r.get('has_unsupported') for r in answered]):.2f}")
    print(f"Avg generation time       : {mean([r['generation_s'] for r in rows]):.1f} s")

    out = OUT_DIR / f"gen_{args.prompt}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved answers to {out}")


if __name__ == "__main__":
    main()