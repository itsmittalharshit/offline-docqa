import json
import re
from pathlib import Path

from ingest import load_document

EVAL_PATH = Path("eval/questions.jsonl")
RAW_DIR = Path("data/raw")

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"[A-Za-z]{4,}")
STOP = {"from", "with", "that", "this", "their", "which", "into", "than", "under",
        "about", "around", "current", "level", "reduced", "each", "side", "instead"}


def key_terms(answer):
    """Numbers if the answer has any; otherwise a few important English words."""
    nums = [n.rstrip(",.") for n in NUMBER.findall(answer)]
    nums = [n for n in nums if len(n.replace(",", "")) >= 2]
    if nums:
        return nums
    return [w.lower() for w in WORD.findall(answer) if w.lower() not in STOP][:5]

def norm(text):
    """Lowercase and collapse whitespace so line breaks don't break matching."""
    return " ".join(text.lower().split())

def main():
    docs = {}       # doc name -> pages (load each PDF only once)
    problems = 0

    with open(EVAL_PATH, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {line_no}: INVALID JSON -> {e}")
                problems += 1
                continue

            if not item.get("answerable", True):
                continue

            qid, doc = item["id"], item["doc"]
            if doc not in docs:
                docs[doc] = load_document(RAW_DIR / doc)
            pages = docs[doc]

            bad = [p for p in item["pages"]
                   if not isinstance(p, int) or not 1 <= p <= len(pages)]
            if bad:
                print(f"[{qid}] invalid pages {bad} ({doc} has {len(pages)} pages)")
                problems += 1
                continue

            labelled_norm = norm(" ".join(pages[p - 1]["text"] for p in item["pages"]))
            evidence = item.get("evidence")
            if evidence:
                if norm(evidence) not in labelled_norm:
                    where = [p["page"] for p in pages if norm(evidence) in norm(p["text"])]
                    print(f"[{qid}] evidence '{evidence}' not on pages {item['pages']}. Found at: {where}")
                    problems += 1
                continue

            terms = key_terms(item["answer"])
            if not terms:
                print(f"[{qid}] no checkable terms in answer — verify by hand")
                continue

            labelled = " ".join(pages[p - 1]["text"] for p in item["pages"]).lower()
            if not any(t.lower() in labelled for t in terms):
                where = {t: [p["page"] for p in pages if t.lower() in p["text"].lower()][:8]
                         for t in terms}
                print(f"[{qid}] answer terms not on pages {item['pages']}. Found at: {where}")
                problems += 1

    print(f"\n{problems} problem(s) found")


if __name__ == "__main__":
    main()