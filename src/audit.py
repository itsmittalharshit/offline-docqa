import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_PATH = Path("data/logs/audit.jsonl")


def log_interaction(question, answer, r, claims, config):
    """Append one redacted record per question. question/answer must already be redacted."""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "question": question,
        "answer": answer,
        "refused": r["refused"],
        "retrieved": [{"id": c["id"], "doc": c["doc"],
                       "pages": [c["page_start"], c["page_end"]], "score": round(s, 4)}
                      for c, s in r["results"]],
        "cited": r["cited"],
        "invalid_citations": r["invalid_citations"],
        "claims": [{"level": c["level"], "score": c["score"]} for c in claims],
        "retrieval_s": round(r["retrieval_s"], 2),
        "generation_s": round(r["generation_s"], 2),
    }
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")