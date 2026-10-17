import json

WRONG = {"q01", "q03", "q04", "q13", "q14"}   # judged by reading the answers

with open("data/processed/gen_v3.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

# An answer is only as trustworthy as its weakest claim
scored = sorted((min(c["score"] for c in r["claims"]), r["id"])
                for r in rows if r.get("claims"))

print("Lowest claim score per answer:")
for score, qid in scored:
    print(f"  {score:.3f}  {qid}  {'<-- WRONG' if qid in WRONG else ''}")

print("\nthreshold | wrong caught | false alarms")
for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
    caught = sum(1 for s, q in scored if s < t and q in WRONG)
    false = sum(1 for s, q in scored if s < t and q not in WRONG)
    print(f"   {t:.2f}   |    {caught}/{len(WRONG)}       |     {false}")