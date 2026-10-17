import sys

from ingest import load_document

doc, *terms = sys.argv[1:]
pages = load_document(f"data/raw/{doc}")
for p in pages:
    text = " ".join(p["text"].lower().split())
    if all(t.lower() in text for t in terms):
        i = text.find(terms[0].lower())
        print(f"p.{p['page']}: ...{text[max(0, i - 80): i + 120]}...\n")