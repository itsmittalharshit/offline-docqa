import hashlib
import json
import re
import sys
from pathlib import Path
from privacy import Redactor, redact_pages
from ingest import load_document

SENTENCE_SPLIT = re.compile(r"(?<![A-Z])(?<=[.?!])\s+(?=[A-Z(₹\d])")


def split_sentences(paragraph):
    return [s.strip() for s in SENTENCE_SPLIT.split(paragraph) if s.strip()]


def words(text):
    return len(text.split())


def to_units(pages):
    """Flatten a document into (page, paragraph_id, sentence) units."""
    units = []
    para_id = 0
    for p in pages:
        for para in p["paragraphs"]:
            for sent in split_sentences(para):
                units.append((p["page"], para_id, sent))
            para_id += 1
    return units


def overlap_tail(units, overlap_words):
    """Take sentences from the end until we have about overlap_words."""
    tail, total = [], 0
    for unit in reversed(units):
        if total >= overlap_words:
            break
        tail.insert(0, unit)
        total += words(unit[2])
    return tail


def make_chunk(doc_name, units):
    parts = []
    prev_para = None
    for page, para_id, sent in units:
        if prev_para is not None:
            parts.append("\n\n" if para_id != prev_para else " ")
        parts.append(sent)
        prev_para = para_id
    text = "".join(parts)

    pages = [u[0] for u in units]
    chunk_id = hashlib.sha256(f"{doc_name}|{text}".encode()).hexdigest()[:16]
    return {
        "id": chunk_id,
        "doc": doc_name,
        "page_start": min(pages),
        "page_end": max(pages),
        "n_words": words(text),
        "text": text,
    }


def chunk_document(doc_name, pages, max_words=250, overlap_words=50):
    chunks = []
    current, current_words = [], 0
    for unit in to_units(pages):
        n = words(unit[2])
        if current and current_words + n > max_words:
            chunks.append(make_chunk(doc_name, current))
            current = overlap_tail(current, overlap_words)
            current_words = sum(words(u[2]) for u in current)
        current.append(unit)
        current_words += n
    if current:
        chunks.append(make_chunk(doc_name, current))
    return chunks


if __name__ == "__main__":
    redact = "--no-redact" not in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    redactor = Redactor() if redact else None
    raw_dir = Path("data/raw")
    out_path = Path("data/processed/chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    pdfs = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".pdf")
    for pdf in pdfs:
        print(f"Processing {pdf.name} ...")
        pages = load_document(pdf)
        low = sum(p["low_text"] for p in pages)
        if low / len(pages) > 0.8:
            print(f"  WARNING: {low}/{len(pages)} pages have almost no text "
                  f"— file may be corrupt or scanned")
        if redactor:
            counts = redact_pages(pages, redactor)
            print(f"  redacted: {dict(counts) if counts else 'nothing found'}")
        chunks = chunk_document(pdf.name, pages)
        if not chunks:
            print(f"  WARNING: no text extracted from {pdf.name}, skipped")
            continue
        all_chunks.extend(chunks)
        avg = sum(c["n_words"] for c in chunks) / len(chunks)
        print(f"  {len(chunks)} chunks, avg {avg:.0f} words")

    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {out_path}")

    show = int(positional[0]) if positional else 0
    c = all_chunks[show]
    print(f"\n---- Chunk {show} | {c['doc']} p.{c['page_start']}-{c['page_end']} "
          f"| {c['n_words']} words ----")
    print(c["text"])