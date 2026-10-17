import re
import sys
from collections import Counter
import pymupdf

PAGE_NUMBER = re.compile(r"^(\d+|[ivxlcdm]+)$",re.IGNORECASE)
SENTENCE_END = (".", "?", "!", ":", ";")
LIST_START = re.compile(r"^(\(?[a-z0-9]{1,4}\)|\d+\.|[IVX]+\.\d+\.|•)\s", re.IGNORECASE)
CAPTION = re.compile(r"^(Chart|Table|Figure)\s+[\w.-]+\s*:", re.IGNORECASE)

def clean_block(raw):
    lines = [line.strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]
    # empty raw
    if not lines:
        return None

    # single words many lines
    avg_words = sum(len(line.split()) for line in lines)/len(lines)
    if len(lines)>=5 and avg_words<1.5:
        return None

    text = " ".join(lines)
    text = re.sub(r"[.…]{4,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if PAGE_NUMBER.match(text):
        return None
    return text

def extract_pdf(path):
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        blocks = []
        for block in page.get_text("blocks"):
            raw_text, block_type = block[4], block[6]
            if block_type != 0:
                continue
            text = clean_block(raw_text)
            if(text):
                blocks.append(text)
        pages.append({"page":i+1,"blocks":blocks})
    doc.close()
    return pages

def normalize(text):
    return re.sub(r"\d+", "#", text.lower())

def remove_repeated(pages, threshold = 0.3):
    counts = Counter()
    for p in pages:
        counts.update({normalize(b) for b in p["blocks"]})
    limit = threshold*len(pages)
    repeated = {key for key, c in counts.items()
                if c > limit and len(key.split()) < 15}

    for p in pages:
        p["blocks"] = [b for b in p["blocks"] if normalize(b) not in repeated]
    return repeated

def is_mostly_numbers(text, min_letter_ratio=0.3):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    letters = sum(c.isalpha() for c in chars)
    return letters / len(chars) < min_letter_ratio


def build_paragraphs(blocks):
    paragraphs = []
    for block in blocks:
        starts_new = (
            not paragraphs
            or paragraphs[-1].endswith(SENTENCE_END)
            or LIST_START.match(block)
            or CAPTION.match(block)
            or CAPTION.match(paragraphs[-1])
        )
        if starts_new:
            paragraphs.append(block)
        else:
            paragraphs[-1] += " " + block
    return [p for p in paragraphs if not is_mostly_numbers(p)]

def load_document(path):
    """Full pipeline: extract -> remove headers -> rebuild paragraphs."""
    pages = extract_pdf(path)
    remove_repeated(pages)
    for p in pages:
        p["paragraphs"] = build_paragraphs(p["blocks"])
        p["text"] = "\n\n".join(p["paragraphs"])
        p["low_text"] = len(p["text"]) < 100
    return pages

if __name__ == "__main__":
    path = sys.argv[1]
    page_to_show = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    pages = load_document(path)
    print(f"Pages: {len(pages)}")
    print(f"Low-text pages: {[p['page'] for p in pages if p['low_text']]}")
    print(f"---- Page {page_to_show} ----")
    print(pages[page_to_show - 1]["text"])