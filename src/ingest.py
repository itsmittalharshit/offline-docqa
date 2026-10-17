import sys
import pymupdf

def extract_pdf(path):
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i+1 , "text": text})
    doc.close()
    return pages

if __name__ == "__main__":
    path = sys.argv[1]
    pages = extract_pdf(path)
    print(f"Pages : {len(pages)}")
    print("---first page---")
    print(pages[13]["text"][:1000])
