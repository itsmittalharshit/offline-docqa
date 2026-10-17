import numpy as np

from retriever import Retriever


def main(k=5):
    retriever = Retriever()
    while True:
        query = input("\nQuestion (Enter to quit): ").strip()
        if not query:
            break
        for rank, (c, score) in enumerate(retriever.search(query, k), start=1):
            print(f"\n#{rank}  score={score:.3f}  "
                  f"{c['doc']} p.{c['page_start']}-{c['page_end']}")
            print(c["text"][:300].replace("\n", " ") + " ...")


if __name__ == "__main__":
    main()