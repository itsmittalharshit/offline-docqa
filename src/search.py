import numpy as np

from embed import INDEX_PATH, embed_query, load_chunks, load_model


def main(k=5):
    chunks = {c["id"]: c for c in load_chunks()}
    data = np.load(INDEX_PATH)
    ids, vectors = data["ids"], data["vectors"]
    model = load_model()

    while True:
        query = input("\nQuestion (Enter to quit): ").strip()
        if not query:
            break
        q = embed_query(model, query)
        scores = vectors @ q
        top = np.argsort(-scores)[:k]

        for rank, i in enumerate(top, start=1):
            c = chunks[ids[i]]
            print(f"\n#{rank}  score={scores[i]:.3f}  "
                  f"{c['doc']} p.{c['page_start']}-{c['page_end']}")
            print(c["text"][:300].replace("\n", " ") + " ...")


if __name__ == "__main__":
    main()