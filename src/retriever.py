import numpy as np

from embed import INDEX_PATH, embed_query, load_chunks, load_model


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        data = np.load(INDEX_PATH)
        self.ids = list(data["ids"])
        self.vectors = data["vectors"]
        self.model = load_model()

    def search(self, query, k=5):
        q = embed_query(self.model, query)
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.by_id[self.ids[i]], float(scores[i])) for i in top]