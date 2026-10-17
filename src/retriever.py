import numpy as np

from embed import Embedder, index_path, load_chunks


class Retriever:
    def __init__(self, model_key="e5-small"):
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        data = np.load(index_path(model_key))
        self.ids = list(data["ids"])
        self.vectors = data["vectors"]
        self.embedder = Embedder(model_key)

    def search(self, query, k=5):
        q = self.embedder.query(query)
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.by_id[self.ids[i]], float(scores[i])) for i in top]