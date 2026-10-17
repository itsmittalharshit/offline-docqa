import numpy as np

from bm25 import BM25
from embed import Embedder, index_path, load_chunks


def rrf(result_lists, k=60):
    """Reciprocal Rank Fusion: combine rankings using ranks only."""
    fused = {}
    for results in result_lists:
        for rank, (cid, _score) in enumerate(results, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    def __init__(self, model_key="e5-small", mode="hybrid"):
        self.mode = mode
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}

        data = np.load(index_path(model_key))
        self.ids = [str(i) for i in data["ids"]]
        self.vectors = data["vectors"]
        self.embedder = Embedder(model_key) if mode != "bm25" else None
        self.bm25 = BM25()

    def _dense(self, query, k):
        q = self.embedder.query(query)
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def search(self, query, k=5, pool=50):
        if self.mode == "dense":
            ranked = self._dense(query, k)
        elif self.mode == "bm25":
            ranked = self.bm25.search(query, k)
        else:  # hybrid: each method gives `pool` candidates, fused list cut to k
            ranked = rrf([self._dense(query, pool), self.bm25.search(query, pool)])[:k]
        return [(self.by_id[cid], score) for cid, score in ranked]