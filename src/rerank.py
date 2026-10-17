from sentence_transformers import CrossEncoder

RERANKERS = {
    "minilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "bge-m3": "BAAI/bge-reranker-v2-m3",
}


class Reranker:
    def __init__(self, key="minilm"):
        self.model = CrossEncoder(RERANKERS[key], device="mps", max_length=512)

    def rerank(self, query, results, k=5):
        """results: list of (chunk, score) from stage 1. Returns top-k re-sorted."""
        pairs = [(query, chunk["text"]) for chunk, _ in results]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [(chunk, float(s)) for (chunk, _old), s in ranked[:k]]