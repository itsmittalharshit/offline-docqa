import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from device import best_device

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

MODELS = {
    "e5-small": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
}


def index_path(model_key):
    return Path(f"data/processed/index_{model_key}.npz")


def load_chunks(path=CHUNKS_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Embedder:
    def __init__(self, model_key):
        cfg = MODELS[model_key]
        self.model = SentenceTransformer(cfg["name"], device=best_device())
        self.model.max_seq_length = 512
        self.query_prefix = cfg["query_prefix"]
        self.passage_prefix = cfg["passage_prefix"]
        

    def passages(self, texts):
        return self.model.encode(
            [self.passage_prefix + t for t in texts],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def query(self, text):
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)


def load_cache(path):
    if not path.exists():
        return {}
    data = np.load(path)
    return dict(zip(data["ids"], data["vectors"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="e5-small", choices=MODELS)
    args = parser.parse_args()

    path = index_path(args.model)
    chunks = load_chunks()
    cache = load_cache(path)
    missing = [c for c in chunks if c["id"] not in cache]
    print(f"[{args.model}] {len(chunks)} chunks | {len(cache)} cached | {len(missing)} to embed")

    if missing:
        embedder = Embedder(args.model)
        vectors = embedder.passages([c["text"] for c in missing])
        for c, v in zip(missing, vectors):
            cache[c["id"]] = v

    ids = [c["id"] for c in chunks]
    matrix = np.stack([cache[i] for i in ids]).astype(np.float32)
    np.savez(path, ids=np.array(ids), vectors=matrix)
    print(f"Saved index {matrix.shape} to {path}")