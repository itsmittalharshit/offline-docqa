import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/index_e5_small.npz")


def load_chunks(path=CHUNKS_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model():
    return SentenceTransformer(MODEL_NAME, device="mps")


def embed_passages(model, texts):
    return model.encode(
        ["passage: " + t for t in texts],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(model, text):
    return model.encode("query: " + text, normalize_embeddings=True)


def load_cache(path=INDEX_PATH):
    if not path.exists():
        return {}
    data = np.load(path)
    return dict(zip(data["ids"], data["vectors"]))


if __name__ == "__main__":
    chunks = load_chunks()
    cache = load_cache()
    missing = [c for c in chunks if c["id"] not in cache]
    print(f"{len(chunks)} chunks | {len(cache)} cached | {len(missing)} to embed")

    if missing:
        model = load_model()
        vectors = embed_passages(model, [c["text"] for c in missing])
        for c, v in zip(missing, vectors):
            cache[c["id"]] = v

    ids = [c["id"] for c in chunks]
    matrix = np.stack([cache[i] for i in ids]).astype(np.float32)
    np.savez(INDEX_PATH, ids=np.array(ids), vectors=matrix)
    print(f"Saved index {matrix.shape} to {INDEX_PATH}")