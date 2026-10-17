import argparse
import json
import time
from pathlib import Path

try:
    import faiss   # import before torch: both bundle OpenMP and the wrong order can crash on macOS
except ImportError:
    faiss = None

import numpy as np

from lsh import LSHIndex

INDEX_PATH = Path("data/processed/index_e5-small.npz")
EVAL_PATH = Path("eval/questions.jsonl")
QUERY_CACHE = Path("data/processed/bench_queries.npy")


def make_corpus(real, n, noise, rng):
    """n synthetic vectors: random mixes of two real chunk vectors plus a little noise."""
    i = rng.integers(0, len(real), n)
    j = rng.integers(0, len(real), n)
    w = rng.random((n, 1)).astype(np.float32)
    X = w * real[i] + (1 - w) * real[j]
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def load_queries():
    """Embed the eval questions once and cache them, so later runs never load PyTorch."""
    if QUERY_CACHE.exists():
        return np.load(QUERY_CACHE)
    from embed import Embedder
    with open(EVAL_PATH, encoding="utf-8") as f:
        questions = [json.loads(line)["question"] for line in f if line.strip()]
    embedder = Embedder("e5-small")
    Q = np.stack([embedder.query(q) for q in questions]).astype(np.float32)
    np.save(QUERY_CACHE, Q)
    return Q


def run(search_fn, X, Q, truth, kth, k, tol=0.01):
    strict, tolerant, times = [], [], []
    for q, true_ids, kth_score in zip(Q, truth, kth):
        t0 = time.perf_counter()
        found = search_fn(q)
        times.append(time.perf_counter() - t0)
        strict.append(len(set(found) & true_ids) / k)
        if found:
            good = (X[found] @ q) >= kth_score - tol
            tolerant.append(good.sum() / k)
        else:
            tolerant.append(0.0)
    return np.mean(strict), np.mean(tolerant), 1000 * np.mean(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    k = args.k

    rng = np.random.default_rng(0)
    real = np.load(INDEX_PATH)["vectors"]
    X = make_corpus(real, args.n, args.noise, rng)
    Q = load_queries()

    truth, kth = [], []
    for q in Q:
        s = X @ q
        top = np.argpartition(-s, k)[:k]
        truth.append(set(top.tolist()))
        kth.append(s[top].min())

    print(f"\nCorpus {args.n:,} x {X.shape[1]} (mixed pairs, noise {args.noise}) "
          f"| {len(Q)} real questions | k={k}\n")
    print(f"{'method':30} {'strict':>7} {'tol.':>6} {'ms/query':>9} {'cand.':>7} {'build s':>8}")

    def brute(q):
        return np.argpartition(-(X @ q), k)[:k].tolist()

    s_rec, t_rec, ms = run(brute, X, Q, truth, kth, k)
    print(f"{'brute force (NumPy)':30} {s_rec:7.3f} {t_rec:6.3f} {ms:9.2f} {'100%':>7} {'-':>8}")

    configs = [(b, t, True) for b in (8, 12, 16) for t in (4, 8, 16)] + [(12, 8, False)]
    for bits, tables, center in configs:
        t0 = time.perf_counter()
        idx = LSHIndex(bits, tables, center=center).build(X)
        build_s = time.perf_counter() - t0

        n_cand = []

        def lsh(q):
            hits, n = idx.search(q, k)
            n_cand.append(n)
            return [i for i, _ in hits]

        s_rec, t_rec, ms = run(lsh, X, Q, truth, kth, k)
        frac = 100 * np.mean(n_cand) / args.n
        name = f"LSH b={bits} L={tables}" + ("" if center else " no-center")
        print(f"{name:30} {s_rec:7.3f} {t_rec:6.3f} {ms:9.2f} {frac:6.1f}% {build_s:8.1f}")

    if faiss is None:
        print("\n(faiss not installed: python -m pip install faiss-cpu)")
        return

    t0 = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.add(X)
    build_s = time.perf_counter() - t0
    hnsw.hnsw.efSearch = 64

    def faiss_search(q):
        _, I = hnsw.search(q[None, :], k)
        return [int(i) for i in I[0] if i >= 0]

    s_rec, t_rec, ms = run(faiss_search, X, Q, truth, kth, k)
    print(f"{'FAISS HNSW (M=32, ef=64)':30} {s_rec:7.3f} {t_rec:6.3f} {ms:9.2f} {'-':>7} {build_s:8.1f}")


if __name__ == "__main__":
    main()