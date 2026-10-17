from collections import defaultdict

import numpy as np


class LSHIndex:
    """Random-hyperplane LSH for cosine similarity, with multi-probe and centering."""

    def __init__(self, n_bits=8, n_tables=8, multiprobe=True, center=True, seed=0):
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.multiprobe = multiprobe
        self.center = center
        self.seed = seed

    def build(self, vectors):
        self.vectors = vectors
        dim = vectors.shape[1]
        self.mean = vectors.mean(axis=0) if self.center else np.zeros(dim, dtype=vectors.dtype)

        rng = np.random.default_rng(self.seed)
        # one set of n_bits random hyperplanes per table
        self.planes = rng.standard_normal((self.n_tables, self.n_bits, dim)).astype(np.float32)
        self.powers = (1 << np.arange(self.n_bits)).astype(np.int64)   # 1, 2, 4, 8, ...

        centered = vectors - self.mean
        self.tables = []
        for t in range(self.n_tables):
            table = defaultdict(list)
            for i, key in enumerate(self._keys(centered, t)):
                table[int(key)].append(i)
            self.tables.append(table)
        return self

    def _keys(self, X, t):
        """Bucket key for each row of X in table t: the bits packed into one integer."""
        bits = (X @ self.planes[t].T) > 0            # shape (n, n_bits), True/False
        return bits.astype(np.int64) @ self.powers   # e.g. [1,0,1] -> 1 + 4 = 5

    def candidates(self, q):
        qc = (q - self.mean)[None, :]
        found = set()
        for t, table in enumerate(self.tables):
            key = int(self._keys(qc, t)[0])
            probes = [key]
            if self.multiprobe:
                probes += [key ^ int(p) for p in self.powers]   # flip one bit each
            for k in probes:
                found.update(table.get(k, ()))
        return np.fromiter(found, dtype=np.int64)

    def search(self, q, k=10):
        """Return ([(row_index, cosine), ...], number_of_candidates_checked)."""
        cand = self.candidates(q)
        if len(cand) == 0:
            return [], 0
        scores = self.vectors[cand] @ q
        order = np.argsort(-scores)[:k]
        return [(int(cand[i]), float(scores[i])) for i in order], len(cand)