import re
import sqlite3
from pathlib import Path

from embed import load_chunks

DB_PATH = Path("data/processed/chunks.db")
WORD = re.compile(r"[A-Za-z0-9]+")


def build_db(chunks, path=DB_PATH):
    path.unlink(missing_ok=True)            # start fresh each time
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            doc TEXT,
            page_start INTEGER,
            page_end INTEGER,
            n_words INTEGER,
            text TEXT
        )
    """)
    con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, text)")

    con.executemany(
        "INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        [(c["id"], c["doc"], c["page_start"], c["page_end"], c["n_words"], c["text"])
         for c in chunks],
    )
    con.executemany(
        "INSERT INTO chunks_fts (id, text) VALUES (?, ?)",
        [(c["id"], c["text"]) for c in chunks],
    )
    con.commit()
    con.close()


def to_fts_query(text):
    """Turn a free-text question into a safe FTS5 query: "word1" OR "word2" ..."""
    words = WORD.findall(text.lower())
    return " OR ".join(f'"{w}"' for w in words)


class BM25:
    def __init__(self, path=DB_PATH):

        self.con = sqlite3.connect(path, check_same_thread=False)

    def search(self, query, k=10):
        fts_query = to_fts_query(query)
        if not fts_query:
            return []
        rows = self.con.execute(
            """
            SELECT id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, k),
        ).fetchall()
        return [(cid, -score) for cid, score in rows]   # flip sign: higher = better


if __name__ == "__main__":
    chunks = load_chunks()
    build_db(chunks)
    con = sqlite3.connect(DB_PATH)
    for doc, n in con.execute("SELECT doc, COUNT(*) FROM chunks GROUP BY doc"):
        print(f"{doc}: {n} chunks")

    bm25 = BM25()
    for cid, score in bm25.search("SDF limit CSF 75 per cent", k=3):
        print(f"{score:.2f}  {cid}")