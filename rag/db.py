"""
db.py — SQLite-backed document and chunk store.
Stores document metadata and text chunks with their TF-IDF vectors.
No external vector DB needed; everything lives in a single SQLite file.
"""
import sqlite3
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional


DB_PATH = os.path.join(
    os.environ.get("ANDROID_PRIVATE", os.path.expanduser("~")),
    "ragapp.db",
)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")   # faster concurrent writes
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")    # enable CASCADE deletes
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                path      TEXT NOT NULL UNIQUE,
                added_at  TEXT DEFAULT (datetime('now')),
                num_chunks INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_idx INTEGER NOT NULL,
                text      TEXT NOT NULL,
                tokens    TEXT,          -- JSON list of lowercase tokens for BM25
                tfidf_vec BLOB           -- pickled dict {term: tf_idf_score}
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )


# ---------- document helpers ----------

def insert_document(name: str, path: str) -> int:
    with get_conn() as conn:
        # Check if this path already exists — if so, delete its old chunks
        # so re-uploading the same file doesn't accumulate duplicates.
        existing = conn.execute(
            "SELECT id FROM documents WHERE path=?", (path,)
        ).fetchone()
        if existing:
            doc_id = existing[0]
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            return doc_id
        cur = conn.execute(
            "INSERT INTO documents(name, path) VALUES (?, ?)", (name, path)
        )
        return cur.lastrowid


def update_doc_chunk_count(doc_id: int, count: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET num_chunks=? WHERE id=?", (count, doc_id)
        )


def list_documents() -> List[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, path, added_at, num_chunks FROM documents ORDER BY added_at DESC"
        ).fetchall()
    return [
        {"id": r[0], "name": r[1], "path": r[2], "added_at": r[3], "num_chunks": r[4]}
        for r in rows
    ]


def delete_document(doc_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))


# ---------- chunk helpers ----------

def insert_chunks(doc_id: int, chunks: List[dict]) -> None:
    """
    chunks: list of dicts with keys:
        chunk_idx, text, tokens (list[str]), tfidf_vec (dict)
    """
    rows = [
        (
            doc_id,
            c["chunk_idx"],
            c["text"],
            json.dumps(c["tokens"]),
            pickle.dumps(c["tfidf_vec"]),
        )
        for c in chunks
    ]
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO chunks(doc_id, chunk_idx, text, tokens, tfidf_vec) "
            "VALUES (?,?,?,?,?)",
            rows,
        )


def load_all_chunks() -> List[dict]:
    """Load every chunk (text + tokens + tfidf_vec) for the retriever."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, doc_id, chunk_idx, text, tokens, tfidf_vec FROM chunks"
        ).fetchall()
    result = []
    for r in rows:
        result.append(
            {
                "id": r[0],
                "doc_id": r[1],
                "chunk_idx": r[2],
                "text": r[3],
                "tokens": json.loads(r[4]) if r[4] else [],
                "tfidf_vec": pickle.loads(r[5]) if r[5] else {},
            }
        )
    return result


def get_chunk_texts_by_ids(ids: List[int]) -> List[str]:
    placeholders = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
    id_to_text = {r[0]: r[1] for r in rows}
    return [id_to_text[i] for i in ids if i in id_to_text]
