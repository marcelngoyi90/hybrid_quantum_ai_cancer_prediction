from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DATABASE = Path(__file__).resolve().parent / "database" / "knowledge.db"


@lru_cache(maxsize=4)
def _index(database_name: str, modified_ns: int):
    database_path = Path(database_name)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.heading,
                chunks.text,
                documents.id AS document_id,
                documents.title,
                documents.organization,
                documents.category,
                documents.evidence_level,
                documents.source_path,
                documents.source_url,
                documents.external_urls_json
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY chunks.id
            """
        ).fetchall()

    if not rows:
        raise ValueError("The RAG database contains no chunks")

    texts = [str(row["text"]) for row in rows]
    word = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    character = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    word_matrix = word.fit_transform(texts)
    character_matrix = character.fit_transform(texts)
    matrix = hstack((word_matrix, character_matrix), format="csr")
    return rows, word, character, matrix


def retrieve(
    query: str,
    *,
    database_path: Path = DEFAULT_DATABASE,
    top_k: int = 6,
    minimum_score: float = 0.02,
) -> list[dict]:
    query = str(query).strip()
    if not query:
        raise ValueError("Question cannot be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be from 1 to 20")
    if not database_path.is_file():
        raise FileNotFoundError(
            f"RAG database not found: {database_path}. "
            "Run: python -m rag_system.build_database"
        )

    rows, word, character, matrix = _index(
        str(database_path.resolve()), database_path.stat().st_mtime_ns
    )
    query_matrix = hstack(
        (word.transform([query]), character.transform([query])), format="csr"
    )
    scores = cosine_similarity(query_matrix, matrix)[0]
    order = np.argsort(scores)[::-1]
    results: list[dict] = []

    for index in order:
        score = float(scores[index])
        if score < minimum_score:
            continue
        row = rows[int(index)]
        results.append(
            {
                "rank": len(results) + 1,
                "score": round(score, 6),
                "chunk_id": int(row["chunk_id"]),
                "document_id": row["document_id"],
                "title": row["title"],
                "organization": row["organization"],
                "category": row["category"],
                "evidence_level": row["evidence_level"],
                "source_path": row["source_path"],
                "source_url": row["source_url"],
                "external_urls": json.loads(row["external_urls_json"]),
                "heading": row["heading"],
                "text": row["text"],
            }
        )
        if len(results) >= top_k:
            break

    return results


def database_stats(database_path: Path = DEFAULT_DATABASE) -> dict:
    if not database_path.is_file():
        return {"ready": False, "database": str(database_path)}
    with closing(sqlite3.connect(database_path)) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        categories = connection.execute(
            """
            SELECT category, COUNT(*)
            FROM documents
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
    return {
        "ready": True,
        "database": str(database_path),
        "documents": int(metadata.get("document_count", 0)),
        "chunks": int(metadata.get("chunk_count", 0)),
        "categories": {name: count for name, count in categories},
    }
