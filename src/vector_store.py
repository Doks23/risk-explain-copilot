from __future__ import annotations

import json
import math
import re
import sqlite3
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .data_generator import DATA_DIR
from .knowledge_base import KNOWLEDGE_CHUNKS, KnowledgeChunk


VECTOR_DB_PATH = DATA_DIR / "knowledge_vectors.db"
VECTOR_DIMENSIONS = 256

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def initialize_vector_store(db_path: Path = VECTOR_DB_PATH, force: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                tags TEXT NOT NULL,
                vector TEXT NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM knowledge_chunks")
        for chunk in KNOWLEDGE_CHUNKS:
            conn.execute(
                """
                INSERT INTO knowledge_chunks (chunk_id, title, text, tags, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.title,
                    chunk.text,
                    json.dumps(chunk.tags),
                    json.dumps(_embed(_chunk_text(chunk))),
                ),
            )
        conn.commit()


def retrieve_context(query: str, top_k: int = 5, db_path: Path = VECTOR_DB_PATH) -> tuple[dict[str, Any], ...]:
    initialize_vector_store(db_path)
    query_vector = _embed(query)
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for chunk_id, title, text, tags, vector_json in conn.execute(
            "SELECT chunk_id, title, text, tags, vector FROM knowledge_chunks"
        ):
            score = _cosine_similarity(query_vector, json.loads(vector_json))
            if score <= 0:
                continue
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text,
                    "tags": json.loads(tags),
                    "score": round(score, 4),
                }
            )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return tuple(rows[:top_k])


def _chunk_text(chunk: KnowledgeChunk) -> str:
    return f"{chunk.title} {' '.join(chunk.tags)} {chunk.text}"


def _embed(text: str) -> list[float]:
    tokens = _tokens(text)
    counts = Counter(tokens)
    vector = [0.0] * VECTOR_DIMENSIONS
    for token, count in counts.items():
        vector[_stable_bucket(token)] += float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def _tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9_&]+", text.lower())
    expanded: list[str] = []
    for token in raw_tokens:
        if token in STOP_WORDS:
            continue
        expanded.append(_normalize_token(token))
    return [token for token in expanded if token and token not in STOP_WORDS]


def _normalize_token(token: str) -> str:
    aliases = {
        "p&l": "pnl",
        "profit": "pnl",
        "loss": "pnl",
        "value": "var",
        "risk": "risk",
        "desks": "desk",
        "books": "book",
        "products": "product",
        "currencies": "currency",
        "covering": "coverage",
        "covered": "coverage",
        "cover": "coverage",
        "drivers": "driver",
        "moves": "move",
    }
    return aliases.get(token, token)


def _stable_bucket(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % VECTOR_DIMENSIONS


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
