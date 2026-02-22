"""Vector storage backends: JSON (in-memory), SQLite, and Qdrant."""

import sqlite3

import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import CACHE_DIR, COLLECTION_NAME
from loader import embed_chunks, load_chunks
from models import ChunkDoc, RetrievedChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def search_json(query_vec: list[float], chunks: list[ChunkDoc], k: int) -> list[RetrievedChunk]:
    """Cosine search over in-memory chunks (no external storage needed)."""
    scored = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            content=c.content,
            score=cosine_similarity(query_vec, c.embedding),
        )
        for c in chunks
    ]
    return sorted(scored, key=lambda x: x.score, reverse=True)[:k]


def search_sqlite(query_vec: list[float], chunks: list[ChunkDoc], k: int) -> list[RetrievedChunk]:
    """SQLite-backed vector search; embeddings stored as raw float32 BLOBs."""
    db_path = CACHE_DIR / "index.sqlite"
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS chunks "
        "(chunk_id TEXT PRIMARY KEY, doc_id TEXT, content TEXT, embedding BLOB)"
    )
    cur.execute("SELECT COUNT(*) FROM chunks")
    if cur.fetchone()[0] == 0:
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.content,
                np.array(c.embedding, dtype=np.float32).tobytes(),
            )
            for c in chunks
        ]
        cur.executemany("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)", rows)
        con.commit()

    cur.execute("SELECT chunk_id, doc_id, content, embedding FROM chunks")
    results = []
    for chunk_id, doc_id, content, emb_blob in cur.fetchall():
        vec = np.frombuffer(emb_blob, dtype=np.float32).tolist()
        score = cosine_similarity(query_vec, vec)
        results.append(RetrievedChunk(chunk_id=chunk_id, doc_id=doc_id, content=content, score=score))
    con.close()
    return sorted(results, key=lambda x: x.score, reverse=True)[:k]


@st.cache_resource(show_spinner="Initialising Qdrant…")
def get_qdrant_client(dim: int) -> QdrantClient:
    """Return a Qdrant in-memory client pre-populated with all doc embeddings."""
    client = QdrantClient(":memory:")
    chunks = embed_chunks(load_chunks())
    if not chunks:
        return client
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=c.embedding,
            payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content},
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def search_qdrant(query_vec: list[float], k: int) -> list[RetrievedChunk]:
    """Qdrant cosine search using the in-memory client."""
    client = get_qdrant_client(len(query_vec))
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=k,
    )
    return [
        RetrievedChunk(
            chunk_id=h.payload["chunk_id"],
            doc_id=h.payload["doc_id"],
            content=h.payload["content"],
            score=h.score,
        )
        for h in response.points
    ]
