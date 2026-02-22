"""Document loading, chunking, and embedding caching."""

import json

import streamlit as st
from langchain_text_splitters import MarkdownTextSplitter

from config import CACHE_DIR, DOCS_DIR, EMBED_MODEL
from llm import get_embedder
from models import ChunkDoc


@st.cache_data(show_spinner="Loading docs…")
def load_chunks() -> list[ChunkDoc]:
    """Load and chunk all markdown docs from DOCS_DIR."""
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks: list[ChunkDoc] = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        parts = splitter.split_text(text)
        for idx, part in enumerate(parts):
            chunks.append(
                ChunkDoc(
                    chunk_id=f"{md_file.name}::{idx}",
                    doc_id=md_file.name,
                    content=part,
                )
            )
    return chunks


def embed_chunks(chunks: list[ChunkDoc]) -> list[ChunkDoc]:
    """Attach embeddings to chunks, using a per-model JSON cache for speed."""
    safe_model = EMBED_MODEL.replace("/", "_").replace("-", "_")
    cache_file = CACHE_DIR / f"embeddings_{safe_model}.json"
    embedder = get_embedder()

    cached: dict[str, list[float]] = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    missing = [c for c in chunks if c.chunk_id not in cached]
    if missing:
        vecs = embedder.embed_documents([c.content for c in missing])
        for chunk, vec in zip(missing, vecs):
            cached[chunk.chunk_id] = vec
        cache_file.write_text(json.dumps(cached))

    for chunk in chunks:
        chunk.embedding = cached[chunk.chunk_id]
    return chunks
