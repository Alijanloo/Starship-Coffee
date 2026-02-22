"""Pydantic models and LangGraph state definition."""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ChunkDoc(BaseModel):
    """A single text chunk from a markdown document."""

    chunk_id: str
    doc_id: str
    content: str
    embedding: list[float] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """A chunk returned from a similarity search."""

    chunk_id: str
    doc_id: str
    content: str
    score: float


class RAGOutput(BaseModel):
    """Final structured output for the RAG pipeline."""

    answer: str
    citations: list[str]


class RAGState(TypedDict):
    """LangGraph state flowing through the RAG pipeline."""

    query: str
    backend: str
    k: int
    is_injection: bool
    retrieved: list[dict]
    answer: str
    citations: list[str]
