"""LangGraph pipeline: injection detection → retrieval → generation."""

import re

import streamlit as st
from config import INJECTION_PATTERNS
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from llm import get_embedder, get_llm
from loader import embed_chunks, load_chunks
from models import RAGState
from storage import search_json, search_qdrant, search_sqlite


def node_detect_injection(state: RAGState) -> RAGState:
    """Detect prompt-injection or secrets-access attempts."""
    q = state["query"].lower()
    flagged = any(re.search(p, q) for p in INJECTION_PATTERNS)
    if not flagged:
        flagged = "secrets" in q or "secret/" in q
    return {**state, "is_injection": flagged}


def node_retrieve(state: RAGState) -> RAGState:
    """Embed query and retrieve top-k chunks from the chosen backend."""
    if state["is_injection"]:
        return {**state, "retrieved": []}

    embedder = get_embedder()
    query_vec = embedder.embed_query(state["query"])
    chunks = embed_chunks(load_chunks())
    backend = state["backend"]
    k = state["k"]

    if backend == "json":
        results = search_json(query_vec, chunks, k)
    elif backend == "sqlite":
        results = search_sqlite(query_vec, chunks, k)
    else:
        results = search_qdrant(query_vec, k)

    return {**state, "retrieved": [r.model_dump() for r in results]}


def node_generate(state: RAGState) -> RAGState:
    """Call the LLM to produce a ≤100-word answer with citations."""
    if state["is_injection"]:
        return {
            **state,
            "answer": (
                "I can't help with requests to reveal file contents "
                "or access restricted data. "
                "Try asking a specific question about our menu, "
                "policies, or operations."
            ),
            "citations": [],
        }

    retrieved = state["retrieved"]
    if not retrieved:
        return {**state, "answer": "No relevant documents found.", "citations": []}

    context = "\n\n---\n\n".join(f"[{r['doc_id']}]\n{r['content']}" for r in retrieved)
    citations = sorted({r["doc_id"] for r in retrieved})

    system_prompt = (
        "You are a helpful assistant for Starship Coffee Co. "
        "Answer the user's question using ONLY the provided context. "
        "Keep your answer to 100 words or fewer. "
        "Do not invent facts not present in the context."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {state['query']}"

    llm = get_llm()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {**state, "answer": response.content.strip(), "citations": citations}


def build_graph():
    """Compile and return the RAG LangGraph pipeline."""
    graph = StateGraph(RAGState)
    graph.add_node("detect_injection", node_detect_injection)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)

    graph.add_edge(START, "detect_injection")
    graph.add_edge("detect_injection", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


@st.cache_resource
def get_pipeline():
    """Return the cached compiled LangGraph pipeline."""
    return build_graph()
