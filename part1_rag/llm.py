"""LLM and embedding model factories."""

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    EMBED_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_MODEL,
)


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at OpenRouter."""
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=0,
        default_headers=OPENROUTER_HEADERS,
    )


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance pointed at OpenRouter."""
    return OpenAIEmbeddings(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=EMBED_MODEL,
        default_headers=OPENROUTER_HEADERS,
    )
