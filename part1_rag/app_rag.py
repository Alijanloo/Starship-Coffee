"""Task 1 — Streamlit UI for the RAG pipeline."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from graph import get_pipeline  # noqa: E402
from models import RAGOutput, RAGState  # noqa: E402


def main():
    st.set_page_config(page_title="Starship Coffee RAG", page_icon="☕", layout="wide")
    st.title("☕ Starship Coffee — Knowledge Base Q&A")

    with st.sidebar:
        st.header("Settings")
        backend = st.selectbox("Storage backend", ["json", "sqlite", "qdrant"], index=0)
        k = st.number_input("Top-k passages", min_value=1, max_value=20, value=5)
        st.markdown("---")
        st.caption("Powered by LangGraph + OpenRouter")

    query = st.text_input(
        "Ask a question about the docs",
        placeholder="e.g. What is the refund policy?",
    )

    if st.button("Ask", type="primary") and query.strip():
        pipeline = get_pipeline()

        with st.spinner("Thinking…"):
            result: RAGState = pipeline.invoke(
                {
                    "query": query.strip(),
                    "backend": backend,
                    "k": k,
                    "is_injection": False,
                    "retrieved": [],
                    "answer": "",
                    "citations": [],
                }
            )

        output = RAGOutput(answer=result["answer"], citations=result["citations"])
        print(json.dumps(output.model_dump(), indent=2))

        st.subheader("Answer")
        st.write(output.answer)

        if output.citations:
            st.subheader("Citations")
            st.table(pd.DataFrame({"doc_id": output.citations}))

        if result["retrieved"]:
            with st.expander(f"Debug — top-{k} retrieved chunks"):
                for r in result["retrieved"]:
                    st.markdown(f"**{r['doc_id']}** (score: `{r['score']:.3f}`)")
                    st.caption(
                        r["content"][:200] + ("…" if len(r["content"]) > 200 else "")
                    )
                    st.divider()


if __name__ == "__main__":
    main()
