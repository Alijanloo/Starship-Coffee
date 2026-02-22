"""Streamlit UI for the function-calling agent over customer/order data."""

import json
import sys

import pandas as pd
import streamlit as st
from graph import run_agent

PRESETS = [
    "Total spend for customer C-101 from 2025-09-01 to 2025-09-30.",
    "Refund 5.40 credits for order B77.",
    "What is the status and masked email for order `  c9  `?",
]

st.set_page_config(page_title="Starship Coffee — Function Calling", layout="wide")
st.title("Starship Coffee Co. — Function Calling Agent")

if "question" not in st.session_state:
    st.session_state["question"] = ""


def _set_preset(text: str) -> None:
    st.session_state["question"] = text


left, right = st.columns([1, 1])

with left:
    st.subheader("Question")
    for i, label in enumerate(PRESETS, 1):
        st.button(f"Preset {i}", key=f"preset_{i}", on_click=_set_preset, args=(label,))

    question = st.text_area(
        "Ask a question about orders or customers",
        height=120,
        key="question",
    )

    submit = st.button("Run", type="primary")

with right:
    st.subheader("Output")

    if submit and question.strip():
        with st.spinner("Running agent…"):
            output = run_agent(question.strip())

        result_dict = output.model_dump()
        print(json.dumps(result_dict, indent=2), file=sys.stdout, flush=True)

        st.markdown("**Final Answer**")
        st.write(output.final_answer)

        if output.tool_calls:
            st.markdown("**Tool Calls**")
            rows = [
                {
                    "tool": tc.tool,
                    "args": json.dumps(tc.args, ensure_ascii=False),
                    "result": json.dumps(tc.result, ensure_ascii=False),
                }
                for tc in output.tool_calls
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No tool calls were made.")

        st.markdown("**JSON Output**")
        st.json(result_dict)

    elif submit:
        st.warning("Please enter a question.")
