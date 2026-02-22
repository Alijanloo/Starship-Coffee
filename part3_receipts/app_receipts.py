"""Streamlit UI for the receipt OCR app."""

import json
import sys

import pandas as pd
import streamlit as st
from config import PREDICTIONS_FILE
from storage import save_prediction
from vision import parse_receipt

SUPPORTED_TYPES = ["png", "jpg", "jpeg"]

st.set_page_config(page_title="Starship Coffee — Receipt OCR", layout="centered")
st.title("Starship Coffee Co. — Receipt OCR")

uploaded_file = st.file_uploader(
    "Upload a receipt image (PNG or JPG)",
    type=SUPPORTED_TYPES,
)

if uploaded_file is not None:
    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    with st.spinner("Parsing receipt with vision model…"):
        media_type = (
            "image/png" if uploaded_file.name.lower().endswith(".png") else "image/jpeg"
        )
        result = parse_receipt(uploaded_file.read(), media_type)

    save_prediction(result, uploaded_file.name, PREDICTIONS_FILE)

    result_dict = result.model_dump()
    print(json.dumps(result_dict, indent=2), file=sys.stdout, flush=True)

    st.subheader("Extracted Data")
    st.json(result_dict)

    st.subheader("Line Items")
    rows = [
        {
            "name": item.name,
            "qty": item.qty,
            "unit_price": item.unit_price,
            "line_total": item.line_total,
        }
        for item in result.items
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.success(f"Total: **{result.total}**")
    st.caption(f"Prediction saved to `{PREDICTIONS_FILE.name}`")
