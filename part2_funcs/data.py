"""CSV data loading and helper utilities (email masking, normalization)."""

import pandas as pd
import streamlit as st
from config import CUSTOMERS_CSV, ORDERS_CSV
from models import Customer, Order


def mask_email(email: str) -> str:
    """Return an email with the local part masked: first char + *** + @domain."""
    local, domain = email.split("@", 1)
    return f"{local[0]}***@{domain}"


def normalize_order_id(raw: str) -> str:
    """Strip whitespace and uppercase an order ID."""
    return raw.strip().upper()


@st.cache_data
def load_customers() -> dict[str, Customer]:
    """Load customers.csv and return a dict keyed by customer_id."""
    df = pd.read_csv(CUSTOMERS_CSV)
    return {row["customer_id"]: Customer(**row.to_dict()) for _, row in df.iterrows()}


@st.cache_data
def load_orders() -> dict[str, Order]:
    """Load orders.csv, normalize IDs, and return a dict keyed by order_id."""
    df = pd.read_csv(ORDERS_CSV)
    df["order_id"] = df["order_id"].apply(normalize_order_id)
    return {row["order_id"]: Order(**row.to_dict()) for _, row in df.iterrows()}
