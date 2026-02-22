"""Application-wide constants and environment config for the function-calling app."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

for _proxy_key in (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
):
    os.environ.pop(_proxy_key, None)

DATA_DIR = Path(__file__).parent / "data"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
ORDERS_CSV = DATA_DIR / "orders.csv"

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "starship-coffee",
    "X-Title": "Starship Coffee Functions",
}

REFUNDABLE_STATUSES: frozenset[str] = frozenset({"settled", "prepping"})
