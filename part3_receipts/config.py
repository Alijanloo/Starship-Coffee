"""Application-wide constants and environment config for the receipt-OCR app."""

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

RECEIPTS_DIR = Path(__file__).parent / "receipts"
PREDICTIONS_FILE = Path(__file__).parent / "predictions.jsonl"

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_VLM: str = os.getenv("OPENROUTER_VLM", "qwen/qwen3-vl-235b-a22b-thinking")

OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "starship-coffee",
    "X-Title": "Starship Coffee Receipts",
}

SYSTEM_PROMPT: str = (
    "You are a receipt-parsing assistant. "
    "Extract every line item and the final total from the provided receipt image. "
    "If a crossed-out total and a current total both appear, "
    "return only the current total. "
    "Respond with valid JSON only, no markdown fences, matching exactly this shape:\n"
    '{"items": [{"name": "...", "qty": 1, "unit_price": "0.00",'
    ' "line_total": "0.00"}], "total": "0.00"}'
)
