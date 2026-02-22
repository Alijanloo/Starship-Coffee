"""Vision layer: calls the OpenRouter VLM on a receipt image and
returns structured data."""

import base64
import json

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_VLM,
    SYSTEM_PROMPT,
)
from models import ReceiptItem, ReceiptResult
from openai import OpenAI


def _encode_image(image_bytes: bytes, media_type: str) -> str:
    """Return a base64 data-URI string for an image."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def parse_receipt(image_bytes: bytes, media_type: str) -> ReceiptResult:
    """Call the vision model and parse the response into a ReceiptResult."""
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=OPENROUTER_HEADERS,
    )

    data_uri = _encode_image(image_bytes, media_type)

    response = client.chat.completions.create(
        model=OPENROUTER_VLM,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": "Parse this receipt and return the JSON as instructed.",
                    },
                ],
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content or ""
    raw = raw.strip()

    # Strip markdown code fences if the model wraps them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)

    items = [ReceiptItem(**item) for item in data.get("items", [])]
    return ReceiptResult(items=items, total=str(data.get("total", "0.00")))
