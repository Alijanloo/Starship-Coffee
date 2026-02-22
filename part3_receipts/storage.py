"""Persistence layer: appends each receipt prediction to a local JSONL file."""

import json
from pathlib import Path

from models import ReceiptResult


def save_prediction(result: ReceiptResult, filename: str, output_file: Path) -> None:
    """Append a single receipt prediction as one JSON line."""
    record = {"filename": filename, **result.model_dump()}
    with output_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
