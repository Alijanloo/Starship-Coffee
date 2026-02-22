"""Pydantic models for receipt items and the structured OCR result."""

from pydantic import BaseModel, Field


class ReceiptItem(BaseModel):
    """A single line item parsed from a receipt."""

    name: str
    qty: int = 1
    unit_price: str = "0.00"
    line_total: str = "0.00"


class ReceiptResult(BaseModel):
    """Full structured output from the receipt-OCR vision call."""

    items: list[ReceiptItem] = Field(default_factory=list)
    total: str = "0.00"
