"""LangChain tools exposing order and customer operations to the LLM."""

from datetime import datetime

from config import REFUNDABLE_STATUSES
from data import load_customers, load_orders, mask_email, normalize_order_id
from langchain_core.tools import tool


@tool
def get_order(order_id: str) -> dict:
    """Retrieve order status, total, and masked customer email by order ID."""
    orders = load_orders()
    customers = load_customers()
    oid = normalize_order_id(order_id)
    order = orders.get(oid)
    if order is None:
        return {"error": f"Order '{oid}' not found."}
    customer = customers.get(order.customer_id)
    email = mask_email(customer.email) if customer else "unknown"
    return {"status": order.status, "total": order.total, "masked_email": email}


@tool
def refund_order(order_id: str, amount: float) -> dict:
    """Request a refund for an order. Allowed only when status is
    settled or prepping and amount does not exceed total."""
    orders = load_orders()
    oid = normalize_order_id(order_id)
    order = orders.get(oid)
    if order is None:
        return {"ok": False, "reason": f"Order '{oid}' not found."}
    if order.status not in REFUNDABLE_STATUSES:
        return {
            "ok": False,
            "reason": f"Order status '{order.status}' is not eligible for a refund.",
        }
    if amount > order.total:
        return {
            "ok": False,
            "reason": f"Refund amount {amount} exceeds order total {order.total}.",
        }
    return {"ok": True}


@tool
def spend_in_period(customer_id: str, start: str, end: str) -> dict:
    """Calculate total spend for a customer between start and end dates (YYYY-MM-DD)."""
    orders = load_orders()
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    total = sum(
        o.total
        for o in orders.values()
        if o.customer_id == customer_id
        and start_dt <= datetime.fromisoformat(o.created_at) <= end_dt
    )
    return {"total_spend": round(total, 2)}


ALL_TOOLS = [get_order, refund_order, spend_in_period]
TOOLS_BY_NAME: dict = {t.name: t for t in ALL_TOOLS}
