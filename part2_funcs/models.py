"""Pydantic models and LangGraph state definition for the function-calling app."""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Customer(BaseModel):
    """A customer record loaded from customers.csv."""

    customer_id: str
    name: str
    email: str
    tier: str
    credits: int


class Order(BaseModel):
    """An order record loaded from orders.csv."""

    order_id: str
    customer_id: str
    status: str
    item: str
    qty: int
    unit_price: float
    total: float
    created_at: str


class ToolCallRecord(BaseModel):
    """A single recorded tool invocation with its result."""

    tool: str
    args: dict[str, Any]
    result: dict[str, Any]


class AgentOutput(BaseModel):
    """Final structured output from the function-calling agent."""

    final_answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class AgentState(TypedDict):
    """LangGraph state flowing through the agent graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_records: list[ToolCallRecord]
