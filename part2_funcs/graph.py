"""LangGraph agent graph wiring LLM tool-calling loop."""

import json

import streamlit as st
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_MODEL,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from models import AgentOutput, AgentState, ToolCallRecord
from tools import ALL_TOOLS, TOOLS_BY_NAME

SYSTEM_PROMPT = (
    "You are a helpful assistant for Starship Coffee Co. "
    "Use the provided tools to answer questions about orders and customers. "
    "Always use tools to look up data; never invent numbers or statuses. "
    "Mask all customer emails in your final answer."
)


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance bound with all tools, pointed at OpenRouter."""
    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=0,
        default_headers=OPENROUTER_HEADERS,
    )
    return llm.bind_tools(ALL_TOOLS)


def node_agent(state: AgentState) -> dict:
    """Call the LLM with current message history and return its response."""
    llm = get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def node_tools(state: AgentState) -> dict:
    """Execute all tool calls from the last AI message and record results."""
    last_ai = state["messages"][-1]
    existing = state.get("tool_call_records") or []
    new_records: list[ToolCallRecord] = []
    tool_messages: list[ToolMessage] = []

    for tc in last_ai.tool_calls:
        tool_fn = TOOLS_BY_NAME[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        new_records.append(
            ToolCallRecord(tool=tc["name"], args=tc["args"], result=result)
        )
        tool_messages.append(
            ToolMessage(content=json.dumps(result), tool_call_id=tc["id"])
        )

    return {
        "messages": tool_messages,
        "tool_call_records": existing + new_records,
    }


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    """Compile and return the function-calling agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", node_agent)
    graph.add_node("tools", node_tools)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


@st.cache_resource
def get_agent():
    """Return the cached compiled agent graph."""
    return build_graph()


def run_agent(question: str) -> AgentOutput:
    """Run the agent on a question and return structured output."""
    agent = get_agent()
    final_state = agent.invoke(
        {"messages": [{"role": "user", "content": question}], "tool_call_records": []}
    )
    last_message = final_state["messages"][-1]
    return AgentOutput(
        final_answer=last_message.content,
        tool_calls=final_state.get("tool_call_records") or [],
    )
