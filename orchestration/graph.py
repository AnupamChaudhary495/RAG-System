"""LangGraph state machine definition and conditional routing functions.

Topology
--------
  START → router → [retriever | generator]  (conditional)
  retriever → generator
  generator → [rewriter | END]              (conditional)
  rewriter  → retriever
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from orchestration.nodes.generator import generator_node
from orchestration.nodes.retriever_node import retriever_node
from orchestration.nodes.router import router_node
from orchestration.nodes.rewriter import rewriter_node
from orchestration.state import RAGState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_router(state: RAGState) -> str:
    """Route to 'retriever' or 'generator' based on the router's decision.

    Args:
        state: Current RAG state (router_decision already set).

    Returns:
        Name of the next node: "retriever" or "generator".
    """
    if state["router_decision"] == "retrieve":
        return "retriever"
    return "generator"


def route_after_generator(state: RAGState) -> str:
    """Continue to END or trigger the rewrite → re-retrieve fallback loop.

    The loop fires when the generator is under-confident AND the retry cap
    has not been reached.  At most 2 retries are allowed.

    Args:
        state: Current RAG state (confidence_score and retry_count set).

    Returns:
        "rewriter" to retry, or END to finish.
    """
    if state["confidence_score"] < 0.4 and state["retry_count"] < 2:
        return "rewriter"
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph.

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    builder = StateGraph(RAGState)

    # Register nodes
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("generator", generator_node)
    builder.add_node("rewriter", rewriter_node)

    # Edges
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"retriever": "retriever", "generator": "generator"},
    )
    builder.add_edge("retriever", "generator")
    builder.add_conditional_edges(
        "generator",
        route_after_generator,
        {"rewriter": "rewriter", END: END},
    )
    builder.add_edge("rewriter", "retriever")

    return builder.compile()
