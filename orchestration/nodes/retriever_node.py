"""Retriever node — wraps the Phase 3 hybrid retrieval engine.

The Retriever instance is lazy-loaded on first call to avoid paying the
BGE-M3 model startup cost at import time and to make the module testable
without a live Qdrant instance.
"""

from __future__ import annotations

from retrieval.retriever import Retriever, RetrievalResult
from orchestration.state import RAGState

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Return (and cache) the singleton Retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retriever_node(state: RAGState) -> dict:
    """Retrieve relevant chunks for the active query.

    Uses state['rewritten_query'] when set (retry path), otherwise falls
    back to state['query'] (first-pass path).

    Args:
        state: Current RAG state.

    Returns:
        Partial state update: {"retrieved_chunks": list[RetrievalResult]}.
    """
    active_query = state.get("rewritten_query") or state["query"]

    retriever = _get_retriever()
    results: list[RetrievalResult] = retriever.retrieve(active_query, top_k=5)

    print(f"[retriever] {len(results)} chunks retrieved")
    return {"retrieved_chunks": results}
