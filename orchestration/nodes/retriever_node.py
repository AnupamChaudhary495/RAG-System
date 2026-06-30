"""Retriever node — wraps the Phase 3 hybrid retrieval engine."""

from __future__ import annotations

import asyncio

from retrieval.retriever import Retriever, RetrievalResult
from orchestration.state import RAGState

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


async def retriever_node(state: RAGState) -> dict:
    active_query = state.get("rewritten_query") or state["query"]

    retriever = _get_retriever()
    # Run synchronous retriever in thread pool to avoid blocking the event loop
    results: list[RetrievalResult] = await asyncio.to_thread(
        retriever.retrieve, active_query, 5
    )

    print(f"[retriever] {len(results)} chunks retrieved")
    return {"retrieved_chunks": results}
