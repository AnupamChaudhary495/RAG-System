"""RAGState TypedDict — the shared state passed between all graph nodes."""

from __future__ import annotations

from typing import TypedDict


class RAGState(TypedDict):
    """Mutable state object threaded through every node in the RAG graph.

    Fields
    ------
    query:
        The active query string.  May be replaced by the rewriter.
    original_query:
        The user's original query — preserved from init, never overwritten.
    conversation_history:
        Ordered list of prior turns: [{role: str, content: str}, ...].
    router_decision:
        Set by the router node: "retrieve" or "conversational".
    retrieved_chunks:
        Populated by the retriever node with list[RetrievalResult].
    rewritten_query:
        Set by the rewriter node when a retry is triggered; None otherwise.
    answer:
        Final answer text produced by the generator node.
    source_chunk_ids:
        Chunk UUIDs cited by the generator in its answer.
    confidence_score:
        Self-reported confidence (0.0–1.0) from the generator node.
    retry_count:
        Number of rewrite + re-retrieve cycles so far; capped at 2.
    """

    query: str
    original_query: str
    conversation_history: list[dict]
    router_decision: str
    retrieved_chunks: list
    rewritten_query: str | None
    answer: str
    source_chunk_ids: list[str]
    confidence_score: float
    retry_count: int
