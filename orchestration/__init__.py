"""LangGraph orchestration layer for the Enterprise RAG system.

Phase 4: routes queries through a state machine of router → retriever →
generator → (optional) rewriter → retriever loop, then returns a structured
JSON response with answer and source citations.
"""

__all__ = ["build_app", "run_query"]
