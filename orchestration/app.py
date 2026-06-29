"""Application entry point for the LangGraph RAG orchestration pipeline.

Exposes build_app() and run_query() for programmatic use, plus a CLI.

Usage (Python)::

    from orchestration.app import run_query

    result = run_query(
        "What chunking strategy was used in the ingestion pipeline?",
        conversation_history=[]
    )
    print(result["answer"])
    print(result["source_chunk_ids"])
    print(f"confidence: {result['confidence_score']:.2f}")
    print(f"retries: {result['retry_count']}")

Usage (CLI)::

    uv run python -m orchestration.app "What is reciprocal rank fusion?"
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of this file's package directory)
load_dotenv(Path(__file__).parent.parent / ".env")

from langgraph.pregel import Pregel as CompiledGraph

from orchestration.graph import build_graph
from orchestration.state import RAGState

_app: CompiledGraph | None = None


def build_app() -> CompiledGraph:
    """Compile and return the LangGraph application.

    The compiled graph is cached after the first call.

    Returns:
        Compiled LangGraph state machine.
    """
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """Run a single query through the full RAG pipeline.

    Constructs the initial RAGState, invokes the compiled graph, and
    returns a structured result dict.

    Args:
        query:                The user's question.
        conversation_history: Prior conversation turns as
                              [{"role": str, "content": str}, ...].

    Returns:
        Dict with keys: answer, source_chunk_ids, confidence_score,
        retry_count, router_decision.
    """
    app = build_app()

    initial_state: RAGState = {
        "query": query,
        "original_query": query,
        "conversation_history": conversation_history or [],
        "router_decision": "",
        "retrieved_chunks": [],
        "rewritten_query": None,
        "answer": "",
        "source_chunk_ids": [],
        "confidence_score": 0.0,
        "retry_count": 0,
    }

    final_state: RAGState = app.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "source_chunk_ids": final_state["source_chunk_ids"],
        "confidence_score": final_state["confidence_score"],
        "retry_count": final_state["retry_count"],
        "router_decision": final_state["router_decision"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Run a single query from the command line.

    Usage:
        python -m orchestration.app "<query>"
    """
    if len(sys.argv) != 2:
        print('Usage: python -m orchestration.app "<query>"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"\nQuery: {query}\n")

    result = run_query(query)

    print(f"\n{'─' * 60}")
    print(f"Answer:      {result['answer']}")
    print(f"Confidence:  {result['confidence_score']:.2f}")
    print(f"Retries:     {result['retry_count']}")
    print(f"Router:      {result['router_decision']}")
    print(f"Sources:     {result['source_chunk_ids']}")


if __name__ == "__main__":
    _cli()
