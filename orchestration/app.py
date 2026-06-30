"""Application entry point for the LangGraph RAG orchestration pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langgraph.pregel import Pregel as CompiledGraph

from orchestration.graph import build_graph
from orchestration.state import RAGState

_app: CompiledGraph | None = None


def build_app() -> CompiledGraph:
    global _app
    if _app is None:
        _app = build_graph()
    return _app


async def run_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> dict:
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

    final_state: RAGState = await app.ainvoke(initial_state)

    return {
        "answer": final_state["answer"],
        "source_chunk_ids": final_state["source_chunk_ids"],
        "confidence_score": final_state["confidence_score"],
        "retry_count": final_state["retry_count"],
        "router_decision": final_state["router_decision"],
    }


def _cli() -> None:
    if len(sys.argv) != 2:
        print('Usage: python -m orchestration.app "<query>"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"\nQuery: {query}\n")

    result = asyncio.run(run_query(query))

    print(f"\n{'─' * 60}")
    print(f"Answer:      {result['answer']}")
    print(f"Confidence:  {result['confidence_score']:.2f}")
    print(f"Retries:     {result['retry_count']}")
    print(f"Router:      {result['router_decision']}")
    print(f"Sources:     {result['source_chunk_ids']}")


if __name__ == "__main__":
    _cli()
