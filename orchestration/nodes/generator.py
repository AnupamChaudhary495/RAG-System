"""Generator node — synthesises an answer from retrieved chunks via an LLM.

Uses gpt-4o (configurable via GENERATOR_MODEL env var) with JSON mode.
Injects conversation history, numbered chunk context, and the original query
into the prompt.  Self-reports a confidence score that drives the retry loop.
"""

from __future__ import annotations

import json
import os

import openai

from retrieval.retriever import RetrievalResult
from orchestration.state import RAGState

_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer ONLY using the provided context.\n"
    "If the context does not contain sufficient information to answer, "
    "set confidence below 0.4 and explain what is missing.\n"
    "Do not invent facts."
)

_RESPONSE_SCHEMA = """Respond with valid JSON matching this exact schema:
{
  "answer":           "<your response text>",
  "source_chunk_ids": ["<chunk_id>", ...],
  "confidence":       <float between 0.0 and 1.0>
}"""


def _format_chunks(chunks: list[RetrievalResult]) -> str:
    """Format retrieved chunks as a numbered context block."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.page_number if chunk.page_number is not None else "?"
        lines.append(f"[{i}] source: {chunk.source_filename} p.{page}")
        lines.append(f"    {chunk.text}")
    return "\n".join(lines)


def generator_node(state: RAGState) -> dict:
    """Generate an answer grounded in retrieved chunks.

    Args:
        state: Current RAG state (expects retrieved_chunks to be populated).

    Returns:
        Partial state update with answer, source_chunk_ids, confidence_score.
    """
    model = os.getenv("GENERATOR_MODEL", "gpt-4o")
    client = openai.OpenAI()

    chunks: list[RetrievalResult] = state.get("retrieved_chunks", [])
    history = state["conversation_history"][-5:] if state["conversation_history"] else []

    context_block = _format_chunks(chunks) if chunks else "(no context retrieved)"

    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {state['query']}\n\n"
        f"{_RESPONSE_SCHEMA}"
    )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0.2,
    )

    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)

    answer: str = parsed.get("answer", "")
    source_ids: list[str] = parsed.get("source_chunk_ids", [])
    confidence: float = float(parsed.get("confidence", 0.0))

    print(f"[generator] confidence: {confidence:.2f}")
    return {
        "answer": answer,
        "source_chunk_ids": source_ids,
        "confidence_score": confidence,
    }
