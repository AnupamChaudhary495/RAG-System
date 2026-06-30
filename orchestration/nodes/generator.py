"""Generator node — synthesises an answer from retrieved chunks via an LLM.

Uses ChatOpenAI with streaming=True so LangGraph's astream_events can emit
token-level events (on_chat_model_stream, name="generator") for SSE delivery.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from orchestration.state import RAGState
from retrieval.retriever import RetrievalResult

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
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.page_number if chunk.page_number is not None else "?"
        lines.append(f"[{i}] source: {chunk.source_filename} p.{page}")
        lines.append(f"    {chunk.text}")
    return "\n".join(lines)


async def generator_node(state: RAGState) -> dict:
    model = os.getenv("GENERATOR_MODEL", "gpt-4o")

    llm = ChatOpenAI(
        model=model,
        streaming=True,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}},
    ).with_config({"run_name": "generator"})

    chunks: list[RetrievalResult] = state.get("retrieved_chunks", [])
    history = state["conversation_history"][-5:] if state["conversation_history"] else []

    context_block = _format_chunks(chunks) if chunks else "(no context retrieved)"
    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {state['query']}\n\n"
        f"{_RESPONSE_SCHEMA}"
    )

    lc_messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    lc_messages.append(HumanMessage(content=user_content))

    response = await llm.ainvoke(lc_messages)
    raw = response.content or "{}"
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
