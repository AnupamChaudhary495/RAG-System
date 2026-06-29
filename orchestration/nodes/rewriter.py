"""Query Rewriter node — rephrases the query to surface better documents.

Only called when confidence_score < 0.4 and retry_count < 2.  Uses
gpt-4o-mini to generate a single alternative phrasing that may retrieve
more relevant chunks on the next pass.
"""

from __future__ import annotations

import json
import os

import openai

from orchestration.state import RAGState

_SYSTEM_PROMPT = (
    "You are a search query optimizer for a document-retrieval system.\n"
    "Given an original query and a description of what information was missing, "
    "generate ONE alternative phrasing of the query that is more likely to "
    "retrieve the needed information.\n"
    "Respond with valid JSON only: {\"rewritten_query\": \"<new query>\"}"
)


def rewriter_node(state: RAGState) -> dict:
    """Rewrite the query to improve retrieval on the next attempt.

    Reads the generator's answer (which describes what was missing) and the
    original query, then produces a rephrased query.  Increments retry_count.

    Args:
        state: Current RAG state.

    Returns:
        Partial state update: {"rewritten_query": str, "retry_count": int}.
    """
    model = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    client = openai.OpenAI()

    retry_count: int = state.get("retry_count", 0)

    previous = state.get("rewritten_query")
    previous_note = (
        f"\nPrevious rewrite attempt: \"{previous}\"" if previous else ""
    )

    user_content = (
        f"Original query: \"{state['original_query']}\"{previous_note}\n"
        f"What was missing: \"{state['answer']}\"\n\n"
        "Generate a better search query."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0.7,
    )

    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    rewritten: str = parsed.get("rewritten_query", state["original_query"])

    new_retry_count = retry_count + 1
    print(f"[rewriter] retry {new_retry_count}: '{rewritten}'")
    return {
        "rewritten_query": rewritten,
        "retry_count": new_retry_count,
    }
