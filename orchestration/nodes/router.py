"""Router node — classifies each query as 'retrieve' or 'conversational'.

Uses gpt-4o-mini (configurable via ROUTER_MODEL env var) with JSON mode to
guarantee a valid classification string.  Unexpected values default to
'retrieve' so the pipeline always attempts retrieval when uncertain.
"""

from __future__ import annotations

import json
import os

import openai

from orchestration.state import RAGState

_VALID_DECISIONS = {"retrieve", "conversational"}

_SYSTEM_PROMPT = """You are a query classifier for a document-retrieval system.

Classify the user query into EXACTLY ONE of these categories:
- "retrieve"       : The query requires searching the knowledge base for
                     specific information (facts, explanations, definitions).
- "conversational" : The query is a greeting, meta question, clarification,
                     or can be answered without document retrieval.

Respond with valid JSON only:
{"decision": "retrieve"}
or
{"decision": "conversational"}
"""


def router_node(state: RAGState) -> dict:
    """Classify the query and update state['router_decision'].

    Args:
        state: Current RAG state.

    Returns:
        Partial state update: {"router_decision": "retrieve" | "conversational"}.
    """
    model = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    client = openai.OpenAI()

    # Build context window: last 3 conversation turns + current query
    history = state["conversation_history"][-3:] if state["conversation_history"] else []
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": state["query"]})

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        decision = parsed.get("decision", "retrieve")
    except (json.JSONDecodeError, AttributeError):
        decision = "retrieve"

    # Guard against unexpected values
    if decision not in _VALID_DECISIONS:
        decision = "retrieve"

    print(f"[router] decision: {decision}")
    return {"router_decision": decision}
