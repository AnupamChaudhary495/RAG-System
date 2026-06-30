"""Router node — classifies each query as 'retrieve' or 'conversational'."""

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


async def router_node(state: RAGState) -> dict:
    model = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    client = openai.AsyncOpenAI()

    history = state["conversation_history"][-3:] if state["conversation_history"] else []
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": state["query"]})

    response = await client.chat.completions.create(
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

    if decision not in _VALID_DECISIONS:
        decision = "retrieve"

    print(f"[router] decision: {decision}")
    return {"router_decision": decision}
