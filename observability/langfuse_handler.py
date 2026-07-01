"""Langfuse observability integration for LangGraph RAG pipeline.

Provides a per-request CallbackHandler that traces all LangChain/LangGraph
calls — token counts, latency, cost — through the Langfuse platform.

Targets the langfuse 4.x SDK, which is compatible with the langchain 1.x stack
used by this project:
  - The handler lives at ``langfuse.langchain.CallbackHandler`` (the legacy
    ``langfuse.callback`` module was removed in langfuse 3.x).
  - Authentication (public/secret key, host) is configured on the ``Langfuse``
    client rather than passed to the handler constructor.
  - Session, user, and free-form metadata are attached per-invocation through
    the LangChain run config's ``metadata`` (reserved keys
    ``langfuse_session_id`` / ``langfuse_user_id``), not the handler.

Usage in the FastAPI endpoint:
    from observability.langfuse_handler import make_langfuse_config
    config = make_langfuse_config(session_id=request.session_id, query=request.query)
    async for event in graph.astream_events(state, version="v2", config=config):
        ...
"""

from __future__ import annotations

import logging
import os

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

DEFAULT_HOST = "https://cloud.langfuse.com"


def make_langfuse_handler(session_id: str, query: str) -> CallbackHandler | None:
    """Build a per-request Langfuse CallbackHandler, or None if unconfigured.

    Reads ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_HOST``
    from the environment. If either key is missing or empty, logs a warning and
    returns ``None`` so the pipeline runs normally without observability.

    When both keys are present, the ``Langfuse`` client is (re)configured with
    the credentials and a fresh ``CallbackHandler`` is returned. A new handler
    is created on every call — no module-level singleton — so each chat turn
    produces an independent, correctly-scoped trace. Session/user scoping and
    query metadata are attached by :func:`make_langfuse_config` via the run
    config, per the langfuse 4.x integration model.

    Args:
        session_id: Chat session identifier (attached downstream as the
            Langfuse session and user id).
        query: The user query (attached downstream as trace metadata).

    Returns:
        A configured ``CallbackHandler`` when both keys are present, else
        ``None``.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", DEFAULT_HOST)

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse keys not configured (LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY); tracing disabled."
        )
        return None

    # Configure the Langfuse client (auth) for this request. In langfuse 4.x
    # the CallbackHandler reads credentials from the configured client rather
    # than accepting them as constructor arguments.
    Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    return CallbackHandler(public_key=public_key)


def make_langfuse_config(session_id: str, query: str) -> dict:
    """Build a LangGraph ``config`` dict wiring in the Langfuse handler.

    The returned dict carries both the callback handler and the trace metadata.
    In langfuse 4.x, session/user scoping is supplied through the run config's
    ``metadata`` using the reserved keys ``langfuse_session_id`` and
    ``langfuse_user_id``; the raw query is included for trace inspection.

    Args:
        session_id: Chat session identifier. Used to group all turns of one
            conversation into a single Langfuse session, and reused as the
            Langfuse user id.
        query: The user query, attached to the trace metadata.

    Returns:
        ``{"callbacks": [handler], "metadata": {...}}`` when Langfuse is
        configured, else an empty dict ``{}``. The result is passed directly as
        ``config=`` to ``graph.astream_events()``.
    """
    handler = make_langfuse_handler(session_id=session_id, query=query)
    if handler is None:
        return {}
    return {
        "callbacks": [handler],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": session_id,
            "query": query,
        },
    }
