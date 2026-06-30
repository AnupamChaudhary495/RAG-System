"""Redis-backed session memory for conversation history.

Key schema : rag:session:{session_id}:history
Value       : JSON-encoded list[{"role": str, "content": str}]
TTL         : 86400 seconds (24 h)
Max length  : last 20 messages (trimmed after each append)
"""

from __future__ import annotations

import json

import redis.asyncio as aioredis

_redis: aioredis.Redis | None = None


def set_redis_client(client: aioredis.Redis) -> None:
    global _redis
    _redis = client


def _key(session_id: str) -> str:
    return f"rag:session:{session_id}:history"


async def get_history(session_id: str) -> list[dict]:
    raw = await _redis.get(_key(session_id))
    if raw is None:
        return []
    return json.loads(raw)


async def append_turns(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    history = await get_history(session_id)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    history = history[-20:]
    await _redis.set(_key(session_id), json.dumps(history), ex=86400)


async def clear_session(session_id: str) -> None:
    await _redis.delete(_key(session_id))
