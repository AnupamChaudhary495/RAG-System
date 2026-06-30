"""Tests for Phase 5 — FastAPI streaming backend and session management.

Test groups
-----------
  TestSession          — unit tests for api/session.py (mock Redis)
  TestHealthEndpoint   — GET /health
  TestChatStream       — POST /chat/stream SSE events
  TestSessionEndpoints — DELETE /session/{id}, GET /session/{id}/history
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis():
    """Async mock for the Redis client, injected into api.session._redis."""
    import api.session as session_module

    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=b"PONG")
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.aclose = AsyncMock()

    session_module._redis = mock
    yield mock
    session_module._redis = None


@pytest.fixture
def client(mock_redis):
    """TestClient with patched Redis so the lifespan doesn't need a real instance."""
    from api.main import app
    import redis.asyncio as aioredis

    with patch.object(aioredis, "from_url", return_value=mock_redis):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def mock_graph():
    """Patch build_app to return a graph that emits two token events then ends."""

    async def fake_stream(initial_state, version="v2"):
        token1 = MagicMock()
        token1.content = "Hello "
        yield {"event": "on_chat_model_stream", "name": "generator",
               "data": {"chunk": token1}}

        token2 = MagicMock()
        token2.content = "world!"
        yield {"event": "on_chat_model_stream", "name": "generator",
               "data": {"chunk": token2}}

        yield {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {
            "answer": "Hello world!",
            "source_chunk_ids": ["chunk-1"],
            "confidence_score": 0.9,
            "retry_count": 0,
            "router_decision": "retrieve",
        }}}

    graph = MagicMock()
    graph.astream_events = fake_stream

    with patch("api.main.build_app", return_value=graph):
        yield graph


# ---------------------------------------------------------------------------
# Session unit tests
# ---------------------------------------------------------------------------

class TestSession:
    @pytest.mark.asyncio
    async def test_get_history_returns_empty_list_when_key_missing(self, mock_redis):
        import api.session as session_module
        mock_redis.get.return_value = None
        result = await session_module.get_history("sess-missing")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_history_parses_stored_json(self, mock_redis):
        import api.session as session_module
        stored = [{"role": "user", "content": "Hi"}]
        mock_redis.get.return_value = json.dumps(stored).encode()
        result = await session_module.get_history("sess-1")
        assert result == stored

    @pytest.mark.asyncio
    async def test_append_turns_calls_set_with_ttl(self, mock_redis):
        import api.session as session_module
        mock_redis.get.return_value = None
        await session_module.append_turns("sess-1", "Hello", "Hi there!")
        mock_redis.set.assert_awaited_once()
        kwargs = mock_redis.set.call_args.kwargs
        assert kwargs.get("ex") == 86400

    @pytest.mark.asyncio
    async def test_append_turns_stores_both_roles(self, mock_redis):
        import api.session as session_module
        mock_redis.get.return_value = None
        await session_module.append_turns("sess-1", "user msg", "assistant msg")
        saved_json = mock_redis.set.call_args.args[1]
        saved = json.loads(saved_json)
        roles = [m["role"] for m in saved]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_history_trimmed_to_20_messages_after_append(self, mock_redis):
        import api.session as session_module
        # Pre-load 20 messages
        existing = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(20)
        ]
        mock_redis.get.return_value = json.dumps(existing).encode()
        await session_module.append_turns("sess-1", "extra user", "extra asst")
        saved_json = mock_redis.set.call_args.args[1]
        saved = json.loads(saved_json)
        assert len(saved) == 20

    @pytest.mark.asyncio
    async def test_clear_session_calls_delete(self, mock_redis):
        import api.session as session_module
        await session_module.clear_session("sess-del")
        mock_redis.delete.assert_awaited_once_with("rag:session:sess-del:history")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200_with_status_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_returns_redis_status_field(self, client):
        response = client.get("/health")
        assert "redis" in response.json()


# ---------------------------------------------------------------------------
# Chat stream endpoint
# ---------------------------------------------------------------------------

class TestChatStream:
    def _parse_events(self, text: str) -> list[dict]:
        events = []
        for line in text.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        return events

    def test_returns_text_event_stream_content_type(self, client, mock_graph, mock_redis):
        response = client.post("/chat/stream",
                               json={"query": "test", "session_id": "s1"})
        assert "text/event-stream" in response.headers["content-type"]

    def test_emits_at_least_one_token_event(self, client, mock_graph, mock_redis):
        response = client.post("/chat/stream",
                               json={"query": "Hello?", "session_id": "s1"})
        events = self._parse_events(response.text)
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) >= 1

    def test_token_content_is_non_empty_string(self, client, mock_graph, mock_redis):
        response = client.post("/chat/stream",
                               json={"query": "Hello?", "session_id": "s1"})
        events = self._parse_events(response.text)
        for e in events:
            if e["type"] == "token":
                assert isinstance(e["content"], str)
                assert len(e["content"]) > 0
                break

    def test_emits_metadata_event(self, client, mock_graph, mock_redis):
        response = client.post("/chat/stream",
                               json={"query": "Hello?", "session_id": "s1"})
        events = self._parse_events(response.text)
        meta = [e for e in events if e["type"] == "metadata"]
        assert len(meta) == 1
        assert "source_chunk_ids" in meta[0]
        assert "confidence_score" in meta[0]

    def test_done_event_is_last(self, client, mock_graph, mock_redis):
        response = client.post("/chat/stream",
                               json={"query": "Hello?", "session_id": "s1"})
        events = self._parse_events(response.text)
        assert events[-1]["type"] == "done"

    def test_missing_query_returns_422(self, client, mock_redis):
        response = client.post("/chat/stream", json={"session_id": "s1"})
        assert response.status_code == 422

    def test_missing_session_id_returns_422(self, client, mock_redis):
        response = client.post("/chat/stream", json={"query": "test"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------

class TestSessionEndpoints:
    def test_delete_session_returns_200(self, client, mock_redis):
        response = client.delete("/session/my-session")
        assert response.status_code == 200
        assert response.json()["deleted"] == "my-session"

    def test_delete_session_calls_redis_delete(self, client, mock_redis):
        client.delete("/session/my-session")
        mock_redis.delete.assert_awaited()

    def test_get_history_returns_empty_list_by_default(self, client, mock_redis):
        mock_redis.get.return_value = None
        response = client.get("/session/my-session/history")
        assert response.status_code == 200
        body = response.json()
        assert body["history"] == []
        assert body["session_id"] == "my-session"

    def test_get_history_returns_stored_messages(self, client, mock_redis):
        stored = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        mock_redis.get.return_value = json.dumps(stored).encode()
        response = client.get("/session/my-session/history")
        assert response.json()["history"] == stored
