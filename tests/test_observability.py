"""Unit tests for Phase 6 — Langfuse observability integration.

Targets the langfuse 4.x SDK. The Langfuse client, the CallbackHandler, and
os.environ are always mocked so no live Langfuse credentials or network access
are required.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from observability.langfuse_handler import (
    make_langfuse_config,
    make_langfuse_handler,
)


class TestMakeLangfuseHandler:
    def test_returns_none_when_keys_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            assert make_langfuse_handler("s", "q") is None

    def test_returns_none_when_public_key_empty(self):
        env = {"LANGFUSE_SECRET_KEY": "sk"}
        with patch.dict(os.environ, env, clear=True):
            assert make_langfuse_handler("s", "q") is None

    def test_returns_none_when_secret_key_empty(self):
        env = {"LANGFUSE_PUBLIC_KEY": "pk"}
        with patch.dict(os.environ, env, clear=True):
            assert make_langfuse_handler("s", "q") is None

    def test_returns_handler_when_both_keys_present(self):
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk-123",
            "LANGFUSE_SECRET_KEY": "sk-456",
            "LANGFUSE_HOST": "http://selfhosted:3000",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "observability.langfuse_handler.Langfuse"
        ) as MockLF, patch(
            "observability.langfuse_handler.CallbackHandler"
        ) as MockCH:
            handler = make_langfuse_handler("sess-1", "my query")

        assert handler is MockCH.return_value
        # Auth is configured on the Langfuse client in langfuse 4.x.
        MockLF.assert_called_once()
        lf_kwargs = MockLF.call_args.kwargs
        assert lf_kwargs["public_key"] == "pk-123"
        assert lf_kwargs["secret_key"] == "sk-456"
        assert lf_kwargs["host"] == "http://selfhosted:3000"
        # The handler receives the public key to bind it to the right project.
        MockCH.assert_called_once()
        assert MockCH.call_args.kwargs["public_key"] == "pk-123"

    def test_uses_default_host(self):
        env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
        with patch.dict(os.environ, env, clear=True), patch(
            "observability.langfuse_handler.Langfuse"
        ) as MockLF, patch("observability.langfuse_handler.CallbackHandler"):
            make_langfuse_handler("s", "q")

        assert MockLF.call_args.kwargs["host"] == "https://cloud.langfuse.com"

    def test_uses_custom_host(self):
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
            "LANGFUSE_HOST": "http://selfhosted:3000",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "observability.langfuse_handler.Langfuse"
        ) as MockLF, patch("observability.langfuse_handler.CallbackHandler"):
            make_langfuse_handler("s", "q")

        assert MockLF.call_args.kwargs["host"] == "http://selfhosted:3000"


class TestMakeLangfuseConfig:
    def test_returns_empty_dict_when_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            assert make_langfuse_config("s", "q") == {}

    def test_returns_callbacks_dict_when_keys_present(self):
        env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
        with patch.dict(os.environ, env, clear=True), patch(
            "observability.langfuse_handler.Langfuse"
        ), patch("observability.langfuse_handler.CallbackHandler"):
            config = make_langfuse_config("s", "q")

        assert "callbacks" in config
        assert isinstance(config["callbacks"], list)
        assert len(config["callbacks"]) == 1

    def test_two_calls_produce_independent_handlers(self):
        env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
        with patch.dict(os.environ, env, clear=True), patch(
            "observability.langfuse_handler.Langfuse"
        ), patch("observability.langfuse_handler.CallbackHandler") as MockCH:
            config_a = make_langfuse_config("session-A", "q1")
            config_b = make_langfuse_config("session-B", "q2")

        # A fresh handler is constructed per call (no module-level singleton).
        assert MockCH.call_count == 2
        # Session scoping (attached via config metadata in langfuse 4.x) differs.
        session_a = config_a["metadata"]["langfuse_session_id"]
        session_b = config_b["metadata"]["langfuse_session_id"]
        assert session_a == "session-A"
        assert session_b == "session-B"
        assert session_a != session_b
