"""Unit and integration tests for Phase 4 — LangGraph orchestration layer.

Test groups
-----------
  TestRAGState         — TypedDict field validation (no mocking)
  TestRoutingFunctions — pure routing logic (no LLM calls)
  TestRouterNode       — router node with mocked AsyncOpenAI
  TestGeneratorNode    — generator node with mocked ChatOpenAI
  TestRewriterNode     — rewriter node with mocked AsyncOpenAI
  TestIntegration      — full graph execution with all external calls mocked
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.graph import END


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    state = {
        "query": "What is RAG?",
        "original_query": "What is RAG?",
        "conversation_history": [],
        "router_decision": "",
        "retrieved_chunks": [],
        "rewritten_query": None,
        "answer": "",
        "source_chunk_ids": [],
        "confidence_score": 0.0,
        "retry_count": 0,
    }
    state.update(overrides)
    return state


def _mock_openai_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def _mock_retrieval_result(
    chunk_id: str = "chunk-001",
    text: str = "Retrieval-Augmented Generation.",
) -> MagicMock:
    result = MagicMock()
    result.chunk_id = chunk_id
    result.text = text
    result.source_filename = "paper.pdf"
    result.page_number = 1
    result.section_heading = "Introduction"
    result.chunk_index = 0
    result.token_count = 10
    return result


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestRAGState:
    def test_can_be_instantiated_with_all_required_fields(self):
        from orchestration.state import RAGState
        state: RAGState = _base_state()
        required = {
            "query", "original_query", "conversation_history",
            "router_decision", "retrieved_chunks", "rewritten_query",
            "answer", "source_chunk_ids", "confidence_score", "retry_count",
        }
        assert required.issubset(state.keys())

    def test_initial_retry_count_is_zero(self):
        assert _base_state()["retry_count"] == 0

    def test_original_query_preserved_after_update(self):
        state = _base_state(original_query="original")
        state["query"] = "rephrased version"
        assert state["original_query"] == "original"

    def test_rewritten_query_defaults_to_none(self):
        assert _base_state()["rewritten_query"] is None

    def test_confidence_score_defaults_to_zero(self):
        assert _base_state()["confidence_score"] == 0.0


# ---------------------------------------------------------------------------
# Routing function tests (pure — no LLM)
# ---------------------------------------------------------------------------

class TestRoutingFunctions:
    def test_route_after_router_returns_retriever_for_retrieve(self):
        from orchestration.graph import route_after_router
        assert route_after_router(_base_state(router_decision="retrieve")) == "retriever"

    def test_route_after_router_returns_generator_for_conversational(self):
        from orchestration.graph import route_after_router
        assert route_after_router(_base_state(router_decision="conversational")) == "generator"

    def test_route_after_generator_returns_end_when_confidence_high(self):
        from orchestration.graph import route_after_generator
        assert route_after_generator(_base_state(confidence_score=0.8, retry_count=0)) == END

    def test_route_after_generator_returns_end_at_exactly_04(self):
        from orchestration.graph import route_after_generator
        assert route_after_generator(_base_state(confidence_score=0.4, retry_count=0)) == END

    def test_route_after_generator_returns_end_when_max_retries_reached(self):
        from orchestration.graph import route_after_generator
        assert route_after_generator(_base_state(confidence_score=0.1, retry_count=2)) == END

    def test_route_after_generator_returns_rewriter_when_low_confidence(self):
        from orchestration.graph import route_after_generator
        assert route_after_generator(_base_state(confidence_score=0.2, retry_count=0)) == "rewriter"

    def test_route_after_generator_returns_rewriter_at_retry_count_1(self):
        from orchestration.graph import route_after_generator
        assert route_after_generator(_base_state(confidence_score=0.1, retry_count=1)) == "rewriter"


# ---------------------------------------------------------------------------
# Router node tests  (now async — uses AsyncOpenAI)
# ---------------------------------------------------------------------------

class TestRouterNode:
    async def _call_router(self, json_content: str, query: str = "What is RAG?") -> dict:
        with patch("orchestration.nodes.router.openai") as mock_oai:
            mock_oai.AsyncOpenAI.return_value.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response(json_content)
            )
            from orchestration.nodes.router import router_node
            return await router_node(_base_state(query=query))

    @pytest.mark.asyncio
    async def test_sets_retrieve_decision(self):
        result = await self._call_router('{"decision": "retrieve"}')
        assert result["router_decision"] == "retrieve"

    @pytest.mark.asyncio
    async def test_sets_conversational_decision(self):
        result = await self._call_router('{"decision": "conversational"}')
        assert result["router_decision"] == "conversational"

    @pytest.mark.asyncio
    async def test_defaults_to_retrieve_on_unexpected_value(self):
        result = await self._call_router('{"decision": "unknown_value"}')
        assert result["router_decision"] == "retrieve"

    @pytest.mark.asyncio
    async def test_defaults_to_retrieve_on_malformed_json(self):
        result = await self._call_router("not valid json at all")
        assert result["router_decision"] == "retrieve"

    @pytest.mark.asyncio
    async def test_returns_only_router_decision_key(self):
        result = await self._call_router('{"decision": "retrieve"}')
        assert set(result.keys()) == {"router_decision"}


# ---------------------------------------------------------------------------
# Generator node tests  (now async — uses ChatOpenAI)
# ---------------------------------------------------------------------------

class TestGeneratorNode:
    async def _call_generator(
        self,
        answer: str = "The answer.",
        source_ids: list[str] | None = None,
        confidence: float = 0.85,
        chunks: list | None = None,
    ) -> dict:
        payload = json.dumps({
            "answer": answer,
            "source_chunk_ids": source_ids or ["chunk-001"],
            "confidence": confidence,
        })
        mock_response = MagicMock()
        mock_response.content = payload

        with patch("orchestration.nodes.generator.ChatOpenAI") as MockChatOpenAI:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            MockChatOpenAI.return_value.with_config.return_value = mock_llm

            from orchestration.nodes.generator import generator_node
            state = _base_state(retrieved_chunks=chunks or [_mock_retrieval_result()])
            return await generator_node(state)

    @pytest.mark.asyncio
    async def test_parses_answer_field(self):
        result = await self._call_generator(answer="RAG combines retrieval and generation.")
        assert result["answer"] == "RAG combines retrieval and generation."

    @pytest.mark.asyncio
    async def test_parses_source_chunk_ids(self):
        result = await self._call_generator(source_ids=["abc", "def"])
        assert result["source_chunk_ids"] == ["abc", "def"]

    @pytest.mark.asyncio
    async def test_parses_confidence_score(self):
        result = await self._call_generator(confidence=0.72)
        assert abs(result["confidence_score"] - 0.72) < 1e-6

    @pytest.mark.asyncio
    async def test_updates_all_three_state_fields(self):
        result = await self._call_generator()
        assert {"answer", "source_chunk_ids", "confidence_score"}.issubset(result.keys())

    @pytest.mark.asyncio
    async def test_confidence_stored_as_float(self):
        result = await self._call_generator(confidence=0.9)
        assert isinstance(result["confidence_score"], float)


# ---------------------------------------------------------------------------
# Rewriter node tests  (now async — uses AsyncOpenAI)
# ---------------------------------------------------------------------------

class TestRewriterNode:
    async def _call_rewriter(
        self,
        rewritten: str = "Alternative query phrasing",
        retry_count: int = 0,
    ) -> dict:
        payload = json.dumps({"rewritten_query": rewritten})
        with patch("orchestration.nodes.rewriter.openai") as mock_oai:
            mock_oai.AsyncOpenAI.return_value.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response(payload)
            )
            from orchestration.nodes.rewriter import rewriter_node
            state = _base_state(
                answer="The context did not contain enough detail.",
                retry_count=retry_count,
            )
            return await rewriter_node(state)

    @pytest.mark.asyncio
    async def test_increments_retry_count_by_one(self):
        result = await self._call_rewriter(retry_count=0)
        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_from_nonzero_retry_count(self):
        result = await self._call_rewriter(retry_count=1)
        assert result["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_sets_rewritten_query_to_nonempty_string(self):
        result = await self._call_rewriter(rewritten="Better phrasing of the query")
        assert isinstance(result["rewritten_query"], str)
        assert len(result["rewritten_query"]) > 0

    @pytest.mark.asyncio
    async def test_rewritten_query_matches_llm_output(self):
        result = await self._call_rewriter(rewritten="Specific alternative query")
        assert result["rewritten_query"] == "Specific alternative query"


# ---------------------------------------------------------------------------
# Integration tests (all external calls mocked)
# ---------------------------------------------------------------------------

class TestIntegration:
    """Run the full compiled graph with all LLM and Qdrant calls mocked."""

    async def _run_graph(
        self,
        router_decision: str = "retrieve",
        confidence: float = 0.9,
        max_retries_hit: bool = False,
    ) -> dict:
        router_payload = json.dumps({"decision": router_decision})
        gen_low = json.dumps({"answer": "Insufficient context.", "source_chunk_ids": [], "confidence": 0.1})
        gen_high = json.dumps({"answer": "Final answer.", "source_chunk_ids": ["c1"], "confidence": confidence})

        gen_mock_responses = (
            [MagicMock(content=gen_low), MagicMock(content=gen_low), MagicMock(content=gen_high)]
            if max_retries_hit
            else [MagicMock(content=gen_high)]
        )

        rewriter_payload = json.dumps({"rewritten_query": "rephrased query"})

        with (
            patch("orchestration.nodes.router.openai") as mock_router_oai,
            patch("orchestration.nodes.generator.ChatOpenAI") as MockChatOpenAI,
            patch("orchestration.nodes.rewriter.openai") as mock_rew_oai,
            patch("orchestration.nodes.retriever_node.Retriever") as MockRetriever,
            patch("orchestration.app._app", None),
        ):
            mock_router_oai.AsyncOpenAI.return_value.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response(router_payload)
            )
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=gen_mock_responses)
            MockChatOpenAI.return_value.with_config.return_value = mock_llm

            mock_rew_oai.AsyncOpenAI.return_value.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response(rewriter_payload)
            )
            MockRetriever.return_value.retrieve.return_value = [_mock_retrieval_result()]

            import orchestration.nodes.retriever_node as rn_mod
            rn_mod._retriever = None

            from orchestration.app import run_query
            return await run_query("What is RAG?")

    @pytest.mark.asyncio
    async def test_returns_dict_with_all_required_keys(self):
        result = await self._run_graph()
        required = {"answer", "source_chunk_ids", "confidence_score",
                    "retry_count", "router_decision"}
        assert required.issubset(result.keys())

    @pytest.mark.asyncio
    async def test_retrieve_path_populates_answer(self):
        result = await self._run_graph(router_decision="retrieve", confidence=0.9)
        assert result["answer"] == "Final answer."
        assert result["router_decision"] == "retrieve"

    @pytest.mark.asyncio
    async def test_conversational_path_bypasses_retriever(self):
        with (
            patch("orchestration.nodes.router.openai") as mock_router_oai,
            patch("orchestration.nodes.generator.ChatOpenAI") as MockChatOpenAI,
            patch("orchestration.nodes.retriever_node.Retriever") as MockRetriever,
            patch("orchestration.app._app", None),
        ):
            mock_router_oai.AsyncOpenAI.return_value.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response('{"decision": "conversational"}')
            )
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                return_value=MagicMock(content='{"answer": "Hello!", "source_chunk_ids": [], "confidence": 0.95}')
            )
            MockChatOpenAI.return_value.with_config.return_value = mock_llm

            import orchestration.nodes.retriever_node as rn_mod
            rn_mod._retriever = None

            from orchestration.app import run_query
            await run_query("Hello!")

            MockRetriever.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_loop_capped_at_two_retries(self):
        result = await self._run_graph(max_retries_hit=True)
        assert result["retry_count"] <= 2

    @pytest.mark.asyncio
    async def test_confidence_score_in_result(self):
        result = await self._run_graph(confidence=0.88)
        assert isinstance(result["confidence_score"], float)
