"""Unit tests for Phase 3 — hybrid retrieval engine.

RRF tests are pure-function (no mocking needed).
Reranker and Retriever tests mock FlagReranker and QdrantClient respectively
so no live GPU or Qdrant instance is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit(chunk_id: str, text: str = "sample text", score: float = 1.0) -> tuple:
    """Build a (chunk_id, payload, score) search hit tuple."""
    return (
        chunk_id,
        {
            "text": text,
            "source_filename": "doc.pdf",
            "page_number": 1,
            "timestamp": "2026-06-27T10:00:00+00:00",
            "section_heading": "Introduction",
            "chunk_index": 0,
            "token_count": 10,
        },
        score,
    )


# ---------------------------------------------------------------------------
# RRF tests — pure Python, no mocking
# ---------------------------------------------------------------------------

from retrieval.fusion import reciprocal_rank_fusion, RRFResult


class TestRRF:
    def test_document_in_both_lists_scores_higher_than_single_list(self):
        dense = [_hit("A"), _hit("B")]
        sparse = [_hit("A"), _hit("C")]   # "A" appears in both

        results = reciprocal_rank_fusion(dense, sparse)
        by_id = {r.chunk_id: r.rrf_score for r in results}

        # "A" in both lists, "B" and "C" in one list each
        assert by_id["A"] > by_id["B"]
        assert by_id["A"] > by_id["C"]

    def test_rrf_scores_are_strictly_positive(self):
        dense = [_hit("X"), _hit("Y")]
        sparse = [_hit("Z")]

        results = reciprocal_rank_fusion(dense, sparse)
        for r in results:
            assert r.rrf_score > 0.0, f"Expected positive score for {r.chunk_id}"

    def test_fusion_deduplicates_chunk_ids(self):
        dense = [_hit("A"), _hit("B"), _hit("C")]
        sparse = [_hit("B"), _hit("C"), _hit("D")]

        results = reciprocal_rank_fusion(dense, sparse)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found after fusion"

    def test_union_of_ids_is_complete(self):
        dense = [_hit("A"), _hit("B")]
        sparse = [_hit("C"), _hit("D")]

        results = reciprocal_rank_fusion(dense, sparse)
        ids = {r.chunk_id for r in results}
        assert ids == {"A", "B", "C", "D"}

    def test_empty_sparse_list_handled_gracefully(self):
        dense = [_hit("A"), _hit("B")]
        results = reciprocal_rank_fusion(dense, [])
        assert len(results) == 2
        assert all(r.rrf_score > 0 for r in results)

    def test_empty_dense_list_handled_gracefully(self):
        sparse = [_hit("X")]
        results = reciprocal_rank_fusion([], sparse)
        assert len(results) == 1
        assert results[0].chunk_id == "X"

    def test_both_empty_returns_empty_list(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_results_sorted_descending_by_rrf_score(self):
        # "A" is rank-1 in both lists → highest score
        dense = [_hit("A"), _hit("B"), _hit("C")]
        sparse = [_hit("A"), _hit("D"), _hit("E")]

        results = reciprocal_rank_fusion(dense, sparse)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_dense_payload_takes_priority_over_sparse(self):
        """When a chunk_id appears in both lists, dense payload is kept."""
        dense_payload = {"text": "dense text", "source_filename": "dense.pdf",
                         "page_number": 1, "section_heading": None,
                         "chunk_index": 0, "token_count": 5}
        sparse_payload = {"text": "sparse text", "source_filename": "sparse.pdf",
                          "page_number": 2, "section_heading": "S",
                          "chunk_index": 1, "token_count": 6}

        dense = [("shared-id", dense_payload, 0.9)]
        sparse = [("shared-id", sparse_payload, 0.8)]

        results = reciprocal_rank_fusion(dense, sparse)
        assert results[0].text == "dense text"

    def test_rrf_formula_values(self):
        """Verify the exact formula: 1/(60+1) + 1/(60+1) for rank-1 in both."""
        dense = [_hit("A")]
        sparse = [_hit("A")]

        results = reciprocal_rank_fusion(dense, sparse)
        expected = 1 / (60 + 1) + 1 / (60 + 1)
        assert abs(results[0].rrf_score - expected) < 1e-9


# ---------------------------------------------------------------------------
# Reranker tests — FlagReranker mocked
# ---------------------------------------------------------------------------

from retrieval.reranker import CrossEncoderReranker


def _rrf_result(chunk_id: str, text: str) -> RRFResult:
    return RRFResult(
        chunk_id=chunk_id,
        text=text,
        rrf_score=0.5,
        payload={"text": text},
    )


class TestCrossEncoderReranker:
    def _make_reranker(self, scores: list[float]) -> CrossEncoderReranker:
        with patch("retrieval.reranker.FlagReranker") as MockFR:
            instance = MockFR.return_value
            instance.compute_score.return_value = scores
            reranker = CrossEncoderReranker()
            reranker._model = instance
        return reranker

    def test_returns_top_k_in_descending_order(self):
        reranker = self._make_reranker([1.0, 3.0, 2.0])
        candidates = [
            _rrf_result("A", "text A"),
            _rrf_result("B", "text B"),
            _rrf_result("C", "text C"),
        ]
        ranked = reranker.rerank("query", candidates, top_k=2)

        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]        # descending score order
        assert ranked[0][0].chunk_id == "B"        # score 3.0 → top
        assert ranked[1][0].chunk_id == "C"        # score 2.0 → second

    def test_called_with_correct_pair_format(self):
        with patch("retrieval.reranker.FlagReranker") as MockFR:
            instance = MockFR.return_value
            instance.compute_score.return_value = [0.5, 0.9]
            reranker = CrossEncoderReranker()
            reranker._model = instance

        query = "test query"
        candidates = [_rrf_result("X", "chunk X"), _rrf_result("Y", "chunk Y")]
        reranker.rerank(query, candidates, top_k=2)

        call_args = instance.compute_score.call_args
        pairs = call_args[0][0]
        assert pairs == [["test query", "chunk X"], ["test query", "chunk Y"]]

    def test_empty_candidates_returns_empty_list(self):
        reranker = self._make_reranker([])
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_top_k_larger_than_candidates_returns_all(self):
        reranker = self._make_reranker([0.5])
        candidates = [_rrf_result("A", "text")]
        ranked = reranker.rerank("query", candidates, top_k=10)
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# Retriever tests — Qdrant client + BGE3Embedder mocked
# ---------------------------------------------------------------------------

from retrieval.retriever import Retriever, RetrievalResult
from embedding.embedder import DENSE_DIM


def _make_mock_hit(chunk_id: str, text: str = "chunk text"):
    """Build a mock Qdrant ScoredPoint."""
    hit = MagicMock()
    hit.id = chunk_id
    hit.score = 0.9
    hit.payload = {
        "text": text,
        "source_filename": "paper.pdf",
        "page_number": 1,
        "timestamp": "2026-06-27T10:00:00+00:00",
        "section_heading": "Methods",
        "chunk_index": 0,
        "token_count": 15,
    }
    return hit


def _make_retriever(
    collection_exists: bool = True,
    dense_hits: list | None = None,
    sparse_hits: list | None = None,
    reranker_scores: list[float] | None = None,
):
    """Build a Retriever with all external dependencies mocked."""
    import numpy as np

    dense_hits = dense_hits or [_make_mock_hit(f"id-{i}") for i in range(5)]
    sparse_hits = sparse_hits or [_make_mock_hit(f"id-{i}") for i in range(5)]
    reranker_scores = reranker_scores or [float(i) for i in range(len(dense_hits))]

    with (
        patch("retrieval.retriever.QdrantClient") as MockQdrant,
        patch("retrieval.retriever.BGE3Embedder") as MockEmbedder,
    ):
        # Qdrant client mock
        mock_client = MockQdrant.return_value
        col_mock = MagicMock()
        col_mock.name = "rag_chunks"
        mock_client.get_collections.return_value.collections = (
            [col_mock] if collection_exists else []
        )
        mock_client.search.side_effect = [dense_hits, sparse_hits]

        # Embedder mock
        mock_encoded = MagicMock()
        mock_encoded.dense = [[0.1] * DENSE_DIM]
        mock_encoded.sparse = [{1: 0.5, 2: 0.3}]
        MockEmbedder.return_value.encode_batch.return_value = mock_encoded

        retriever = Retriever()
        retriever._client = mock_client

    # Inject mock reranker (bypasses lazy-load)
    mock_reranker = MagicMock()
    rrf_candidates = reciprocal_rank_fusion(
        [(str(h.id), h.payload, h.score) for h in dense_hits],
        [(str(h.id), h.payload, h.score) for h in sparse_hits],
    )
    top_k = min(5, len(rrf_candidates))
    mock_reranker.rerank.return_value = [
        (rrf_candidates[i], reranker_scores[i]) for i in range(top_k)
    ]
    retriever._reranker = mock_reranker

    return retriever, mock_client


class TestRetriever:
    def test_returns_exactly_top_k_results(self):
        retriever, _ = _make_retriever()
        results = retriever.retrieve("test query", top_k=5)
        assert len(results) == 5

    def test_all_required_fields_populated(self):
        retriever, _ = _make_retriever()
        results = retriever.retrieve("test query", top_k=5)

        for r in results:
            assert isinstance(r, RetrievalResult)
            assert r.chunk_id
            assert r.text
            assert isinstance(r.reranker_score, float)
            assert r.source_filename
            assert isinstance(r.chunk_index, int)
            assert isinstance(r.token_count, int)

    def test_text_matches_qdrant_payload(self):
        hit = _make_mock_hit("special-id", text="unique chunk content")
        retriever, _ = _make_retriever(
            dense_hits=[hit],
            sparse_hits=[hit],
            reranker_scores=[1.0],
        )
        results = retriever.retrieve("query", top_k=1)
        assert results[0].text == "unique chunk content"

    def test_dense_and_sparse_search_called_once_per_query(self):
        retriever, mock_client = _make_retriever()
        retriever.retrieve("query", top_k=5)
        assert mock_client.search.call_count == 2

    def test_raises_value_error_when_collection_missing(self):
        with (
            patch("retrieval.retriever.QdrantClient") as MockQdrant,
            patch("retrieval.retriever.BGE3Embedder"),
        ):
            mock_client = MockQdrant.return_value
            mock_client.get_collections.return_value.collections = []
            with pytest.raises(ValueError, match="rag_chunks"):
                Retriever()

    def test_results_sorted_descending_by_reranker_score(self):
        hits = [_make_mock_hit(f"id-{i}") for i in range(3)]
        scores = [3.0, 1.0, 2.0]

        retriever, _ = _make_retriever(
            dense_hits=hits,
            sparse_hits=hits,
            reranker_scores=scores,
        )
        # Override reranker to return scores in given order
        rrf_candidates = reciprocal_rank_fusion(
            [(str(h.id), h.payload, h.score) for h in hits],
            [(str(h.id), h.payload, h.score) for h in hits],
        )
        retriever._reranker.rerank.return_value = [
            (rrf_candidates[i], scores[i]) for i in range(3)
        ]
        results = retriever.retrieve("query", top_k=3)
        result_scores = [r.reranker_score for r in results]
        assert result_scores == sorted(result_scores, reverse=True)
