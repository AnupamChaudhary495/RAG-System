"""Unit tests for the Phase 2 embedding and vector storage layer.

All tests that touch Qdrant mock the client — no live instance required.
BGE-M3 model loading is also mocked so tests run without a GPU or the
multi-GB model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(**overrides) -> dict:
    """Return a minimal valid Phase 1 chunk dict."""
    base = {
        "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
        "text": "Retrieval-Augmented Generation improves factual accuracy.",
        "source_filename": "paper.pdf",
        "page_number": 1,
        "timestamp": "2026-06-27T10:00:00+00:00",
        "section_heading": "Introduction",
        "chunk_index": 0,
        "token_count": 9,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Embedder tests (BGE-M3 model mocked)
# ---------------------------------------------------------------------------

class TestBGE3Embedder:
    """Test BGE3Embedder without loading the real model."""

    def _make_embedder(self):
        """Return a BGE3Embedder whose internal BGEM3FlagModel is mocked."""
        import numpy as np
        from embedding.embedder import BGE3Embedder, DENSE_DIM

        with patch("embedding.embedder.BGEM3FlagModel") as MockModel:
            instance = MockModel.return_value
            # Simulate model.encode() output
            instance.encode.return_value = {
                "dense_vecs": [np.ones(DENSE_DIM, dtype="float32")],
                "lexical_weights": [{"42": 0.8, "17": 0.3}],
            }
            embedder = BGE3Embedder()
            embedder._model = instance          # inject mock directly
        return embedder, instance

    def test_dense_vector_has_1024_dimensions(self):
        from embedding.embedder import DENSE_DIM
        import numpy as np

        embedder, mock_model = self._make_embedder()
        mock_model.encode.return_value = {
            "dense_vecs": [__import__("numpy").ones(DENSE_DIM, dtype="float32")],
            "lexical_weights": [{"1": 0.5}],
        }
        result = embedder.encode_batch(["test sentence"])
        assert len(result.dense[0]) == DENSE_DIM

    def test_dense_vector_values_are_floats(self):
        from embedding.embedder import DENSE_DIM
        import numpy as np

        embedder, mock_model = self._make_embedder()
        mock_model.encode.return_value = {
            "dense_vecs": [np.array([0.1] * DENSE_DIM, dtype="float32")],
            "lexical_weights": [{"1": 0.9}],
        }
        result = embedder.encode_batch(["test"])
        assert all(isinstance(v, float) for v in result.dense[0])

    def test_sparse_output_is_nonempty_dict(self):
        from embedding.embedder import DENSE_DIM
        import numpy as np

        embedder, mock_model = self._make_embedder()
        mock_model.encode.return_value = {
            "dense_vecs": [np.ones(DENSE_DIM, dtype="float32")],
            "lexical_weights": [{"42": 0.8, "17": 0.3}],
        }
        result = embedder.encode_batch(["test"])
        assert isinstance(result.sparse[0], dict)
        assert len(result.sparse[0]) > 0

    def test_sparse_values_are_floats(self):
        from embedding.embedder import DENSE_DIM
        import numpy as np

        embedder, mock_model = self._make_embedder()
        mock_model.encode.return_value = {
            "dense_vecs": [np.ones(DENSE_DIM, dtype="float32")],
            "lexical_weights": [{"99": 0.75}],
        }
        result = embedder.encode_batch(["test"])
        for v in result.sparse[0].values():
            assert isinstance(v, float)

    def test_sparse_indices_are_ints(self):
        """FlagEmbedding returns string token IDs — embedder must cast to int."""
        from embedding.embedder import DENSE_DIM
        import numpy as np

        embedder, mock_model = self._make_embedder()
        mock_model.encode.return_value = {
            "dense_vecs": [np.ones(DENSE_DIM, dtype="float32")],
            "lexical_weights": [{"1024": 0.6, "2048": 0.4}],
        }
        result = embedder.encode_batch(["test"])
        for key in result.sparse[0]:
            assert isinstance(key, int), f"Expected int key, got {type(key)}"

    def test_encode_batch_raises_on_empty_input(self):
        from embedding.embedder import BGE3Embedder

        with patch("embedding.embedder.BGEM3FlagModel"):
            embedder = BGE3Embedder()
        with pytest.raises(ValueError, match="non-empty"):
            embedder.encode_batch([])

    def test_batch_count_matches_input_length(self):
        from embedding.embedder import DENSE_DIM
        import numpy as np

        embedder, mock_model = self._make_embedder()
        n = 3
        mock_model.encode.return_value = {
            "dense_vecs": [np.ones(DENSE_DIM, dtype="float32")] * n,
            "lexical_weights": [{"1": 0.5}] * n,
        }
        result = embedder.encode_batch(["a", "b", "c"])
        assert len(result.dense) == n
        assert len(result.sparse) == n


# ---------------------------------------------------------------------------
# SparseVector conversion tests
# ---------------------------------------------------------------------------

class TestSparseVectorConversion:
    def test_indices_and_values_same_length(self):
        from embedding.vector_store import to_sparse_vector

        sparse_dict = {10: 0.9, 200: 0.4, 3000: 0.1}
        sv = to_sparse_vector(sparse_dict)
        assert len(sv.indices) == len(sv.values) == 3

    def test_indices_match_keys(self):
        from embedding.vector_store import to_sparse_vector

        sparse_dict = {5: 0.7, 10: 0.3}
        sv = to_sparse_vector(sparse_dict)
        assert set(sv.indices) == {5, 10}

    def test_values_match_weights(self):
        from embedding.vector_store import to_sparse_vector

        sparse_dict = {1: 0.8}
        sv = to_sparse_vector(sparse_dict)
        assert sv.values == [0.8]

    def test_empty_sparse_dict_produces_empty_vector(self):
        from embedding.vector_store import to_sparse_vector

        sv = to_sparse_vector({})
        assert sv.indices == []
        assert sv.values == []


# ---------------------------------------------------------------------------
# Qdrant collection config tests (client mocked)
# ---------------------------------------------------------------------------

class TestCollectionConfig:
    def _mock_client(self, existing_collections: list[str] | None = None) -> MagicMock:
        client = MagicMock()
        names = existing_collections or []
        # NOTE: MagicMock(name=...) sets the mock's repr label, NOT a .name
        # attribute. Build the attribute explicitly to avoid the trap.
        cols = []
        for n in names:
            m = MagicMock()
            m.name = n
            cols.append(m)
        client.get_collections.return_value.collections = cols
        return client

    def test_create_collection_called_with_dense_and_sparse_configs(self):
        from embedding.vector_store import ensure_collection

        client = self._mock_client()
        ensure_collection(client, "test_col")

        call_kwargs = client.create_collection.call_args.kwargs
        # Dense config
        assert "dense" in call_kwargs["vectors_config"]
        # Sparse config
        assert "sparse" in call_kwargs["sparse_vectors_config"]

    def test_dense_config_has_correct_size_and_distance(self):
        from embedding.vector_store import ensure_collection, DENSE_DIM
        from qdrant_client.http.models import Distance

        client = self._mock_client()
        ensure_collection(client, "test_col")

        dense_cfg = client.create_collection.call_args.kwargs["vectors_config"]["dense"]
        assert dense_cfg.size == DENSE_DIM
        assert dense_cfg.distance == Distance.COSINE

    def test_dense_config_has_hnsw_params(self):
        from embedding.vector_store import ensure_collection, HNSW_M, HNSW_EF_CONSTRUCT

        client = self._mock_client()
        ensure_collection(client, "test_col")

        dense_cfg = client.create_collection.call_args.kwargs["vectors_config"]["dense"]
        assert dense_cfg.hnsw_config.m == HNSW_M
        assert dense_cfg.hnsw_config.ef_construct == HNSW_EF_CONSTRUCT

    def test_skips_creation_when_collection_exists(self):
        from embedding.vector_store import ensure_collection

        client = self._mock_client(existing_collections=["rag_chunks"])
        ensure_collection(client, "rag_chunks")

        client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# ingest_vectors validation tests
# ---------------------------------------------------------------------------

class TestIngestVectorsValidation:
    def test_raises_file_not_found_for_missing_json(self):
        from embedding.ingest_vectors import ingest_vectors

        with pytest.raises(FileNotFoundError):
            ingest_vectors("/nonexistent/chunks.json")

    def test_raises_value_error_for_missing_key(self, tmp_path):
        """A chunk missing 'token_count' should raise ValueError with chunk_id."""
        import json
        from embedding.ingest_vectors import ingest_vectors

        bad_chunk = _make_chunk()
        del bad_chunk["token_count"]  # remove a required key

        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(json.dumps([bad_chunk]), encoding="utf-8")

        with pytest.raises(ValueError, match=bad_chunk["chunk_id"]):
            ingest_vectors(str(chunks_file))

    def test_raises_value_error_reports_missing_key_name(self, tmp_path):
        import json
        from embedding.ingest_vectors import ingest_vectors

        bad_chunk = _make_chunk()
        del bad_chunk["source_filename"]

        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(json.dumps([bad_chunk]), encoding="utf-8")

        with pytest.raises(ValueError, match="source_filename"):
            ingest_vectors(str(chunks_file))

    def test_full_pipeline_with_mocked_embedder_and_qdrant(self, tmp_path):
        """End-to-end happy path: mocked model + mocked Qdrant client."""
        import json
        import numpy as np
        from embedding.ingest_vectors import ingest_vectors
        from embedding.embedder import DENSE_DIM

        chunks = [_make_chunk(chunk_id=f"550e8400-e29b-41d4-a716-44665544{i:04d}")
                  for i in range(3)]
        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(json.dumps(chunks), encoding="utf-8")

        mock_encoded = MagicMock()
        mock_encoded.dense = [[0.1] * DENSE_DIM for _ in chunks]
        mock_encoded.sparse = [{1: 0.5, 2: 0.3} for _ in chunks]

        with (
            patch("embedding.ingest_vectors.build_client") as mock_build,
            patch("embedding.ingest_vectors.BGE3Embedder") as MockEmbedder,
        ):
            mock_client = MagicMock()
            mock_client.get_collections.return_value.collections = []
            mock_build.return_value = mock_client
            MockEmbedder.return_value.encode_batch.return_value = mock_encoded

            result = ingest_vectors(str(chunks_file))

        assert result == 3
        assert mock_client.upsert.call_count == 1  # 3 chunks, 1 batch of 32
