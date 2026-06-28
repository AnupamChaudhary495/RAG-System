"""Hybrid retriever: dense + sparse search → RRF → cross-encoder reranking.

Public interface::

    from retrieval.retriever import Retriever, RetrievalResult

    r = Retriever()
    results = r.retrieve("what is reciprocal rank fusion?", top_k=5)
    for res in results:
        print(res.source_filename, res.reranker_score, res.text[:80])
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from embedding.embedder import BGE3Embedder
from retrieval.dense_search import dense_search
from retrieval.fusion import RRFResult, reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse_search import sparse_search


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single ranked retrieval result returned to the caller."""

    chunk_id: str
    text: str
    reranker_score: float
    source_filename: str
    page_number: int | None
    section_heading: str | None
    chunk_index: int
    token_count: int


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """Hybrid retriever combining dense, sparse, RRF, and cross-encoder reranking.

    The BGE-M3 embedder is loaded at construction time (shared weight loading
    cost).  The cross-encoder reranker is lazy-loaded on the first call to
    retrieve() to avoid paying its startup cost unless retrieval is performed.

    Args:
        qdrant_url:        URL of the running Qdrant instance.
        collection_name:   Qdrant collection to search.
        dense_candidates:  Number of dense search candidates before fusion.
        sparse_candidates: Number of sparse search candidates before fusion.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_chunks",
        dense_candidates: int = 50,
        sparse_candidates: int = 50,
    ) -> None:
        self._qdrant_url = qdrant_url
        self._collection_name = collection_name
        self._dense_candidates = dense_candidates
        self._sparse_candidates = sparse_candidates

        self._client = QdrantClient(url=qdrant_url)
        self._embedder = BGE3Embedder()
        self._reranker: CrossEncoderReranker | None = None  # lazy-loaded

        self._verify_collection()

    def _verify_collection(self) -> None:
        """Raise ValueError if the target collection does not exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            raise ValueError(
                f"Qdrant collection '{self._collection_name}' does not exist. "
                "Run Phase 2 (embedding/ingest_vectors.py) first."
            )

    def _get_reranker(self) -> CrossEncoderReranker:
        """Lazy-load and cache the cross-encoder reranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run the full hybrid retrieval pipeline for a single query.

        Pipeline:
            1. Embed query → dense vector (1024-dim) + sparse vector
            2. Dense search  → up to dense_candidates results
            3. Sparse search → up to sparse_candidates results
            4. RRF fusion    → deduplicated, score-merged candidates
            5. Cross-encoder rerank → top_k final results

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return (default 5).

        Returns:
            List of RetrievalResult objects sorted by descending reranker score.
        """
        # 1. Embed query
        encoded = self._embedder.encode_batch([query])
        dense_vec: list[float] = encoded.dense[0]
        sparse_vec: dict[int, float] = encoded.sparse[0]

        # 2. Dense search
        dense_hits = dense_search(
            self._client, self._collection_name, dense_vec, self._dense_candidates
        )
        print(f"[dense]   {len(dense_hits)} results")

        # 3. Sparse search
        sparse_hits = sparse_search(
            self._client, self._collection_name, sparse_vec, self._sparse_candidates
        )
        print(f"[sparse]  {len(sparse_hits)} results")

        # 4. RRF fusion (dense list first → its payload takes priority on ties)
        fused: list[RRFResult] = reciprocal_rank_fusion(dense_hits, sparse_hits)
        print(f"[rrf]     {len(fused)} unique candidates after fusion")

        # 5. Cross-encoder rerank — sort defensively in case the reranker
        #    returns candidates in a non-deterministic order.
        reranker = self._get_reranker()
        ranked = sorted(
            reranker.rerank(query, fused, top_k),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"[rerank]  top {len(ranked)} selected")

        # 6. Build output dataclasses
        return [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                reranker_score=float(score),
                source_filename=candidate.payload.get("source_filename", ""),
                page_number=candidate.payload.get("page_number"),
                section_heading=candidate.payload.get("section_heading"),
                chunk_index=candidate.payload.get("chunk_index", 0),
                token_count=candidate.payload.get("token_count", 0),
            )
            for candidate, score in ranked
        ]
