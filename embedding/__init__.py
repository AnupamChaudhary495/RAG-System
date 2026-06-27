"""Embedding and vector storage layer for the Enterprise RAG system.

Phase 2: reads Phase 1 chunk JSON, embeds with BGE-M3 (dense + sparse),
and upserts into a Qdrant collection named "rag_chunks".
"""

from embedding.ingest_vectors import ingest_vectors

__all__ = ["ingest_vectors"]
