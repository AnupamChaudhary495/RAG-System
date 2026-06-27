"""Qdrant collection management and point upsert logic.

Creates a "rag_chunks" collection with named dense (1024-dim cosine, HNSW)
and sparse (BM42-style) vector configs.  Existing collections are left
untouched — a warning is printed instead of dropping data.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_chunks"
DENSE_DIM = 1024
HNSW_M = 16
HNSW_EF_CONSTRUCT = 200

# Keys that every chunk payload must carry
REQUIRED_PAYLOAD_KEYS = {
    "text",
    "source_filename",
    "page_number",
    "timestamp",
    "section_heading",
    "chunk_index",
    "token_count",
}


def build_client(qdrant_url: str) -> QdrantClient:
    """Create and return a QdrantClient connected to qdrant_url."""
    return QdrantClient(url=qdrant_url)


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Create the Qdrant collection if it does not already exist.

    If the collection exists, logs a warning and returns without modifying it.

    Args:
        client:          Connected QdrantClient instance.
        collection_name: Target collection name.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        logger.warning(
            "[vector_store] Collection '%s' already exists — skipping creation.",
            collection_name,
        )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams())
        },
    )
    logger.info("[vector_store] Created collection '%s'.", collection_name)


def to_sparse_vector(sparse_dict: dict[int, float]) -> SparseVector:
    """Convert a {token_id: weight} dict to a Qdrant SparseVector.

    Args:
        sparse_dict: Integer token IDs mapped to float weights.

    Returns:
        SparseVector with parallel indices and values lists.
    """
    indices = list(sparse_dict.keys())
    values = [sparse_dict[i] for i in indices]
    return SparseVector(indices=indices, values=values)


def build_points(
    chunks: list[dict],
    dense_vecs: list[list[float]],
    sparse_vecs: list[dict[int, float]],
) -> list[PointStruct]:
    """Assemble PointStruct objects from chunks and their embedding vectors.

    Args:
        chunks:      List of chunk dicts from Phase 1 JSON.
        dense_vecs:  Parallel list of 1024-dim dense vectors.
        sparse_vecs: Parallel list of {token_id: weight} sparse dicts.

    Returns:
        List of PointStruct objects ready for upsert.
    """
    points: list[PointStruct] = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector={
                    "dense": dense,
                    "sparse": to_sparse_vector(sparse),
                },
                payload={
                    "text": chunk["text"],
                    "source_filename": chunk["source_filename"],
                    "page_number": chunk["page_number"],
                    "timestamp": chunk["timestamp"],
                    "section_heading": chunk["section_heading"],
                    "chunk_index": chunk["chunk_index"],
                    "token_count": chunk["token_count"],
                },
            )
        )
    return points


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Upsert a batch of PointStructs into the collection.

    Args:
        client:          Connected QdrantClient instance.
        points:          Points to upsert.
        collection_name: Target collection name.
    """
    client.upsert(collection_name=collection_name, points=points, wait=True)
