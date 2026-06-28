"""Qdrant dense vector search.

Accepts an injected QdrantClient — does not instantiate one internally,
making it independently testable with a mock client.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector


# Type alias for a search hit: (chunk_id, payload, score)
SearchHit = tuple[str, dict, float]


def dense_search(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    top_k: int = 50,
) -> list[SearchHit]:
    """Search the Qdrant collection using the dense vector index.

    Args:
        client:          Connected QdrantClient instance.
        collection_name: Name of the target Qdrant collection.
        dense_vector:    Query embedding (1024-dim float32 list).
        top_k:           Number of candidates to retrieve.

    Returns:
        List of (chunk_id, payload, score) tuples ordered by descending
        cosine similarity, length ≤ top_k.
    """
    hits = client.search(
        collection_name=collection_name,
        query_vector=NamedVector(name="dense", vector=dense_vector),
        limit=top_k,
        with_payload=True,
    )
    return [
        (str(hit.id), hit.payload or {}, hit.score)
        for hit in hits
    ]
